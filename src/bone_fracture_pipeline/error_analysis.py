from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Mapping, Sequence

import yaml
from PIL import Image

from bone_fracture_audit.audit import _read_dataset_config, match_images_and_labels, sha256_file
from bone_fracture_audit.yolo import parse_yolo_label
from bone_fracture_pipeline.detection_evaluation import GroundTruth, Prediction, box_iou
from bone_fracture_pipeline.experiment_runner import resolve_experiment_config
from bone_fracture_pipeline.prepare_dataset import CLASS_NAMES


class ErrorAnalysisError(ValueError):
    """Raised when Stage 1 cannot preserve its validation or provenance contract."""


@dataclass(frozen=True)
class ValidationSample:
    sample_id: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    ground_truth: tuple[GroundTruth, ...]
    image_sha256: str
    label_sha256: str


# Restrict the split before opening anything; full-dataset preflight would read held-out files.
def load_validation_samples(
    root: Path, *, split: str = "valid", expected_counts: tuple[int, int, int] = (348, 204, 175)
) -> tuple[ValidationSample, ...]:
    if split != "valid":
        raise ErrorAnalysisError("Stage 1 accepts only the validation split 'valid'.")
    return _load_diagnostic_samples(root, split="valid", expected_counts=expected_counts)


def _check_diagnostic_split(split: str) -> None:
    if split not in ("train", "valid"):
        raise ErrorAnalysisError("Diagnostics accept only train or valid; held-out access is prohibited.")


# Stage 2 adds explicit train routing while the public Stage 1 wrapper stays validation-only.
def _load_diagnostic_samples(
    root: Path, *, split: str, expected_counts: tuple[int, int, int]
) -> tuple[ValidationSample, ...]:
    _check_diagnostic_split(split)
    root = root.resolve()
    if _read_dataset_config(root / "data.yaml") != CLASS_NAMES:
        raise ErrorAnalysisError("Stage 1 requires the six original class IDs in canonical order.")
    for directory in (root / split, root / split / "images", root / split / "labels"):
        if directory.resolve() != directory or not directory.is_dir():
            raise ErrorAnalysisError("Validation directories must be real directories inside the dataset.")
    matching = match_images_and_labels(root / split / "images", root / split / "labels")
    if any(matching[key] for key in (
        "images_without_labels", "labels_without_images", "ambiguous_image_stems", "ambiguous_label_stems"
    )):
        raise ErrorAnalysisError("Validation images and labels must have unique one-to-one identities.")
    samples = []
    for image_path, label_path in matching["pairs"]:
        validate_image_paths([image_path], root / split / "images")
        validate_image_paths([label_path], root / split / "labels")
        parsed = parse_yolo_label(label_path, set(range(len(CLASS_NAMES))))
        if parsed.issues or any(box.annotation_type != "box" for box in parsed.annotations):
            raise ErrorAnalysisError(f"Invalid validation detection label: {label_path}")
        if parsed.is_empty and label_path.stat().st_size:
            raise ErrorAnalysisError("Canonical empty labels must remain zero-byte files.")
        with Image.open(image_path) as image:
            width, height = image.size
        targets = tuple(
            GroundTruth(f"gt/{box.line_number:04d}", box.class_id, (
                (box.x_center - box.width / 2) * width,
                (box.y_center - box.height / 2) * height,
                (box.x_center + box.width / 2) * width,
                (box.y_center + box.height / 2) * height,
            ))
            for box in parsed.annotations
        )
        samples.append(ValidationSample(
            f"{split}/{image_path.stem}", image_path, label_path, width, height, targets,
            sha256_file(image_path), sha256_file(label_path),
        ))
    counts = (len(samples), sum(len(sample.ground_truth) for sample in samples),
              sum(not sample.ground_truth for sample in samples))
    if counts != expected_counts:
        raise ErrorAnalysisError(f"Validation image/annotation/empty-label counts changed: {counts}.")
    return tuple(sorted(samples, key=lambda sample: sample.sample_id))


def validate_image_paths(paths: Sequence[Path], allowed_directory: Path) -> None:
    allowed_directory = allowed_directory.absolute()
    for path in paths:
        # Resolving each path also rejects links or junctions escaping into another split.
        if not path.resolve().is_relative_to(allowed_directory):
            raise ErrorAnalysisError("A loader path escaped its validation-only directory.")


def validation_fingerprint(root: Path, samples: Sequence[ValidationSample]) -> str:
    return _diagnostic_fingerprint(root, samples, split="valid")


def _diagnostic_fingerprint(root: Path, samples: Sequence[ValidationSample], *, split: str) -> str:
    _check_diagnostic_split(split)
    for sample in samples:
        validate_image_paths([sample.image_path], root / split / "images")
        validate_image_paths([sample.label_path], root / split / "labels")
    paths = [root / "data.yaml", *(path for sample in samples for path in (sample.image_path, sample.label_path))]
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda path: (path.relative_to(root).as_posix().casefold(), path.as_posix())):
        digest.update(f"{path.relative_to(root).as_posix()}\t{sha256_file(path)}\n".encode("utf-8"))
    return digest.hexdigest()


# Compare validation hashes with the original CLAHE build without opening any test file.
def verify_preprocessing_lineage(manifest_path: Path, samples: Sequence[ValidationSample]) -> None:
    _verify_diagnostic_lineage(manifest_path, samples, split="valid")


def _verify_diagnostic_lineage(manifest_path: Path, samples: Sequence[ValidationSample], *, split: str) -> None:
    _check_diagnostic_split(split)
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    by_name = {row["processed_filename"]: row for row in rows}
    if len(by_name) != len(rows) or set(by_name) != {sample.image_path.name for sample in samples}:
        raise ErrorAnalysisError("Validation identities differ from the original CLAHE manifest.")
    for sample in samples:
        row = by_name[sample.image_path.name]
        if (row["processed_image_sha256"], row["processed_label_sha256"]) != (
            sample.image_sha256, sample.label_sha256
        ):
            raise ErrorAnalysisError(f"Validation content drifted since the CLAHE build: {sample.sample_id}")


# Use copies so Ultralytics can create its cache without changing the canonical dataset.
def write_validation_snapshot(samples: Sequence[ValidationSample], destination: Path) -> Path:
    return _write_diagnostic_snapshot(samples, destination, split="valid")


def _write_diagnostic_snapshot(samples: Sequence[ValidationSample], destination: Path, *, split: str) -> Path:
    _check_diagnostic_split(split)
    if destination.exists():
        raise ErrorAnalysisError("A validation snapshot already exists and will not be overwritten.")
    for sample in samples:
        source_root = sample.image_path.parent.parent.parent
        if sample.image_path.parent.parent.name != split or not sample.sample_id.startswith(f"{split}/"):
            raise ErrorAnalysisError("A snapshot source must belong to its explicit diagnostic split.")
        validate_image_paths([sample.image_path], source_root / split / "images")
        validate_image_paths([sample.label_path], source_root / split / "labels")
        if destination.resolve().is_relative_to(source_root):
            raise ErrorAnalysisError("The evaluation snapshot must be outside the source dataset.")
    (destination / split / "images").mkdir(parents=True)
    (destination / split / "labels").mkdir()
    for sample in samples:
        image = destination / split / "images" / sample.image_path.name
        label = destination / split / "labels" / sample.label_path.name
        shutil.copyfile(sample.image_path, image)
        shutil.copyfile(sample.label_path, label)
        if (sha256_file(image), sha256_file(label)) != (sample.image_sha256, sample.label_sha256):
            raise ErrorAnalysisError("A validation snapshot copy failed its hash check.")
    # Both framework keys route to this one evaluation-only copy; no training is invoked.
    dataset_yaml = destination / "data.yaml"
    dataset_yaml.write_text(yaml.safe_dump({
        "path": str(destination.resolve()), "train": f"{split}/images", "val": f"{split}/images",
        "names": dict(enumerate(CLASS_NAMES)),
    }, sort_keys=False), encoding="utf-8")
    return dataset_yaml


def best_official_row(path: Path) -> dict[str, float | int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    if not rows or any(not math.isfinite(value) for row in rows for value in row.values()):
        raise ErrorAnalysisError("Official training results contain missing or non-finite values.")
    row = max(rows, key=lambda row: (row["metrics/mAP50-95(B)"], -row["epoch"]))
    return {"epoch": int(row["epoch"]), **{key: row[key] for key in (
        "metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"
    )}}


def compare_official_metrics(expected: Mapping[str, float | int], observed: Mapping[str, float],
                             *, tolerance: float = 0.00005) -> dict[str, object]:
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 0.00005:
        raise ErrorAnalysisError("Metric tolerance must remain between zero and the Stage 1 bound 0.00005.")
    differences = {}
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        value = float(observed[key])
        if not math.isfinite(value):
            raise ErrorAnalysisError("Framework validation produced a non-finite metric.")
        difference = value - float(expected[key])
        differences[key] = {"official": expected[key], "observed": value,
                            "signed_difference": difference, "absolute_difference": abs(difference),
                            "passed": abs(difference) <= tolerance}
    return {
        "passed": all(item["passed"] for item in differences.values()),
        "absolute_tolerance": tolerance,
        "tolerance_reason": (
            "Chosen before evaluation: 5e-5 absolute AP allows the official five-decimal rounding "
            "(at most 5e-6) plus small checkpoint FP16 serialization/FP32 fusion differences. "
            "It is a tight reproduction check, not a statistical equivalence test."
        ),
        "metrics": differences,
    }


def validate_export_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return _validate_diagnostic_rows(rows, split="valid")


def _validate_diagnostic_rows(rows: Sequence[Mapping[str, object]], *, split: str) -> dict[str, int]:
    _check_diagnostic_split(split)
    sample_ids = set()
    total_predictions = total_targets = empty_labels = zero_predictions = 0
    for row in rows:
        if row["split"] != split or not str(row["sample_id"]).startswith(f"{split}/"):
            raise ErrorAnalysisError("Prediction exports must contain validation samples only.")
        if row["sample_id"] in sample_ids:
            raise ErrorAnalysisError("An exported sample appears more than once.")
        sample_ids.add(row["sample_id"])
        width, height = row["width"], row["height"]
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ErrorAnalysisError("Exported original image dimensions must be positive integers.")
        for key in ("predictions", "ground_truth"):
            identifiers = set()
            for item in row[key]:
                if not item["box_id"] or item["box_id"] in identifiers:
                    raise ErrorAnalysisError("Exported box IDs are missing or duplicated.")
                identifiers.add(item["box_id"])
                if type(item["class_id"]) is not int or item["class_id"] not in range(len(CLASS_NAMES)):
                    raise ErrorAnalysisError("An exported class ID is outside the original six classes.")
                box = item["xyxy"]
                box_iou(box, box)
                if not (0 <= box[0] <= box[2] <= width and 0 <= box[1] <= box[3] <= height):
                    raise ErrorAnalysisError("An exported box is outside original-image pixel bounds.")
                if key == "predictions" and not (math.isfinite(item["confidence"]) and 0 <= item["confidence"] <= 1):
                    raise ErrorAnalysisError("An exported prediction confidence is invalid.")
        total_predictions += len(row["predictions"])
        total_targets += len(row["ground_truth"])
        empty_labels += int(not row["ground_truth"])
        zero_predictions += int(not row["predictions"])
    return {"images": len(rows), "predictions": total_predictions, "annotations": total_targets,
            "empty_label_images": empty_labels, "images_without_predictions": zero_predictions}


def read_prediction_export(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    validate_export_rows(rows)
    return rows


def _keep_unfused(model, verbose=True):
    # Epoch validation uses the unfused EMA. Disable only this instance's automatic backend fusion.
    return model


def _capture_validation(checkpoint: Path, dataset_yaml: Path, samples: Sequence[ValidationSample],
                        settings: Mapping[str, object], *, fuse_model: bool = True, source_split: str = "valid"
                        ) -> tuple[dict[str, float], list[dict[str, object]], dict[str, object]]:
    _check_diagnostic_split(source_split)
    if any(not sample.sample_id.startswith(f"{source_split}/") for sample in samples):
        raise ErrorAnalysisError("Capture sample IDs disagree with their explicit source split.")
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import ops

    by_name = {sample.image_path.name: sample for sample in samples}
    captured: dict[str, dict[str, object]] = {}
    checks: dict[str, object] = {}

    class CapturingValidator(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            # Verify actual precision and fusion instead of recording requested flags only.
            parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.model.parameters()})
            batch_norm_layers = sum(isinstance(module, torch.nn.BatchNorm2d) for module in model.model.modules())
            if parameter_dtypes != ["torch.float32"] or model.fp16:
                raise ErrorAnalysisError("The evaluation backend did not preserve FP32 inference.")
            if (batch_norm_layers == 0) != fuse_model:
                raise ErrorAnalysisError("The evaluation backend did not preserve the requested fusion behavior.")
            if torch.is_grad_enabled():
                raise ErrorAnalysisError("Diagnostic capture must run without gradient tracking.")
            checks.update({"parameter_dtypes": parameter_dtypes, "batch_norm_layers": batch_norm_layers,
                           "backend_fp16": bool(model.fp16), "gradient_tracking_enabled": torch.is_grad_enabled()})

        def get_dataloader(self, dataset_path, batch_size):
            allowed = dataset_yaml.parent / source_split / "images"
            if Path(dataset_path).resolve() != allowed.resolve() or self.args.split != "val":
                raise ErrorAnalysisError("Framework evaluation attempted a non-validation route.")
            loader = super().get_dataloader(dataset_path, batch_size)
            paths = [Path(path) for path in loader.dataset.im_files]
            validate_image_paths(paths, allowed)
            if len(paths) != len(samples) or {path.name for path in paths} != set(by_name):
                raise ErrorAnalysisError("The framework loader did not preserve all validation identities.")
            if loader.dataset.augment or not loader.dataset.rect or batch_size != settings["batch"]:
                raise ErrorAnalysisError("The framework validation transform or batching contract changed.")
            checks.update({"validation_images_loaded": len(paths) if source_split == "valid" else 0,
                           "train_images_loaded": len(paths) if source_split == "train" else 0,
                           "source_split": source_split, "test_images_loaded": 0,
                           "rectangular_batches": bool(loader.dataset.rect), "augmentation": False,
                           "batch": batch_size, "maximum_ground_truth_roundtrip_error_px": 0.0})
            return loader

        def update_metrics(self, preds, batch):
            # AP uses the untouched framework predictions in padded input coordinates.
            super().update_metrics(preds, batch)
            for index, prediction in enumerate(preds):
                prepared = self._prepare_batch(index, batch)
                sample = by_name[Path(prepared["im_file"]).name]
                if sample.sample_id in captured or tuple(prepared["ori_shape"]) != (sample.height, sample.width):
                    raise ErrorAnalysisError("Framework sample identity or original dimensions changed.")
                # scale_preds clones the boxes, so export cannot alter framework AP calculation.
                original = self.scale_preds(prediction, prepared)
                values = sorted(zip(original["conf"].tolist(), original["cls"].tolist(),
                                    original["bboxes"].tolist()), key=lambda item: (-item[0], item[1], *item[2]))
                predictions = [asdict(Prediction(f"pred/{number:06d}", int(class_id), tuple(box), float(conf)))
                               for number, (conf, class_id, box) in enumerate(values)]
                # Compare transformed framework targets with the preserved source labels.
                scaled_targets = ops.scale_boxes(prepared["imgsz"], prepared["bboxes"].clone(),
                                                  prepared["ori_shape"], prepared["ratio_pad"]).tolist()
                actual_targets = sorted(zip(prepared["cls"].tolist(), scaled_targets))
                expected_targets = sorted((box.class_id, list(box.xyxy)) for box in sample.ground_truth)
                if len(actual_targets) != len(expected_targets):
                    raise ErrorAnalysisError("The framework dropped or added a ground truth row.")
                for (actual_class, actual_box), (expected_class, expected_box) in zip(actual_targets, expected_targets):
                    error = max(abs(a - b) for a, b in zip(actual_box, expected_box))
                    if actual_class != expected_class or error > 0.05:
                        raise ErrorAnalysisError("Framework ground truth geometry changed beyond subpixel rounding.")
                    checks["maximum_ground_truth_roundtrip_error_px"] = max(
                        checks["maximum_ground_truth_roundtrip_error_px"], error)
                captured[sample.sample_id] = {
                    "sample_id": sample.sample_id, "split": source_split,
                    "image": f"{source_split}/images/{sample.image_path.name}",
                    "width": sample.width, "height": sample.height,
                    "image_sha256": sample.image_sha256, "label_sha256": sample.label_sha256,
                    "predictions": predictions, "ground_truth": [asdict(box) for box in sample.ground_truth],
                }

    model = YOLO(str(checkpoint))
    if model.names != dict(enumerate(CLASS_NAMES)) or model.task != "detect":
        raise ErrorAnalysisError("The checkpoint does not expose the six-class detection task.")
    if not fuse_model:
        model.model.fuse = MethodType(_keep_unfused, model.model)
    metrics = model.val(validator=CapturingValidator, data=str(dataset_yaml), **settings)
    if set(captured) != {sample.sample_id for sample in samples}:
        raise ErrorAnalysisError("Prediction export did not cover every validation image exactly once.")
    values = {key: float(value) for key, value in metrics.results_dict.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ErrorAnalysisError("Framework validation metrics are non-finite.")
    checks["per_class_ap"] = [
        {"class_id": int(class_id), "ap50": float(metrics.box.all_ap[index, 0]),
         "ap50_95": float(metrics.box.all_ap[index].mean())}
        for index, class_id in enumerate(metrics.box.ap_class_index)
    ]
    return values, [captured[key] for key in sorted(captured)], checks


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_stage1(project_root: Path, output: Path, *, fuse_model: bool = True) -> dict[str, object]:
    project_root, output = project_root.resolve(), output.resolve()
    if not output.is_relative_to(project_root / "outputs/error_analysis") or output.exists():
        raise ErrorAnalysisError("Use a fresh run directory below outputs/error_analysis; existing evidence is preserved.")
    resolution = resolve_experiment_config(project_root, "D")
    run = project_root / "outputs/training/official/D_seed42"
    checkpoint = run / "weights/best.pt"
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or resolution["resolved_digest"] != manifest["configuration"]["resolved_digest"]:
        raise ErrorAnalysisError("Current Experiment D configuration differs from the official run.")
    dataset = resolution["resolved"]["dataset"]
    if dataset["root"] != "data/prepared/v3_detection_clahe":
        raise ErrorAnalysisError("Stage 1 requires the canonical six-class CLAHE condition.")
    root = project_root / dataset["root"]
    samples = load_validation_samples(root)
    fingerprint_before = validation_fingerprint(root, samples)
    lineage_path = project_root / "outputs/phase2/phase2d/preprocessing_manifest.csv"
    verify_preprocessing_lineage(lineage_path, samples)
    expected = best_official_row(run / "results.csv")
    if expected["epoch"] != 57:
        raise ErrorAnalysisError("The official CSV no longer selects Experiment D epoch 57.")
    checkpoint_hash = sha256_file(checkpoint)

    # Create the configuration directory before importing Ultralytics to prevent fallback writes.
    output.mkdir(parents=True)
    runtime = output / "runtime"
    runtime.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(runtime)
    import torch
    import ultralytics
    from ultralytics.utils.torch_utils import init_seeds

    if ultralytics.__version__ != "8.4.155" or torch.__version__ != "2.6.0+cu126" or not torch.cuda.is_available():
        raise ErrorAnalysisError("The recorded Ultralytics/PyTorch CUDA environment is required; no dependencies were changed.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if any(float(saved["train_metrics"][key]) != expected[key] for key in expected if key != "epoch"):
        raise ErrorAnalysisError("Checkpoint training metrics do not correspond to the official best CSV row.")
    args = manifest["applied_trainer_config"]["actual_args"]
    for key in ("model", "seed", "deterministic", "imgsz", "batch", "amp", "conf", "iou", "max_det",
                "agnostic_nms", "augment", "single_cls", "quantize"):
        if saved["train_args"][key] != args[key]:
            raise ErrorAnalysisError(f"Checkpoint and recorded trainer disagree on {key}.")
    checkpoint_dtypes = sorted({str(parameter.dtype) for parameter in saved["model"].parameters()})
    # This legacy checkpoint stores explicit model multipliers rather than a YAML filename/scale key.
    model_definition = {key: saved["model"].yaml[key] for key in ("nc", "depth_multiple", "width_multiple")}
    model_definition["training_model"] = args["model"]
    if Path(args["model"]).name != "yolov8s.pt" or model_definition["nc"] != 6:
        raise ErrorAnalysisError("The official checkpoint must remain the six-class YOLOv8s model.")
    stored_epoch = int(saved["epoch"])
    del saved
    dataset_yaml = write_validation_snapshot(samples, output / "validation_snapshot")
    settings = {
        "imgsz": args["imgsz"], "batch": args["batch"] * 2, "device": args["device"], "workers": 0,
        "conf": 0.001, "iou": args["iou"], "max_det": args["max_det"], "rect": True,
        "agnostic_nms": False, "single_cls": False, "classes": None, "augment": False,
        "amp": False, "quantize": None, "cache": False, "plots": False,
        "save": False, "save_json": False, "save_txt": False, "verbose": False,
        "split": "val", "project": str(output), "name": "framework_validation", "exist_ok": False,
    }
    init_seeds(42, deterministic=True)
    observed, rows, loader_checks = _capture_validation(checkpoint, dataset_yaml, samples, settings, fuse_model=fuse_model)
    comparison = compare_official_metrics(expected, observed)
    counts = validate_export_rows(rows)
    if (counts["images"], counts["annotations"], counts["empty_label_images"]) != (348, 204, 175):
        raise ErrorAnalysisError("Exported ground truth counts differ from the validation contract.")
    if any(prediction["confidence"] <= settings["conf"] for row in rows for prediction in row["predictions"]):
        raise ErrorAnalysisError("An exported prediction violates the framework's strict NMS confidence floor.")
    export_path = output / "validation_predictions.jsonl"
    export_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    if read_prediction_export(export_path) != json.loads(json.dumps(rows)):
        raise ErrorAnalysisError("Prediction export failed its lossless JSON roundtrip.")
    fingerprint_after = validation_fingerprint(root, samples)
    if fingerprint_before != fingerprint_after or sha256_file(checkpoint) != checkpoint_hash:
        raise ErrorAnalysisError("Canonical validation data or the checkpoint changed during Stage 1.")
    source_files = {
        name: {"sha256": sha256_file(Path(importlib.import_module(name).__file__))}
        for name in ("ultralytics.engine.validator", "ultralytics.engine.trainer", "ultralytics.models.yolo.detect.val",
                     "ultralytics.models.yolo.detect.train", "ultralytics.nn.backends.pytorch",
                     "ultralytics.data.build", "ultralytics.utils.nms")
    }
    summary = {
        "schema_version": 1, "stage": 1, "experiment": "D", "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed" if comparison["passed"] else "metric_reproduction_failed",
        "checkpoint": {"path": checkpoint.relative_to(project_root).as_posix(), "sha256": checkpoint_hash,
                       "best_epoch_from_csv": expected["epoch"], "stored_epoch": stored_epoch,
                       "epoch_note": "Optimizer stripping removes epoch; saved train_metrics corroborate CSV epoch 57.",
                       "serialized_parameter_dtypes": checkpoint_dtypes, "model_definition": model_definition},
        "dataset": {"root": dataset["root"], "historical_full_fingerprint": dataset["expected_fingerprint"],
                    "full_fingerprint_recomputed": False, "validation_fingerprint_before": fingerprint_before,
                    "validation_fingerprint_after": fingerprint_after, "fingerprint_scope": "data.yaml plus valid images/labels",
                    "validation_files_matched_original_clahe_manifest": True, "source_unchanged": True,
                    "class_names": list(CLASS_NAMES), "test_files_opened": 0},
        "official_validation": {"reference": (run / "results.csv").relative_to(project_root).as_posix(),
                                "best_row": expected, "observed": observed, "comparison": comparison},
        "prediction_export": {"path": export_path.relative_to(project_root).as_posix(), "sha256": sha256_file(export_path),
                              **counts, "settings": settings, "coordinate_space": "original-image pixels, xyxy",
                              "class_ids_preserved": True, "nms_multi_label": True,
                              "images_at_max_det": sum(len(row["predictions"]) == settings["max_det"] for row in rows),
                              "purpose": "Low-confidence post-NMS candidates for later error analysis",
                              "ap_relation": "Captured from the same framework validation pass; fixed-threshold matching is separate from AP."},
        "framework": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0), "seed": 42, "deterministic": True,
                      "precision": "FP32 inference from the preserved checkpoint", "model_fusion": fuse_model,
                      "source_files": source_files},
        "loader_checks": loader_checks,
        "evaluation_settings_reason": (
            "The pinned detection trainer doubles validation batch size and forces rectangular validation batches. "
            "Validation NMS defaults conf=None to 0.001, uses multi_label=True, class-aware NMS, IoU 0.7 and max_det 300. "
            "workers=0 changes loading concurrency only. A validation-only copy contains caches; no canonical split is rewritten."
        ),
        "matching": {"algorithm": "confidence-first greedy one-to-one", "prediction_order": "descending confidence then stable box ID",
                     "target_choice": "highest eligible IoU then stable ground truth ID", "threshold_comparison": ">=",
                     "class_aware_default": True, "class_agnostic_supported": True,
                     "wrong_class": "unmatched prediction is FP; unmatched target remains FN",
                     "duplicates": "at most one prediction can claim a target; remaining predictions are FP",
                     "undefined_metric_convention": "zero when denominator is zero",
                     "ap_equivalence_claimed": False},
        "limitations": [
            "Candidates below confidence 0.001 or removed by NMS/max_det are unavailable; this is not a raw-head export.",
            "Confidence-first diagnostic matching differs from Ultralytics' IoU-first AP matching on conflicts and ties.",
            "The official AP reproduction check must pass before these candidates can be treated as an equivalent evaluation.",
            "Standalone validation calculates detection metrics, not trainer-integrated validation losses.",
            "Checkpoints serialize FP16 weights; the original full-precision epoch EMA cannot be recovered from this artifact.",
            "Repeatability requires a separate fresh-directory run; it is not guaranteed across GPU/framework versions.",
            "Empty labels are unannotated cases, not clinically verified healthy images.",
        ],
    }
    _write_json(output / "stage1_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce Experiment D validation and export Stage 1 candidates.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/error_analysis/stage1/D_seed42"))
    parser.add_argument("--unfused", action="store_true", help="Retain batch normalization as in trainer epoch validation.")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else args.project_root / args.output
    summary = run_stage1(args.project_root, output, fuse_model=not args.unfused)
    print(json.dumps({"status": summary["status"], "comparison": summary["official_validation"]["comparison"],
                      "export_counts": {key: summary["prediction_export"][key] for key in
                                        ("images", "predictions", "annotations", "empty_label_images")}}, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
