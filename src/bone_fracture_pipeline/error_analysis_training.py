"""Train-only inference for Stage 2, sharing the guarded Stage 1 capture path."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from bone_fracture_audit.audit import sha256_file
from bone_fracture_pipeline.error_analysis import (
    ErrorAnalysisError, ValidationSample, _capture_validation, _diagnostic_fingerprint,
    _load_diagnostic_samples, _validate_diagnostic_rows, _verify_diagnostic_lineage,
    _write_diagnostic_snapshot, _write_json, load_validation_samples, read_prediction_export, validation_fingerprint,
)
from bone_fracture_pipeline.experiment_runner import resolve_experiment_config


def verify_sample_export(rows: Sequence[Mapping[str, object]], samples: Sequence[ValidationSample], *, split: str) -> None:
    _validate_diagnostic_rows(rows, split=split)
    by_id = {sample.sample_id: sample for sample in samples}
    if len(rows) != len(samples) or {row["sample_id"] for row in rows} != set(by_id):
        raise ErrorAnalysisError("The export does not contain exactly the expected split identities.")
    for row in rows:
        sample = by_id[row["sample_id"]]
        expected_targets = json.loads(json.dumps([asdict(box) for box in sample.ground_truth]))
        if (row["image_sha256"], row["label_sha256"], row["width"], row["height"], row["ground_truth"], row["image"]) != (
            sample.image_sha256, sample.label_sha256, sample.width, sample.height, expected_targets,
            f"{split}/images/{sample.image_path.name}",
        ):
            raise ErrorAnalysisError(f"Export content or source identity changed: {sample.sample_id}")


# A tracked reference is not enough: require the local prediction bytes and source labels to match.
def read_stage1_validation(project_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    evidence = json.loads((project_root / "docs/evidence/error_analysis/stage1_validation.json").read_text(encoding="utf-8"))
    if evidence["dataset"]["root"] != "data/prepared/v3_detection_clahe":
        raise ErrorAnalysisError("Stage 2 requires the canonical six-class CLAHE diagnostic source.")
    path = (project_root / evidence["prediction_export"]["path"]).resolve()
    if not path.is_relative_to(project_root / "outputs/error_analysis/stage1") or not path.is_file():
        raise ErrorAnalysisError("The referenced local Stage 1 prediction export is missing or outside Stage 1.")
    if sha256_file(path) != evidence["prediction_export"]["sha256"]:
        raise ErrorAnalysisError("The local Stage 1 prediction export failed its recorded hash check.")
    if evidence["validation_passed"] or evidence["official_validation"]["comparison"]["absolute_tolerance"] != 0.00005:
        raise ErrorAnalysisError("The known failed Stage 1 reproduction status/tolerance must remain unchanged.")
    rows = read_prediction_export(path)
    root = project_root / evidence["dataset"]["root"]
    samples = load_validation_samples(root)
    verify_sample_export(rows, samples, split="valid")
    if validation_fingerprint(root, samples) != evidence["dataset"]["validation_fingerprint_after"]:
        raise ErrorAnalysisError("Canonical validation content no longer matches Stage 1 provenance.")
    checkpoint = (project_root / evidence["checkpoint"]["path"]).resolve()
    if checkpoint != project_root / "outputs/training/official/D_seed42/weights/best.pt" or sha256_file(checkpoint) != evidence["checkpoint"]["sha256"]:
        raise ErrorAnalysisError("Stage 2 checkpoint identity differs from the preserved Stage 1 source.")
    if (len(rows), sum(len(row["predictions"]) for row in rows)) != (348, 1755):
        raise ErrorAnalysisError("Stage 1 primary export counts changed.")
    return rows, evidence


def export_training_predictions(project_root: Path, output: Path, stage1: Mapping[str, object]) -> dict[str, object]:
    project_root, output = project_root.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(project_root / "outputs/error_analysis/stage2"):
        raise ErrorAnalysisError("Training inference requires a fresh directory under outputs/error_analysis/stage2.")
    run = project_root / "outputs/training/official/D_seed42"
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    resolution = resolve_experiment_config(project_root, "D")
    if resolution["resolved_digest"] != manifest["configuration"]["resolved_digest"]:
        raise ErrorAnalysisError("Frozen D configuration differs from its official manifest.")
    root = project_root / "data/prepared/v3_detection_clahe"
    checkpoint = project_root / stage1["checkpoint"]["path"]
    if sha256_file(checkpoint) != stage1["checkpoint"]["sha256"]:
        raise ErrorAnalysisError("D checkpoint differs from the Stage 1 diagnostic source.")
    samples = _load_diagnostic_samples(root, split="train", expected_counts=(1211, 698, 607))
    _verify_diagnostic_lineage(project_root / "outputs/phase2/phase2d/preprocessing_manifest.csv", samples, split="train")
    fingerprint = _diagnostic_fingerprint(root, samples, split="train")
    output.mkdir(parents=True)
    runtime = output / "runtime"
    runtime.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(runtime)
    import torch
    import ultralytics
    from ultralytics.utils.torch_utils import init_seeds

    if (torch.__version__, ultralytics.__version__) != ("2.6.0+cu126", "8.4.155") or not torch.cuda.is_available():
        raise ErrorAnalysisError("Training inference requires the preserved CUDA framework environment.")
    # Only output locations change. Framework val mode reads a copy of train, with augmentation disabled.
    settings = dict(stage1["prediction_export"]["settings"])
    settings.update(project=str(output), name="framework_evaluation")
    expected = {"imgsz": 640, "batch": 16, "conf": 0.001, "iou": 0.7, "max_det": 300,
                "agnostic_nms": False, "single_cls": False, "augment": False, "quantize": None,
                "amp": False, "classes": None, "rect": True, "split": "val"}
    if any(settings[key] != value for key, value in expected.items()):
        raise ErrorAnalysisError("Training inference must retain the primary Stage 1 inference settings.")
    dataset_yaml = _write_diagnostic_snapshot(samples, output / "train_snapshot", split="train")
    init_seeds(42, deterministic=True)
    native, rows, checks = _capture_validation(checkpoint, dataset_yaml, samples, settings, source_split="train")
    rows = json.loads(json.dumps(rows, allow_nan=False))
    counts = _validate_diagnostic_rows(rows, split="train")
    verify_sample_export(rows, samples, split="train")
    if any(prediction["confidence"] <= 0.001 for row in rows for prediction in row["predictions"]):
        raise ErrorAnalysisError("A training candidate violates the native export confidence floor.")
    path = output / "training_predictions.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    with path.open(encoding="utf-8") as handle:
        recovered = [json.loads(line) for line in handle]
    if recovered != rows:
        raise ErrorAnalysisError("Training export did not survive a lossless JSON roundtrip.")
    if fingerprint != _diagnostic_fingerprint(root, samples, split="train") or sha256_file(checkpoint) != stage1["checkpoint"]["sha256"]:
        raise ErrorAnalysisError("Canonical training inputs or D checkpoint changed during inference.")
    summary = {"split": "train", "checkpoint": stage1["checkpoint"], "dataset_root": "data/prepared/v3_detection_clahe",
               "training_fingerprint_before": fingerprint, "training_fingerprint_after": fingerprint,
               "fingerprint_scope": "data.yaml plus train images and labels only", "source_unchanged": True,
               "path": path.relative_to(project_root).as_posix(), "sha256": sha256_file(path),
               "counts": counts, "settings": settings, "loader_checks": checks,
               "framework": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                             "gpu": torch.cuda.get_device_name(0), "precision": "FP32", "model_fusion": True,
                             "seed": 42, "deterministic": True},
               "native_training_split_metrics": native, "training_or_gradient_updates_performed": False,
               "test_files_opened": 0, "official_results_replaced": False}
    _write_json(output / "training_export_summary.json", summary)
    return summary


def read_training_export(project_root: Path, directory: Path, stage1: Mapping[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    directory = directory.resolve()
    if not directory.is_relative_to(project_root / "outputs/error_analysis/stage2"):
        raise ErrorAnalysisError("A reusable train export must remain under Stage 2 outputs.")
    metadata = json.loads((directory / "training_export_summary.json").read_text(encoding="utf-8"))
    path = directory / "training_predictions.jsonl"
    if (metadata["split"] != "train" or metadata["dataset_root"] != "data/prepared/v3_detection_clahe"
            or metadata["checkpoint"]["sha256"] != stage1["checkpoint"]["sha256"]
            or sha256_file(path) != metadata["sha256"] or metadata["training_or_gradient_updates_performed"]):
        raise ErrorAnalysisError("The reusable training export failed provenance checks.")
    if any(metadata["settings"][key] != value for key, value in stage1["prediction_export"]["settings"].items()
           if key not in ("project", "name")):
        raise ErrorAnalysisError("Reusable train and validation exports have different inference settings.")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    samples = _load_diagnostic_samples(project_root / metadata["dataset_root"], split="train", expected_counts=(1211, 698, 607))
    verify_sample_export(rows, samples, split="train")
    if _diagnostic_fingerprint(project_root / metadata["dataset_root"], samples, split="train") != metadata["training_fingerprint_after"]:
        raise ErrorAnalysisError("Training source no longer matches its reusable export.")
    if _validate_diagnostic_rows(rows, split="train") != metadata["counts"]:
        raise ErrorAnalysisError("Training export counts no longer match its summary.")
    return rows, metadata
