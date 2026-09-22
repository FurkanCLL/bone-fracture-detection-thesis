from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


DIAGNOSTIC_DIRECTORY = Path("outputs/phase2/phase2f/validation_diagnostics")
CANONICAL_EVIDENCE_DIRECTORY = Path("docs/evidence/phase2f")
LOSS_NAMES = ("box_loss", "cls_loss", "dfl_loss")


class ValidationDiagnosticError(RuntimeError):
    """Raised when the validation diagnostic cannot reproduce the frozen path."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value: object) -> object:
    """Replace non-finite floats with explicit strings for strict JSON evidence."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "infinity" if value > 0 else "-infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_tensor(tensor: Any) -> dict[str, object]:
    """Return finite-value evidence without serializing a potentially large tensor."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("summarize_tensor expects a torch.Tensor")
    detached = tensor.detach()
    floating = detached.is_floating_point() or detached.is_complex()
    finite = torch.isfinite(detached) if floating else torch.ones_like(detached, dtype=torch.bool)
    finite_count = int(finite.sum().item())
    total = detached.numel()
    summary: dict[str, object] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "values": total,
        "finite_values": finite_count,
        "nonfinite_values": total - finite_count,
        "all_finite": finite_count == total,
    }
    if floating:
        summary.update(
            {
                "nan_values": int(torch.isnan(detached).sum().item()),
                "positive_infinity_values": int(torch.isposinf(detached).sum().item()),
                "negative_infinity_values": int(torch.isneginf(detached).sum().item()),
            }
        )
    if total and finite_count:
        values = detached[finite]
        if values.is_complex():
            values = values.abs()
        values = values.float()
        summary.update(
            {
                "finite_min": float(values.min().item()),
                "finite_max": float(values.max().item()),
                "finite_mean": float(values.mean().item()),
            }
        )
    return summary


def summarize_prediction_tensors(predictions: object) -> dict[str, object]:
    """Walk nested model output and report each tensor's finite state."""
    import torch

    summaries: dict[str, dict[str, object]] = {}

    def visit(value: object, path: str) -> None:
        if isinstance(value, torch.Tensor):
            summaries[path] = summarize_tensor(value)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(predictions, "predictions")
    nonfinite_paths = [path for path, summary in summaries.items() if not summary["all_finite"]]
    return {
        "all_finite": not nonfinite_paths,
        "nonfinite_paths": nonfinite_paths,
        "tensors": summaries,
    }


def summarize_batch_targets(batch: Mapping[str, object]) -> dict[str, object]:
    """Describe image-level target occupancy and normalized target ranges."""
    import torch

    image_files = [str(path) for path in batch["im_file"]]
    batch_indices = batch["batch_idx"].view(-1).long()
    classes = batch["cls"].view(-1)
    boxes = batch["bboxes"]
    images: list[dict[str, object]] = []
    for image_index, image_file in enumerate(image_files):
        mask = batch_indices == image_index
        image_classes = [int(value) for value in classes[mask].detach().cpu().tolist()]
        images.append(
            {
                "image_index": image_index,
                "image_file": image_file,
                "target_count": len(image_classes),
                "empty_label": not image_classes,
                "class_ids": image_classes,
            }
        )
    class_counts = Counter(int(value) for value in classes.detach().cpu().tolist())
    return {
        "image_count": len(image_files),
        "target_count": int(classes.numel()),
        "empty_image_count": sum(int(item["empty_label"]) for item in images),
        "class_ids": sorted(class_counts),
        "class_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
        "batch_idx": summarize_tensor(batch_indices),
        "classes": summarize_tensor(classes),
        "boxes": summarize_tensor(boxes),
        "box_coordinates_within_zero_one": bool(
            boxes.numel() == 0 or ((boxes >= 0).all() and (boxes <= 1).all()).item()
        ),
        "images": images,
    }


def _slice_batch(batch: Mapping[str, object], image_index: int) -> dict[str, object]:
    """Create a one-image batch while preserving only that image's targets."""
    import torch

    sliced: dict[str, object] = {}
    target_mask = batch["batch_idx"].view(-1).long() == image_index
    image_count = len(batch["im_file"])
    target_keys = {"batch_idx", "cls", "bboxes", "segments", "keypoints", "masks", "obb"}
    for key, value in batch.items():
        if key == "img":
            sliced[key] = value[image_index : image_index + 1]
        elif key == "im_file":
            sliced[key] = [value[image_index]]
        elif isinstance(value, torch.Tensor) and key in target_keys:
            sliced[key] = value[target_mask]
        elif isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == image_count:
            sliced[key] = value[image_index : image_index + 1]
        elif isinstance(value, (list, tuple)) and len(value) == image_count:
            sliced[key] = [value[image_index]]
        else:
            sliced[key] = value
    sliced["batch_idx"] = torch.zeros_like(sliced["batch_idx"])
    return sliced


def _without_targets(batch: Mapping[str, object]) -> dict[str, object]:
    """Keep the images unchanged while replacing the target set with a valid empty set."""
    import torch

    empty = dict(batch)
    target_count = int(batch["batch_idx"].numel())
    target_keys = {"batch_idx", "cls", "bboxes", "segments", "keypoints", "masks", "obb"}
    for key in target_keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == target_count:
            empty[key] = value[:0]
    return empty


def _loss_values(loss_items: Mapping[str, object]) -> dict[str, float]:
    return {key: float(value.detach().float().cpu().item()) for key, value in loss_items.items()}


def _evaluate_batch(model: object, batch: Mapping[str, object], *, amp: bool, device_type: str) -> dict[str, object]:
    import torch

    with torch.inference_mode(), torch.amp.autocast(device_type=device_type, enabled=amp):
        predictions = model(batch["img"])
        total_loss, loss_items = model.loss(batch, predictions)
    values = _loss_values(loss_items)
    total_summary = summarize_tensor(total_loss)
    return {
        "predictions": summarize_prediction_tensors(predictions),
        "losses": values,
        "losses_finite": all(_finite_number(value) for value in values.values()),
        "total_loss": total_summary,
    }


def _trace_detection_head(
    model: object,
    batch: Mapping[str, object],
    *,
    amp: bool,
    device_type: str,
) -> dict[str, object]:
    """Locate the first non-finite leaf operation inside the detection head."""
    import torch

    events: list[dict[str, object]] = []
    handles = []
    head = model.model[-1]

    def make_hook(name: str):
        def hook(_module: object, inputs: object, output: object) -> None:
            input_summary = summarize_prediction_tensors(inputs)
            output_summary = summarize_prediction_tensors(output)
            events.append(
                {
                    "module": name,
                    "input_all_finite": input_summary["all_finite"],
                    "output_all_finite": output_summary["all_finite"],
                    "output_nonfinite_paths": output_summary["nonfinite_paths"],
                    "outputs": output_summary["tensors"],
                }
            )

        return hook

    for name, module in head.named_modules():
        if name and not any(module.children()):
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        with torch.inference_mode(), torch.amp.autocast(device_type=device_type, enabled=amp):
            model(batch["img"])
    finally:
        for handle in handles:
            handle.remove()
    first_nonfinite = next((event["module"] for event in events if not event["output_all_finite"]), None)
    return {"first_nonfinite_module": first_nonfinite, "events": events}


def _diagnose_individual_images(
    model: object,
    batch: Mapping[str, object],
    *,
    amp: bool,
    device_type: str,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for image_index, image_file in enumerate(batch["im_file"]):
        image_batch = _slice_batch(batch, image_index)
        evaluated = _evaluate_batch(model, image_batch, amp=amp, device_type=device_type)
        target_count = int(image_batch["cls"].numel())
        results.append(
            {
                "image_index": image_index,
                "image_file": str(image_file),
                "target_count": target_count,
                "empty_label": target_count == 0,
                **evaluated,
            }
        )
    return results


def _checkpoint_state_summary(model: object) -> dict[str, object]:
    nonfinite: list[str] = []
    tensors = 0
    values = 0
    for name, value in model.state_dict().items():
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            tensors += 1
            values += value.numel()
            if not value.isfinite().all().item():
                nonfinite.append(name)
    return {
        "floating_tensors": tensors,
        "floating_values": values,
        "all_finite": not nonfinite,
        "nonfinite_tensors": nonfinite,
    }


# Replays the trainer's rectangular validation loader and its loss calculation without fitting.
def run_batch_loss_diagnostic(
    project_root: Path,
    checkpoint: Path,
    *,
    label: str,
    device: str,
    amp: bool,
    max_batches: int | None = None,
) -> dict[str, object]:
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils.torch_utils import init_seeds

    project_root = project_root.resolve()
    checkpoint = checkpoint.resolve()
    # Reuse the archived failed run so later successful smoke runs cannot change the diagnostic inputs.
    failed_run = project_root / "outputs/training/smoke/_failed_attempts/A_seed42_amp_validation_overflow"
    data_yaml = failed_run / "dataset.yaml"
    trainer_args_path = failed_run / "args.yaml"
    if not checkpoint.is_file() or not data_yaml.is_file() or not trainer_args_path.is_file():
        raise ValidationDiagnosticError(f"Missing checkpoint or runtime dataset YAML for {label}.")

    init_seeds(42, deterministic=True)
    torch_device = torch.device(device)
    model = YOLO(str(checkpoint)).model.to(torch_device).float().eval()
    trainer_args = yaml.safe_load(trainer_args_path.read_text(encoding="utf-8"))
    model.args = get_cfg(overrides=trainer_args)
    validator = DetectionValidator(
        args={
            "model": str(checkpoint),
            "data": str(data_yaml),
            "imgsz": 640,
            "batch": 8,
            "device": device,
            "workers": 0,
            "plots": False,
            "split": "val",
            "rect": True,
            "quantize": 16 if amp else None,
            "verbose": False,
        }
    )
    validator.training = True
    validator.device = torch_device
    validator.data = check_det_dataset(str(data_yaml), split="val")
    validator.stride = max(int(model.stride.max()), 32)
    validator.names = model.names
    dataloader = validator.get_dataloader(validator.data["val"], 8)

    accumulator = {name: 0.0 for name in LOSS_NAMES}
    batches: list[dict[str, object]] = []
    first_nonfinite_batch: int | None = None
    all_empty_batches: list[int] = []
    for batch_index, raw_batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        raw_image_summary = summarize_tensor(raw_batch["img"])
        batch = validator.preprocess(raw_batch)
        targets = summarize_batch_targets(batch)
        evaluated = _evaluate_batch(model, batch, amp=amp, device_type=torch_device.type)
        for name, value in evaluated["losses"].items():
            accumulator[name] += value
        accumulator_finite = all(_finite_number(value) for value in accumulator.values())
        affected = not evaluated["predictions"]["all_finite"] or not evaluated["losses_finite"]
        first_affected = affected and first_nonfinite_batch is None
        if first_affected:
            first_nonfinite_batch = batch_index
        if targets["target_count"] == 0:
            all_empty_batches.append(batch_index)
        row: dict[str, object] = {
            "batch_index": batch_index,
            "raw_images": raw_image_summary,
            "normalized_images": summarize_tensor(batch["img"]),
            "targets": targets,
            **evaluated,
            "accumulated_losses": dict(accumulator),
            "accumulator_finite": accumulator_finite,
        }
        if first_affected:
            row["individual_images"] = _diagnose_individual_images(
                model, batch, amp=amp, device_type=torch_device.type
            )
            row["empty_target_replay"] = _evaluate_batch(
                model, _without_targets(batch), amp=amp, device_type=torch_device.type
            )
            row["detection_head_trace"] = _trace_detection_head(
                model, batch, amp=amp, device_type=torch_device.type
            )
        batches.append(row)

    completed_batches = len(batches)
    normalized = {
        name: value / completed_batches if completed_batches else math.nan for name, value in accumulator.items()
    }
    result = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "label": label,
        "checkpoint": checkpoint.relative_to(project_root).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "device": str(torch_device),
        "amp": amp,
        "input_precision": "float16" if amp else "float32",
        "seed": 42,
        "imgsz": 640,
        "batch_size": 8,
        "rectangular_validation": True,
        "workers": 0,
        "split": "val",
        "test_set_used": False,
        "model_state": _checkpoint_state_summary(model),
        "dataset_images": len(dataloader.dataset),
        "dataloader_batches": len(dataloader),
        "batches_completed": completed_batches,
        "first_nonfinite_batch": first_nonfinite_batch,
        "all_empty_batch_indices": all_empty_batches,
        "accumulated_losses": accumulator,
        "normalized_losses": normalized,
        "normalized_losses_finite": all(_finite_number(value) for value in normalized.values()),
        "batches": batches,
    }
    return result


def installed_source_evidence() -> dict[str, object]:
    """Hash the exact installed modules that define validation, loss, metrics, and CSV flow."""
    from ultralytics.engine import trainer, validator
    from ultralytics.models.yolo.detect import val
    from ultralytics.utils import loss, metrics

    modules = (validator, trainer, val, loss, metrics)
    return {
        module.__name__: {
            "path": str(Path(inspect.getfile(module)).resolve()),
            "sha256": _sha256(Path(inspect.getfile(module)).resolve()),
        }
        for module in modules
    }


def run_standalone_validation_comparison(project_root: Path) -> dict[str, object]:
    """Run the public validation API on the original and failed-run checkpoints."""
    project_root = project_root.resolve()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root / "outputs/phase2/phase2f/ultralytics_config"))
    from ultralytics import YOLO

    archived_run = project_root / "outputs/training/smoke/_failed_attempts/A_seed42_amp_validation_overflow"
    data_yaml = archived_run / "dataset.yaml"
    checkpoints = {
        "original_yolov8s": project_root / "yolov8s.pt",
        "failed_A_last": archived_run / "weights/last.pt",
        "failed_A_best": archived_run / "weights/best.pt",
    }
    missing = [str(path) for path in (data_yaml, *checkpoints.values()) if not path.is_file()]
    if missing:
        raise ValidationDiagnosticError(f"Missing standalone validation inputs: {missing}")

    results: dict[str, object] = {}
    # Standalone validation intentionally checks the public metric path separately from trainer losses.
    for label, checkpoint in checkpoints.items():
        metrics = YOLO(str(checkpoint)).val(
            data=str(data_yaml),
            split="val",
            imgsz=640,
            batch=8,
            device="0",
            workers=0,
            plots=False,
            verbose=False,
            project=str(project_root / DIAGNOSTIC_DIRECTORY / "standalone"),
            name=label,
            exist_ok=True,
        )
        values = {key: float(value) for key, value in metrics.results_dict.items()}
        results[label] = {
            "checkpoint": checkpoint.relative_to(project_root).as_posix(),
            "checkpoint_sha256": _sha256(checkpoint),
            "metrics": values,
            "all_metrics_finite": all(_finite_number(value) for value in values.values()),
        }
    report = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "split": "val",
        "test_set_used": False,
        "validation_loss_computed": False,
        "note": (
            "Ultralytics standalone model.val reports detection metrics but does not execute "
            "the trainer-integrated validation-loss branch."
        ),
        "runs": results,
    }
    _write_json(project_root / DIAGNOSTIC_DIRECTORY / "standalone_validation.json", report)
    return report


def summarize_existing_diagnostics(project_root: Path) -> dict[str, object]:
    """Condense the ignored per-batch traces into reviewable canonical evidence."""
    project_root = project_root.resolve()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root / "outputs/phase2/phase2f/ultralytics_config"))
    diagnostic_directory = project_root / DIAGNOSTIC_DIRECTORY
    report_names = (
        "original_cuda0_amp.json",
        "last_cuda0_amp.json",
        "best_cuda0_amp.json",
        "best_cuda0_fp32.json",
        "best_cpu_fp32.json",
    )
    reports: dict[str, dict[str, object]] = {}
    # These five runs isolate checkpoint state, AMP, and device while holding validation data fixed.
    for name in report_names:
        path = diagnostic_directory / name
        if not path.is_file():
            raise ValidationDiagnosticError(f"Missing diagnostic report: {path}")
        reports[name] = json.loads(path.read_text(encoding="utf-8"))

    trace_path = diagnostic_directory / "best_layer_trace_cuda0_amp.json"
    standalone_path = diagnostic_directory / "standalone_validation.json"
    if not trace_path.is_file() or not standalone_path.is_file():
        raise ValidationDiagnosticError("Layer trace and standalone validation evidence are required.")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    standalone = json.loads(standalone_path.read_text(encoding="utf-8"))
    first_batch = trace["batches"][0]
    first_module = first_batch["detection_head_trace"]["first_nonfinite_module"]
    first_event = next(
        event
        for event in first_batch["detection_head_trace"]["events"]
        if event["module"] == first_module
    )
    first_output = next(iter(first_event["outputs"].values()))

    comparisons: list[dict[str, object]] = []
    for name, report in reports.items():
        nonfinite_batches = sum(not batch["losses_finite"] for batch in report["batches"])
        prediction_nonfinite_batches = sum(
            not batch["predictions"]["all_finite"] for batch in report["batches"]
        )
        comparisons.append(
            {
                "report": name,
                "checkpoint": report["checkpoint"],
                "device": report["device"],
                "amp": report["amp"],
                "batches_completed": report["batches_completed"],
                "first_nonfinite_batch": report["first_nonfinite_batch"],
                "prediction_nonfinite_batches": prediction_nonfinite_batches,
                "loss_nonfinite_batches": nonfinite_batches,
                "normalized_losses": report["normalized_losses"],
                "normalized_losses_finite": report["normalized_losses_finite"],
            }
        )

    compact = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "conclusion": (
            "The failed A checkpoint overflows in FP16 validation inference; disabling AMP keeps "
            "the identical validation path finite on CUDA and CPU."
        ),
        "first_nonfinite_stage": {
            "batch_index": 0,
            "module": first_module,
            "module_output": first_output,
            "prediction_nonfinite_before_loss": True,
            "propagation": [
                "model prediction",
                "per-batch validation loss",
                "loss accumulator",
                "normalized validator result",
                "trainer metrics",
                "results.csv",
            ],
        },
        "first_batch_inputs": {
            "raw_images": first_batch["raw_images"],
            "normalized_images": first_batch["normalized_images"],
            "targets": first_batch["targets"],
        },
        "empty_label_edge_case": {
            "natural_empty_images_in_first_batch": first_batch["targets"]["empty_image_count"],
            "all_empty_natural_batches": trace["all_empty_batch_indices"],
            "targetless_replay_predictions_finite": first_batch["empty_target_replay"]["predictions"][
                "all_finite"
            ],
            "targetless_replay_losses": first_batch["empty_target_replay"]["losses"],
            "conclusion": (
                "Removing every target did not remove the prediction overflow, so empty labels "
                "are not the trigger."
            ),
        },
        "precision_comparisons": comparisons,
        "standalone_validation": standalone,
        "checkpoint_parameters_finite": reports["best_cuda0_amp.json"]["model_state"],
        "applied_fix": {
            "setting": "training.amp",
            "old_value": True,
            "new_value": False,
            "scope": "shared baseline for A/B/C/D",
            "site_packages_modified": False,
        },
        "failed_run_archive": "outputs/training/smoke/_failed_attempts/A_seed42_amp_validation_overflow",
        "test_set_used": False,
    }
    evidence_directory = project_root / CANONICAL_EVIDENCE_DIRECTORY
    # Canonical evidence stays compact and strict-JSON; full tensors and hooks remain under outputs/.
    compact = _json_safe(compact)
    _write_json(evidence_directory / "validation_nan_diagnosis.json", compact)
    _write_json(
        evidence_directory / "installed_validation_source_hashes.json",
        {
            "schema_version": 1,
            "created_at_utc": _utc_now(),
            "ultralytics_sources": installed_source_evidence(),
            "site_packages_modified": False,
        },
    )
    write_batch_csv(evidence_directory / "validation_batch_summary.csv", list(reports.values()))
    return compact


def write_batch_csv(path: Path, reports: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "label",
        "device",
        "amp",
        "batch_index",
        "image_count",
        "target_count",
        "empty_image_count",
        "class_ids",
        "prediction_finite",
        "box_loss",
        "cls_loss",
        "dfl_loss",
        "losses_finite",
        "accumulator_finite",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for batch in report["batches"]:
                targets = batch["targets"]
                losses = batch["losses"]
                writer.writerow(
                    {
                        "label": report["label"],
                        "device": report["device"],
                        "amp": report["amp"],
                        "batch_index": batch["batch_index"],
                        "image_count": targets["image_count"],
                        "target_count": targets["target_count"],
                        "empty_image_count": targets["empty_image_count"],
                        "class_ids": " ".join(str(value) for value in targets["class_ids"]),
                        "prediction_finite": batch["predictions"]["all_finite"],
                        "box_loss": losses["box_loss"],
                        "cls_loss": losses["cls_loss"],
                        "dfl_loss": losses["dfl_loss"],
                        "losses_finite": batch["losses_finite"],
                        "accumulator_finite": batch["accumulator_finite"],
                    }
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Phase 2F validation losses without training.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-batches", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = arguments.project_root.resolve()
    diagnostic_directory = project_root / DIAGNOSTIC_DIRECTORY
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root / "outputs/phase2/phase2f/ultralytics_config"))
    report = run_batch_loss_diagnostic(
        project_root,
        arguments.checkpoint,
        label=arguments.label,
        device=arguments.device,
        amp=arguments.amp,
        max_batches=arguments.max_batches,
    )
    suffix = "amp" if arguments.amp else "fp32"
    output_path = diagnostic_directory / f"{arguments.label}_{arguments.device.replace(':', '')}_{suffix}.json"
    _write_json(output_path, report)
    print(
        f"Validation diagnostic {arguments.label}: first_nonfinite_batch={report['first_nonfinite_batch']}, "
        f"normalized_losses={report['normalized_losses']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
