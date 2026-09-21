from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from bone_fracture_audit.audit import _read_dataset_config, sha256_file
from bone_fracture_audit.yolo import parse_yolo_label
from bone_fracture_pipeline.augmentation_policy import (
    AUGMENTATION_SETTINGS,
    compare_dataset_identities,
    load_augmentation_config,
    validate_augmentation_config,
    verify_framework_semantics,
    verify_paired_reproducibility,
)
from bone_fracture_pipeline.prepare_dataset import (
    CANONICAL_EXPECTATIONS,
    CLASS_NAMES,
    SPLITS,
    fingerprint_dataset,
)
from bone_fracture_pipeline.training_protocol import (
    AUGMENTATION_OFF_SETTINGS,
    ProtocolValidationError,
    collect_environment_metadata,
    load_protocol_config,
    protocol_digest,
    validate_prepared_dataset,
    validate_protocol_config,
    verify_ultralytics_semantics,
)


TOOL_VERSION = "1.0.0"
EXPERIMENTS = ("A", "B", "C", "D")
EXPECTED_WEIGHTS_SHA256 = "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"
EXPERIMENT_CONFIG_DIRECTORY = Path("configs/experiments")
BASELINE_CONFIG_PATH = Path("configs/training/baseline.yaml")
AUGMENTATION_CONFIG_PATH = Path("configs/training/augmentation_conservative.yaml")
SMOKE_SAMPLE_CONFIG_PATH = Path("configs/training/validation_smoke_samples.yaml")
PHASE2C_ENVIRONMENT_PATH = Path("docs/evidence/phase2c/training_environment.json")
CANONICAL_EVIDENCE_DIRECTORY = Path("docs/evidence/phase2f")


class ExperimentValidationError(RuntimeError):
    """Raised when a frozen experiment cannot be run without protocol drift."""


def load_yaml_mapping(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise ExperimentValidationError(f"{label} does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ExperimentValidationError(f"{label} must contain a YAML mapping.")
    return loaded


def config_digest(config: Mapping[str, object]) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# Resolves every condition from the same baseline so unrelated settings cannot diverge.
def resolve_experiment_config(project_root: Path, experiment: str) -> dict[str, object]:
    experiment = experiment.upper()
    if experiment not in EXPERIMENTS:
        raise ExperimentValidationError(f"Unknown experiment: {experiment}")

    experiment_path = project_root / EXPERIMENT_CONFIG_DIRECTORY / f"{experiment}.yaml"
    definition = load_yaml_mapping(experiment_path, f"Experiment {experiment} configuration")
    _require_exact_keys(
        definition,
        {"experiment_version", "experiment", "inherits", "image_condition", "augmentation_condition"},
        f"Experiment {experiment}",
    )
    if definition.get("experiment_version") != 1 or definition.get("experiment") != experiment:
        raise ExperimentValidationError(f"Experiment {experiment} identity or version is invalid.")
    if definition.get("inherits") != BASELINE_CONFIG_PATH.as_posix():
        raise ExperimentValidationError(f"Experiment {experiment} must inherit the frozen baseline.")

    baseline_path = project_root / BASELINE_CONFIG_PATH
    baseline = load_protocol_config(baseline_path)
    validate_protocol_config(baseline)
    resolved = json.loads(json.dumps(baseline))

    image_condition = _mapping(definition, "image_condition")
    _require_exact_keys(
        image_condition,
        {"name", "root", "data_yaml", "expected_fingerprint"},
        f"Experiment {experiment} image_condition",
    )
    expected_image_name = "png_control" if experiment in {"A", "C"} else "clahe"
    if image_condition.get("name") != expected_image_name:
        raise ExperimentValidationError(f"Experiment {experiment} has the wrong image condition.")
    dataset = _mapping(resolved, "dataset")
    for key in ("root", "data_yaml", "expected_fingerprint"):
        value = image_condition.get(key)
        if not isinstance(value, str) or not value:
            raise ExperimentValidationError(f"Experiment {experiment} image_condition.{key} is invalid.")
        dataset[key] = value

    augmentation_condition = _mapping(definition, "augmentation_condition")
    _require_exact_keys(
        augmentation_condition,
        {"name", "policy"},
        f"Experiment {experiment} augmentation_condition",
    )
    augmentation_on = experiment in {"C", "D"}
    expected_name = "conservative" if augmentation_on else "off"
    if augmentation_condition.get("name") != expected_name:
        raise ExperimentValidationError(f"Experiment {experiment} has the wrong augmentation condition.")
    policy = augmentation_condition.get("policy")
    if augmentation_on:
        if policy != AUGMENTATION_CONFIG_PATH.as_posix():
            raise ExperimentValidationError(f"Experiment {experiment} must use the conservative policy.")
        augmentation_config = load_augmentation_config(project_root / AUGMENTATION_CONFIG_PATH)
        validate_augmentation_config(augmentation_config)
        resolved["augmentation"] = dict(_mapping(augmentation_config, "augmentation"))
    elif policy is not None:
        raise ExperimentValidationError(f"Experiment {experiment} must not load an augmentation policy.")

    return {
        "experiment": experiment,
        "definition_path": experiment_path.relative_to(project_root).as_posix(),
        "definition": definition,
        "definition_digest": config_digest(definition),
        "baseline_path": BASELINE_CONFIG_PATH.as_posix(),
        "baseline_digest": protocol_digest(baseline),
        "augmentation_policy_path": AUGMENTATION_CONFIG_PATH.as_posix() if augmentation_on else None,
        "augmentation_policy_digest": (
            config_digest(load_augmentation_config(project_root / AUGMENTATION_CONFIG_PATH))
            if augmentation_on
            else None
        ),
        "image_condition": expected_image_name,
        "augmentation_condition": expected_name,
        "resolved": resolved,
        "resolved_digest": config_digest(resolved),
    }


def _scientific_config(resolution: Mapping[str, object]) -> dict[str, object]:
    resolved = _mapping(resolution, "resolved")
    dataset = dict(_mapping(resolved, "dataset"))
    return {
        "dataset": dataset,
        "model": dict(_mapping(resolved, "model")),
        "training": dict(_mapping(resolved, "training")),
        "augmentation": dict(_mapping(resolved, "augmentation")),
        "evaluation": dict(_mapping(resolved, "evaluation")),
        "seed_policy": dict(_mapping(resolved, "seed_policy")),
        "framework": dict(_mapping(resolved, "framework")),
    }


def _different_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[str] = set()
        for key in set(left) | set(right):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(_different_paths(left[key], right[key], child))
        return paths
    return set() if left == right and type(left) is type(right) else {prefix}


# Proves the 2x2 matrix changes only the planned preprocessing and augmentation factors.
def validate_experiment_matrix(project_root: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    resolutions = {name: resolve_experiment_config(project_root, name) for name in EXPERIMENTS}
    scientific = {name: _scientific_config(value) for name, value in resolutions.items()}
    dataset_differences = {
        "dataset.root",
        "dataset.data_yaml",
        "dataset.expected_fingerprint",
    }
    augmentation_differences = {
        "augmentation.degrees",
        "augmentation.hsv_v",
        "augmentation.scale",
        "augmentation.translate",
    }
    expected_pairs = {
        "A_vs_B": dataset_differences,
        "A_vs_C": augmentation_differences,
        "B_vs_D": augmentation_differences,
        "C_vs_D": dataset_differences,
    }
    comparisons: dict[str, object] = {}
    for pair, expected in expected_pairs.items():
        left_name, right_name = pair.split("_vs_")
        actual = _different_paths(scientific[left_name], scientific[right_name])
        if actual != expected:
            raise ExperimentValidationError(
                f"Unexpected matrix differences for {pair}: {sorted(actual)}; expected {sorted(expected)}"
            )
        comparisons[pair] = {"valid": True, "different_paths": sorted(actual)}

    if _different_paths(scientific["A"], scientific["D"]) != dataset_differences | augmentation_differences:
        raise ExperimentValidationError("Experiment A and D differ outside the two planned factors.")

    result = {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "validated_at_utc": _utc_now(),
        "success": True,
        "experiments": {
            name: {
                "definition_path": resolution["definition_path"],
                "definition_digest": resolution["definition_digest"],
                "baseline_digest": resolution["baseline_digest"],
                "augmentation_policy_digest": resolution["augmentation_policy_digest"],
                "resolved_digest": resolution["resolved_digest"],
                "image_condition": resolution["image_condition"],
                "augmentation_condition": resolution["augmentation_condition"],
            }
            for name, resolution in resolutions.items()
        },
        "comparisons": comparisons,
        "only_planned_factors_differ": True,
    }
    return resolutions, result


def inspect_detection_dataset(project_root: Path, config: Mapping[str, object]) -> dict[str, object]:
    dataset = _mapping(config, "dataset")
    root = _resolve(project_root, dataset.get("root"), "dataset.root")
    if _read_dataset_config(root / "data.yaml") != CLASS_NAMES:
        raise ExperimentValidationError("Dataset class mapping differs from the approved six classes.")

    split_results: dict[str, object] = {}
    total_annotations = 0
    total_empty = 0
    total_class_counts: Counter[int] = Counter()
    valid_ids = set(range(len(CLASS_NAMES)))
    for split in SPLITS:
        image_paths = sorted(path for path in (root / split / "images").iterdir() if path.is_file())
        label_paths = sorted((root / split / "labels").glob("*.txt"))
        image_stems = {path.stem.casefold() for path in image_paths}
        label_stems = {path.stem.casefold() for path in label_paths}
        if image_stems != label_stems:
            raise ExperimentValidationError(f"Image/label identities differ in {split}.")

        annotation_count = 0
        empty_count = 0
        class_counts: Counter[int] = Counter()
        for label_path in label_paths:
            parsed = parse_yolo_label(label_path, valid_ids)
            if parsed.issues or any(item.annotation_type != "box" for item in parsed.annotations):
                raise ExperimentValidationError(f"Invalid detection label: {label_path}")
            annotation_count += len(parsed.annotations)
            empty_count += int(parsed.is_empty)
            class_counts.update(item.class_id for item in parsed.annotations)

        expected = CANONICAL_EXPECTATIONS.splits[split]
        actual_counts = tuple(class_counts[index] for index in range(len(CLASS_NAMES)))
        if (len(image_paths), annotation_count, empty_count, actual_counts) != (
            expected.images,
            expected.annotations,
            expected.empty_labels,
            expected.class_counts,
        ):
            raise ExperimentValidationError(f"Dataset counts differ from the approved values in {split}.")
        split_results[split] = {
            "images": len(image_paths),
            "labels": len(label_paths),
            "annotations": annotation_count,
            "empty_labels": empty_count,
            "class_counts": {CLASS_NAMES[index]: actual_counts[index] for index in range(len(CLASS_NAMES))},
        }
        total_annotations += annotation_count
        total_empty += empty_count
        total_class_counts.update(class_counts)

    fingerprint = fingerprint_dataset(root)
    if fingerprint.digest != dataset.get("expected_fingerprint"):
        raise ExperimentValidationError("Dataset fingerprint differs from the frozen experiment configuration.")
    return {
        "valid": True,
        "root": root.relative_to(project_root).as_posix(),
        "fingerprint": fingerprint.digest,
        "file_count": fingerprint.file_count,
        "images": CANONICAL_EXPECTATIONS.images,
        "labels": CANONICAL_EXPECTATIONS.images,
        "annotations": total_annotations,
        "empty_labels": total_empty,
        "class_names": list(CLASS_NAMES),
        "class_counts": {
            CLASS_NAMES[index]: total_class_counts[index] for index in range(len(CLASS_NAMES))
        },
        "splits": split_results,
    }


def collect_and_compare_environment(
    project_root: Path,
    config: Mapping[str, object],
    weights_path: Path,
) -> dict[str, object]:
    environment = collect_environment_metadata(project_root, config, weights_path)
    reference_path = project_root / PHASE2C_ENVIRONMENT_PATH
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    comparisons = {
        "python.version": (environment["python"]["version"], reference["python"]["version"]),
        "python.implementation": (
            environment["python"]["implementation"],
            reference["python"]["implementation"],
        ),
        "libraries": (environment["libraries"], reference["libraries"]),
        "cuda.available": (environment["cuda"]["available"], reference["cuda"]["available"]),
        "cuda.runtime_version": (
            environment["cuda"]["runtime_version"],
            reference["cuda"]["runtime_version"],
        ),
        "cuda.cudnn_version": (
            environment["cuda"]["cudnn_version"],
            reference["cuda"]["cudnn_version"],
        ),
        "gpu.name": (environment["gpu"]["name"], reference["gpu"]["name"]),
        "gpu.vram_bytes": (environment["gpu"]["vram_bytes"], reference["gpu"]["vram_bytes"]),
        "gpu.driver_version": (
            environment["gpu"]["driver_version"],
            reference["gpu"]["driver_version"],
        ),
        "weights.sha256": (environment["weights"]["sha256"], reference["weights"]["sha256"]),
    }
    differences = {
        name: {"current": current, "phase2c": previous}
        for name, (current, previous) in comparisons.items()
        if current != previous
    }
    if differences:
        raise ExperimentValidationError(f"Training environment drifted from Phase 2C: {differences}")
    if not environment["cuda"]["available"]:
        raise ExperimentValidationError("CUDA device 0 is not available.")
    if importlib.util.find_spec("albumentations") is not None:
        raise ExperimentValidationError("Albumentations is installed and could add unapproved transforms.")
    environment["comparison_to_phase2c"] = {
        "reference": PHASE2C_ENVIRONMENT_PATH.as_posix(),
        "matches": True,
        "differences": {},
        "fields_checked": sorted(comparisons),
    }
    environment["albumentations_installed"] = False
    return environment


def run_preflight(
    project_root: Path,
    resolution: Mapping[str, object],
    matrix_result: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    config = _mapping(resolution, "resolved")
    baseline = load_protocol_config(project_root / BASELINE_CONFIG_PATH)
    try:
        protocol_result = validate_protocol_config(baseline)
        dataset_contract = validate_prepared_dataset(project_root, config)
        framework_result = verify_ultralytics_semantics(baseline)
    except ProtocolValidationError as error:
        raise ExperimentValidationError(str(error)) from error

    weights_path = project_root / str(_mapping(config, "model")["weights"])
    if not weights_path.is_file() or sha256_file(weights_path) != EXPECTED_WEIGHTS_SHA256:
        raise ExperimentValidationError("Pretrained yolov8s.pt is missing or has the wrong SHA-256 hash.")
    dataset_detail = inspect_detection_dataset(project_root, config)
    environment = collect_and_compare_environment(project_root, config, weights_path)

    augmentation_on = resolution["augmentation_condition"] == "conservative"
    augmentation_semantics: dict[str, object]
    if augmentation_on:
        policy = load_augmentation_config(project_root / AUGMENTATION_CONFIG_PATH)
        augmentation_semantics = verify_framework_semantics(project_root, policy)
    else:
        if dict(_mapping(config, "augmentation")) != AUGMENTATION_OFF_SETTINGS:
            raise ExperimentValidationError("A no-augmentation condition contains active transform settings.")
        augmentation_semantics = {
            "valid": True,
            "controlled_augmentation": False,
            "all_frozen_transform_values_off": True,
            "albumentations_installed": False,
        }

    split_policy = _mapping(_mapping(config, "dataset"), "split_policy")
    if split_policy.get("training") != "train" or split_policy.get("validation") != "valid":
        raise ExperimentValidationError("Training or validation split routing changed.")
    if split_policy.get("test_usage") != "final_evaluation_only":
        raise ExperimentValidationError("Test-set isolation policy changed.")

    result = {
        "schema_version": 1,
        "checked_at_utc": _utc_now(),
        "success": True,
        "experiment": resolution["experiment"],
        "configuration": {
            "definition_digest": resolution["definition_digest"],
            "baseline_digest": resolution["baseline_digest"],
            "augmentation_policy_digest": resolution["augmentation_policy_digest"],
            "resolved_digest": resolution["resolved_digest"],
            "protocol": protocol_result,
            "matrix_valid": matrix_result["success"],
        },
        "dataset_contract": dataset_contract,
        "dataset_detail": dataset_detail,
        "weights": {
            "path": weights_path.relative_to(project_root).as_posix(),
            "size_bytes": weights_path.stat().st_size,
            "sha256": sha256_file(weights_path),
        },
        "framework": framework_result,
        "augmentation_semantics": augmentation_semantics,
        "test_set_isolated": True,
        "test_split_used": False,
    }
    return result, environment


def verify_cd_pairing(project_root: Path, resolutions: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    c_config = _mapping(resolutions["C"], "resolved")
    d_config = _mapping(resolutions["D"], "resolved")
    allowed = {"dataset.root", "dataset.data_yaml", "dataset.expected_fingerprint"}
    if _different_paths(_scientific_config(resolutions["C"]), _scientific_config(resolutions["D"])) != allowed:
        raise ExperimentValidationError("Final C and D configurations differ outside preprocessing.")
    for key in ("seed", "deterministic", "workers", "batch", "cache", "rect"):
        if _mapping(c_config, "training").get(key) != _mapping(d_config, "training").get(key):
            raise ExperimentValidationError(f"C/D loader setting differs: {key}")

    control_root = _resolve(project_root, _mapping(c_config, "dataset")["root"], "C dataset root")
    clahe_root = _resolve(project_root, _mapping(d_config, "dataset")["root"], "D dataset root")
    samples, identity = compare_dataset_identities(control_root, clahe_root)
    policy = load_augmentation_config(project_root / AUGMENTATION_CONFIG_PATH)
    pairing = verify_paired_reproducibility(samples, policy)
    return {
        "schema_version": 1,
        "validated_at_utc": _utc_now(),
        "success": True,
        "c_resolved_digest": resolutions["C"]["resolved_digest"],
        "d_resolved_digest": resolutions["D"]["resolved_digest"],
        "final_resolved_config_differences": sorted(allowed),
        "identity": identity,
        "pairing": pairing,
    }


def build_training_arguments(
    project_root: Path,
    resolution: Mapping[str, object],
    *,
    smoke: bool,
    require_new_output: bool = True,
) -> tuple[dict[str, object], Path, str]:
    config = _mapping(resolution, "resolved")
    training = dict(_mapping(config, "training"))
    experiment = str(resolution["experiment"])
    run_kind = "smoke" if smoke else "official"
    run_name = f"{experiment}_seed{training['seed']}"
    output_root = project_root / str(training.pop("project")) / run_kind
    run_directory = output_root / run_name
    if require_new_output and run_directory.exists():
        raise ExperimentValidationError(f"Run output already exists and will not be overwritten: {run_directory}")

    training["epochs"] = 1 if smoke else training["epochs"]
    arguments = {
        "model": str(project_root / str(_mapping(config, "model")["weights"])),
        "data": str(run_directory / "dataset.yaml"),
        **training,
        **dict(_mapping(config, "augmentation")),
        "project": str(output_root),
        "name": run_name,
        # The directory is pre-created only to persist a running/failed manifest; the guard above prevents reuse.
        "exist_ok": True,
        "resume": False,
        "task": "detect",
        "mode": "train",
    }
    return arguments, run_directory, run_kind


# Ultralytics resolves `path: .` against the process directory, so each run gets an absolute-root copy.
def write_runtime_dataset_yaml(
    project_root: Path,
    resolution: Mapping[str, object],
    run_directory: Path,
) -> dict[str, object]:
    config = _mapping(resolution, "resolved")
    dataset = _mapping(config, "dataset")
    source_path = _resolve(project_root, dataset["data_yaml"], "dataset.data_yaml")
    runtime = load_yaml_mapping(source_path, "Prepared dataset YAML")
    runtime["path"] = str(_resolve(project_root, dataset["root"], "dataset.root"))
    output_path = run_directory / "dataset.yaml"
    output_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
    return {
        "source": source_path.relative_to(project_root).as_posix(),
        "runtime": _display_path(project_root, output_path),
        "runtime_sha256": sha256_file(output_path),
        "absolute_dataset_root": runtime["path"],
    }


def _dataset_cache_paths(project_root: Path, config: Mapping[str, object]) -> list[Path]:
    dataset_root = _resolve(project_root, _mapping(config, "dataset")["root"], "dataset.root")
    return [dataset_root / split / "labels.cache" for split in ("train", "valid")]


def _remove_generated_dataset_caches(paths: Sequence[Path]) -> list[str]:
    removed: list[str] = []
    for path in paths:
        if path.is_file():
            path.unlink()
            removed.append(path.as_posix())
    return removed


def verify_applied_trainer_config(
    run_directory: Path,
    expected: Mapping[str, object],
) -> dict[str, object]:
    args_path = run_directory / "args.yaml"
    actual = load_yaml_mapping(args_path, "Trainer args.yaml")
    checked_keys = [
        "model",
        "data",
        "epochs",
        "batch",
        "imgsz",
        "optimizer",
        "lr0",
        "weight_decay",
        "seed",
        "deterministic",
        "amp",
        "patience",
        "val",
        "pretrained",
        "device",
        "workers",
        "cache",
        "rect",
        "multi_scale",
        "cos_lr",
        "save",
        "save_period",
        "plots",
        *AUGMENTATION_OFF_SETTINGS,
    ]
    differences: dict[str, object] = {}
    for key in checked_keys:
        expected_value = expected.get(key)
        actual_value = actual.get(key)
        if key in {"data", "project"}:
            equal = Path(str(actual_value)).resolve() == Path(str(expected_value)).resolve()
        elif key == "device":
            # Ultralytics serializes CUDA device 0 as "0" even when the launcher passes integer 0.
            equal = str(actual_value) == str(expected_value)
        else:
            equal = actual_value == expected_value and type(actual_value) is type(expected_value)
        if not equal:
            differences[key] = {"expected": expected_value, "actual": actual_value}
    if differences:
        raise ExperimentValidationError(f"Trainer did not apply the resolved configuration: {differences}")
    return {
        "valid": True,
        "args_path": args_path.as_posix(),
        "checked_keys": sorted(set(checked_keys)),
        "differences": {},
        "actual_args_digest": config_digest(actual),
        "actual_args": actual,
    }


def verify_training_outputs(run_directory: Path, expected_epochs: int) -> dict[str, object]:
    results_path = run_directory / "results.csv"
    if not results_path.is_file():
        raise ExperimentValidationError("Training did not produce results.csv.")
    with results_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_epochs:
        raise ExperimentValidationError(f"Expected {expected_epochs} results rows, found {len(rows)}.")

    required_columns = {
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    }
    normalized_rows = [{key.strip(): value for key, value in row.items()} for row in rows]
    missing = sorted(required_columns - set(normalized_rows[-1]))
    if missing:
        raise ExperimentValidationError(f"results.csv is missing required values: {missing}")
    checked_values: dict[str, float] = {}
    for column in sorted(required_columns):
        try:
            value = float(normalized_rows[-1][column])
        except (TypeError, ValueError) as error:
            raise ExperimentValidationError(f"Non-numeric training result in {column}.") from error
        if not math.isfinite(value):
            raise ExperimentValidationError(f"Non-finite training result in {column}.")
        checked_values[column] = value

    last_checkpoint = run_directory / "weights" / "last.pt"
    best_checkpoint = run_directory / "weights" / "best.pt"
    batch_images = sorted(run_directory.glob("train_batch*.jpg"))
    required_files = [run_directory / "results.png", last_checkpoint, best_checkpoint]
    missing_files = [path.as_posix() for path in required_files if not path.is_file()]
    if missing_files or not batch_images:
        raise ExperimentValidationError(
            f"Training output package is incomplete: missing={missing_files}, batch_images={len(batch_images)}"
        )
    return {
        "valid": True,
        "epochs_completed": len(rows),
        "normal_completion": True,
        "finite_metrics_and_losses": True,
        "final_values": checked_values,
        "results_csv": results_path.as_posix(),
        "last_checkpoint": last_checkpoint.as_posix(),
        "best_checkpoint": best_checkpoint.as_posix(),
        "trainer_batch_images": [path.as_posix() for path in batch_images],
        "plots_present": sorted(path.name for path in run_directory.glob("*.png")),
    }


# Confirms the trainer actually loaded only the frozen training and validation split paths.
def verify_trainer_data_routing(
    trainer: object,
    project_root: Path,
    config: Mapping[str, object],
) -> dict[str, object]:
    dataset_root = _resolve(project_root, _mapping(config, "dataset")["root"], "dataset.root")
    train_root = (dataset_root / "train" / "images").resolve()
    valid_root = (dataset_root / "valid" / "images").resolve()
    test_root = (dataset_root / "test" / "images").resolve()

    train_files = [Path(path).resolve() for path in trainer.train_loader.dataset.im_files]
    validation_files = [Path(path).resolve() for path in trainer.validator.dataloader.dataset.im_files]
    if len(train_files) != CANONICAL_EXPECTATIONS.splits["train"].images:
        raise ExperimentValidationError("Trainer loaded the wrong number of training images.")
    if len(validation_files) != CANONICAL_EXPECTATIONS.splits["valid"].images:
        raise ExperimentValidationError("Trainer loaded the wrong number of validation images.")
    if any(not path.is_relative_to(train_root) for path in train_files):
        raise ExperimentValidationError("Trainer loaded a non-training image into the training loader.")
    if any(not path.is_relative_to(valid_root) for path in validation_files):
        raise ExperimentValidationError("Trainer loaded a non-validation image into the validation loader.")
    if any(path.is_relative_to(test_root) for path in train_files + validation_files):
        raise ExperimentValidationError("Trainer accessed a test image during fitting or validation.")
    return {
        "valid": True,
        "training_images_loaded": len(train_files),
        "validation_images_loaded": len(validation_files),
        "training_root": train_root.relative_to(project_root).as_posix(),
        "validation_root": valid_root.relative_to(project_root).as_posix(),
        "test_images_loaded": 0,
    }


# Records the concrete transform objects built by Ultralytics for technical augmentation QA.
def inspect_trainer_transform_chain(trainer: object, run_directory: Path) -> dict[str, object]:
    transforms = list(getattr(trainer.train_loader.dataset.transforms, "transforms", []))
    attribute_names = (
        "p",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "hgain",
        "sgain",
        "vgain",
        "direction",
    )
    entries: list[dict[str, object]] = []
    for transform in transforms:
        values: dict[str, object] = {}
        for name in attribute_names:
            value = getattr(transform, name, None)
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[name] = value
            elif isinstance(value, (tuple, list)):
                values[name] = list(value)
        if type(transform).__name__ == "Albumentations":
            values["active"] = getattr(transform, "transform", None) is not None
        entries.append({"class": type(transform).__name__, "settings": values})
    result = {
        "valid": True,
        "source": "trainer.train_loader.dataset.transforms",
        "transform_count": len(entries),
        "transforms": entries,
    }
    _write_json(run_directory / "trainer_transform_chain.json", result)
    return result


def run_validation_inference(
    project_root: Path,
    resolution: Mapping[str, object],
    run_directory: Path,
) -> dict[str, object]:
    from ultralytics import YOLO

    sample_config = load_yaml_mapping(project_root / SMOKE_SAMPLE_CONFIG_PATH, "Smoke sample configuration")
    if sample_config.get("sample_set_version") != 1 or sample_config.get("split") != "valid":
        raise ExperimentValidationError("Smoke inference sample set must be fixed to the validation split.")
    sample_ids = sample_config.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids or not all(isinstance(item, str) for item in sample_ids):
        raise ExperimentValidationError("Smoke inference sample IDs are invalid.")

    config = _mapping(resolution, "resolved")
    dataset_root = _resolve(project_root, _mapping(config, "dataset")["root"], "dataset.root")
    image_directory = dataset_root / "valid" / "images"
    image_by_stem = {path.stem: path for path in image_directory.iterdir() if path.is_file()}
    missing = [sample_id for sample_id in sample_ids if sample_id not in image_by_stem]
    if missing:
        raise ExperimentValidationError(f"Validation smoke samples are missing: {missing}")
    sources = [str(image_by_stem[sample_id]) for sample_id in sample_ids]

    checkpoint = run_directory / "weights" / "best.pt"
    model = YOLO(str(checkpoint))
    predictions = model.predict(
        source=sources,
        imgsz=int(_mapping(config, "training")["imgsz"]),
        device=int(_mapping(config, "training")["device"]),
        augment=False,
        save=True,
        project=str(run_directory / "inference"),
        name="validation_smoke",
        exist_ok=False,
        verbose=False,
    )
    if len(predictions) != len(sample_ids):
        raise ExperimentValidationError("Validation smoke inference returned the wrong number of results.")

    result_rows: list[dict[str, object]] = []
    for sample_id, prediction in zip(sample_ids, predictions):
        boxes = prediction.boxes
        tensors = [boxes.xyxy, boxes.conf, boxes.cls]
        if any(not tensor.isfinite().all().item() for tensor in tensors):
            raise ExperimentValidationError(f"Inference produced non-finite values for {sample_id}.")
        height, width = prediction.orig_shape
        if len(boxes):
            coordinates = boxes.xyxy
            if (
                (coordinates[:, 0] < 0).any()
                or (coordinates[:, 1] < 0).any()
                or (coordinates[:, 2] > width).any()
                or (coordinates[:, 3] > height).any()
            ):
                raise ExperimentValidationError(f"Inference produced an out-of-bounds box for {sample_id}.")
        result_rows.append(
            {
                "sample_id": sample_id,
                "split": "valid",
                "detections": len(boxes),
                "image_width": width,
                "image_height": height,
            }
        )
    output = {
        "valid": True,
        "checkpoint": checkpoint.as_posix(),
        "sample_config": SMOKE_SAMPLE_CONFIG_PATH.as_posix(),
        "split": "valid",
        "test_images": 0,
        "finite_predictions": True,
        "results": result_rows,
    }
    _write_json(run_directory / "inference_smoke.json", output)
    return output


def run_experiment(project_root: Path, experiment: str, *, smoke: bool) -> dict[str, object]:
    project_root = project_root.resolve()
    _configure_ultralytics(project_root)
    resolutions, matrix_result = validate_experiment_matrix(project_root)
    resolution = resolutions[experiment.upper()]
    arguments, run_directory, run_kind = build_training_arguments(project_root, resolution, smoke=smoke)
    preflight, environment = run_preflight(project_root, resolution, matrix_result)

    cache_paths = _dataset_cache_paths(project_root, _mapping(resolution, "resolved"))
    existing_caches = [path for path in cache_paths if path.exists()]
    if existing_caches:
        raise ExperimentValidationError(f"Unexpected pre-existing dataset cache files: {existing_caches}")

    from ultralytics import YOLO
    import torch

    run_directory.mkdir(parents=True)
    runtime_dataset_yaml = write_runtime_dataset_yaml(project_root, resolution, run_directory)
    manifest_path = run_directory / "run_manifest.json"
    started_at = _utc_now()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "experiment": experiment.upper(),
        "run_kind": run_kind,
        "status": "running",
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "official_thesis_result": False if smoke else True,
        "smoke_override": {"epochs": 1} if smoke else None,
        "configuration": {
            key: resolution[key]
            for key in (
                "definition_path",
                "definition_digest",
                "baseline_path",
                "baseline_digest",
                "augmentation_policy_path",
                "augmentation_policy_digest",
                "resolved_digest",
                "image_condition",
                "augmentation_condition",
            )
        },
        "resolved_config": resolution["resolved"],
        "runtime_dataset_yaml": runtime_dataset_yaml,
        "trainer_arguments": arguments,
        "preflight": preflight,
        "environment": environment,
        "repository_commit_sha": _repository_commit(project_root),
        "test_set_used": False,
        "output_directory": run_directory.relative_to(project_root).as_posix(),
    }
    _write_json(manifest_path, manifest)

    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    generated_caches: list[str] = []
    try:
        model = YOLO(str(project_root / str(_mapping(_mapping(resolution, "resolved"), "model")["weights"])))
        model.train(**arguments)
        elapsed = time.perf_counter() - started
        applied = verify_applied_trainer_config(run_directory, arguments)
        outputs = verify_training_outputs(run_directory, int(arguments["epochs"]))
        data_routing = verify_trainer_data_routing(
            model.trainer,
            project_root,
            _mapping(resolution, "resolved"),
        )
        transform_chain = inspect_trainer_transform_chain(model.trainer, run_directory)
        inference = run_validation_inference(project_root, resolution, run_directory)
        generated_caches = _remove_generated_dataset_caches(cache_paths)

        # The derived dataset must return to its approved fingerprint after framework cache cleanup.
        post_fingerprint = fingerprint_dataset(
            _resolve(project_root, _mapping(_mapping(resolution, "resolved"), "dataset")["root"], "dataset.root")
        ).digest
        expected_fingerprint = _mapping(_mapping(resolution, "resolved"), "dataset")["expected_fingerprint"]
        if post_fingerprint != expected_fingerprint:
            raise ExperimentValidationError("Dataset fingerprint changed during training.")

        train_images = CANONICAL_EXPECTATIONS.splits["train"].images
        batches = math.ceil(train_images / int(arguments["batch"]))
        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "runtime": {
                    "seconds": round(elapsed, 3),
                    "minutes": round(elapsed / 60, 3),
                    "training_images": train_images,
                    "training_batches": batches,
                    "images_per_second_end_to_end": round(train_images / elapsed, 3),
                    "batches_per_second_end_to_end": round(batches / elapsed, 3),
                },
                "gpu_memory": {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                    "peak_allocated_gib": round(torch.cuda.max_memory_allocated(0) / (1024**3), 3),
                    "peak_reserved_gib": round(torch.cuda.max_memory_reserved(0) / (1024**3), 3),
                    "batch_size_8_stable": True,
                    "oom": False,
                },
                "applied_trainer_config": applied,
                "output_verification": outputs,
                "trainer_data_routing": data_routing,
                "trainer_transform_chain": transform_chain,
                "validation_inference": inference,
                "dataset_cache_cleanup": {
                    "removed_generated_files": generated_caches,
                    "post_training_fingerprint": post_fingerprint,
                    "fingerprint_restored": True,
                },
            }
        )
    except Exception as error:
        generated_caches = _remove_generated_dataset_caches(cache_paths)
        oom = "out of memory" in str(error).lower()
        manifest.update(
            {
                "status": "failed",
                "completed_at_utc": _utc_now(),
                "failure": {"type": type(error).__name__, "message": str(error), "oom": oom},
                "dataset_cache_cleanup": {"removed_generated_files": generated_caches},
            }
        )
        _write_json(manifest_path, manifest)
        raise
    _write_json(manifest_path, manifest)
    return manifest


def write_freeze_evidence(project_root: Path) -> dict[str, object]:
    _configure_ultralytics(project_root)
    resolutions, matrix = validate_experiment_matrix(project_root)
    pairing = verify_cd_pairing(project_root, resolutions)
    reference_resolution = resolutions["A"]
    config = _mapping(reference_resolution, "resolved")
    weights_path = project_root / str(_mapping(config, "model")["weights"])
    environment = collect_and_compare_environment(project_root, config, weights_path)
    evidence_directory = project_root / CANONICAL_EVIDENCE_DIRECTORY
    evidence_directory.mkdir(parents=True, exist_ok=True)
    _write_json(evidence_directory / "experiment_matrix_validation.json", matrix)
    _write_json(evidence_directory / "cd_pairing_validation.json", pairing)
    _write_json(evidence_directory / "environment_freeze.json", environment)
    return {"matrix": matrix, "pairing": pairing, "environment": environment}


def summarize_smoke_runs(project_root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    manifests: dict[str, dict[str, object]] = {}
    for experiment in EXPERIMENTS:
        path = project_root / "outputs" / "training" / "smoke" / f"{experiment}_seed42" / "run_manifest.json"
        if not path.is_file():
            raise ExperimentValidationError(f"Smoke manifest is missing for Experiment {experiment}.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("run_kind") != "smoke":
            raise ExperimentValidationError(f"Experiment {experiment} smoke run did not complete successfully.")
        manifests[experiment] = manifest
        final_values = manifest["output_verification"]["final_values"]
        rows.append(
            {
                "experiment": experiment,
                "image_condition": manifest["configuration"]["image_condition"],
                "augmentation": manifest["configuration"]["augmentation_condition"],
                "epochs": manifest["output_verification"]["epochs_completed"],
                "batch": manifest["trainer_arguments"]["batch"],
                "runtime_seconds": manifest["runtime"]["seconds"],
                "peak_allocated_gib": manifest["gpu_memory"]["peak_allocated_gib"],
                "peak_reserved_gib": manifest["gpu_memory"]["peak_reserved_gib"],
                "box_loss": final_values["train/box_loss"],
                "cls_loss": final_values["train/cls_loss"],
                "dfl_loss": final_values["train/dfl_loss"],
                "map50": final_values["metrics/mAP50(B)"],
                "map50_95": final_values["metrics/mAP50-95(B)"],
                "checkpoint_last": True,
                "checkpoint_best": True,
                "validation_inference": True,
                "test_used": False,
                "status": "passed",
            }
        )

    evidence_directory = project_root / CANONICAL_EVIDENCE_DIRECTORY
    evidence_directory.mkdir(parents=True, exist_ok=True)
    csv_path = evidence_directory / "smoke_run_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "success": True,
        "all_four_smoke_runs_passed": True,
        "official_training_performed": False,
        "test_set_used": False,
        "batch_size_8_stable_all_runs": all(item["gpu_memory"]["batch_size_8_stable"] for item in manifests.values()),
        "runs": {
            experiment: {
                "manifest": f"outputs/training/smoke/{experiment}_seed42/run_manifest.json",
                "status": manifest["status"],
                "runtime": manifest["runtime"],
                "gpu_memory": manifest["gpu_memory"],
                "output_verification": {
                    key: manifest["output_verification"][key]
                    for key in ("valid", "epochs_completed", "normal_completion", "finite_metrics_and_losses")
                },
                "validation_inference": {
                    "valid": manifest["validation_inference"]["valid"],
                    "split": manifest["validation_inference"]["split"],
                    "test_images": manifest["validation_inference"]["test_images"],
                },
            }
            for experiment, manifest in manifests.items()
        },
    }
    _write_json(evidence_directory / "smoke_test_summary.json", summary)
    return summary


def _repository_commit(project_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _configure_ultralytics(project_root: Path) -> None:
    config_directory = project_root / "outputs" / "phase2" / "phase2f" / "ultralytics_config"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_directory)


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ExperimentValidationError(f"Configuration section {key!r} must be a mapping.")
    return value


def _require_exact_keys(actual: Mapping[str, object], expected: set[str], label: str) -> None:
    actual_keys = set(actual)
    if actual_keys != expected:
        raise ExperimentValidationError(
            f"{label} has unexpected or missing keys: actual={sorted(actual_keys)}, expected={sorted(expected)}"
        )


def _resolve(project_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExperimentValidationError(f"{label} must be a non-empty path string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze, validate, and launch Phase 2F experiments.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--smoke", action="store_true", help="Override only the epoch count to one.")
    parser.add_argument("--validate-freeze", action="store_true", help="Write matrix, environment, and C/D evidence.")
    parser.add_argument("--summarize-smoke", action="store_true", help="Summarize four completed smoke runs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = arguments.project_root.resolve()
    selected_actions = sum(
        bool(value)
        for value in (arguments.experiment, arguments.validate_freeze, arguments.summarize_smoke)
    )
    if selected_actions != 1:
        print("Choose exactly one action: --experiment, --validate-freeze, or --summarize-smoke.")
        return 2
    try:
        if arguments.validate_freeze:
            write_freeze_evidence(project_root)
            print("Phase 2F experiment matrix, environment, and C/D pairing validation passed.")
        elif arguments.summarize_smoke:
            summarize_smoke_runs(project_root)
            print("All four Phase 2F smoke runs passed and were summarized.")
        else:
            manifest = run_experiment(project_root, arguments.experiment, smoke=arguments.smoke)
            print(
                f"Experiment {arguments.experiment} {manifest['run_kind']} run completed: "
                f"{manifest['output_directory']}"
            )
    except (ExperimentValidationError, ProtocolValidationError) as error:
        print(f"Phase 2F validation failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
