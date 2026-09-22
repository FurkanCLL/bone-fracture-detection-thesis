from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Mapping

import yaml

from bone_fracture_audit.audit import sha256_file
from bone_fracture_pipeline.prepare_dataset import CLASS_NAMES, SPLITS, fingerprint_dataset


TOOL_VERSION = "1.0.0"
REQUIRED_TRAINING_SETTINGS = {
    "imgsz": 640,
    "epochs": 100,
    "batch": 8,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "weight_decay": 0.0005,
    "seed": 42,
    "deterministic": True,
    "amp": False,
    "patience": 0,
    "val": True,
    "pretrained": True,
    "device": 0,
    "project": "outputs/training",
}
AUGMENTATION_OFF_SETTINGS = {
    "augment": False,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 0,
    "auto_augment": None,
    "erasing": 0.0,
}


class ProtocolValidationError(RuntimeError):
    """Raised when the frozen training protocol does not pass validation."""


def load_protocol_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ProtocolValidationError(f"Baseline configuration does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ProtocolValidationError("Baseline configuration must contain a YAML mapping.")
    return data


# Hashes parsed content so harmless YAML formatting changes do not alter the protocol identity.
def protocol_digest(config: Mapping[str, object]) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_protocol_config(config: Mapping[str, object]) -> dict[str, object]:
    if config.get("protocol_version") != 1:
        raise ProtocolValidationError("protocol_version must be 1.")

    dataset = _mapping(config, "dataset")
    expected_classes = dataset.get("expected_classes")
    if tuple(expected_classes or ()) != CLASS_NAMES:
        raise ProtocolValidationError("The baseline must define the six approved classes in canonical order.")

    model = _mapping(config, "model")
    expected_model = {
        "architecture": "yolov8s",
        "weights": "yolov8s.pt",
        "initialization": "coco_pretrained",
        "task": "detect",
    }
    _require_values(model, expected_model, "model")

    training = _mapping(config, "training")
    _require_values(training, REQUIRED_TRAINING_SETTINGS, "training")

    augmentation = _mapping(config, "augmentation")
    _require_values(augmentation, AUGMENTATION_OFF_SETTINGS, "augmentation")
    for name, value in augmentation.items():
        if name not in AUGMENTATION_OFF_SETTINGS:
            raise ProtocolValidationError(f"Unsupported baseline augmentation setting: {name}")
        if isinstance(value, bool) and name != "augment":
            raise ProtocolValidationError(f"Augmentation setting {name} must be numeric or null, not boolean.")

    split_policy = _mapping(dataset, "split_policy")
    expected_split_policy = {
        "training": "train",
        "validation": "valid",
        "checkpoint_selection": "valid",
        "test": "test",
        "test_usage": "final_evaluation_only",
    }
    _require_values(split_policy, expected_split_policy, "dataset.split_policy")

    evaluation = _mapping(config, "evaluation")
    if evaluation.get("primary_metric") != "metrics/mAP50-95(B)":
        raise ProtocolValidationError("Validation mAP50-95 must remain the primary metric.")
    checkpoint = _mapping(evaluation, "checkpoint_selection")
    _require_values(
        checkpoint,
        {
            "split": "valid",
            "mode": "max",
            "artifact": "best.pt",
            "requires_framework_fitness_verification": True,
        },
        "evaluation.checkpoint_selection",
    )
    if "test" in {split_policy["training"], split_policy["validation"], checkpoint["split"]}:
        raise ProtocolValidationError("The test split cannot be used for fitting or model selection.")

    seed_policy = _mapping(config, "seed_policy")
    if seed_policy.get("primary") != 42 or seed_policy.get("conditional_final_seeds") != [42, 43, 44]:
        raise ProtocolValidationError("The approved seed policy is primary seed 42 and conditional seeds 42/43/44.")

    return {
        "valid": True,
        "protocol_digest": protocol_digest(config),
        "augmentation_settings_checked": sorted(AUGMENTATION_OFF_SETTINGS),
        "test_isolation": True,
    }


def validate_prepared_dataset(project_root: Path, config: Mapping[str, object]) -> dict[str, object]:
    dataset = _mapping(config, "dataset")
    dataset_root = _resolve_project_path(project_root, dataset.get("root"), "dataset.root")
    data_yaml = _resolve_project_path(project_root, dataset.get("data_yaml"), "dataset.data_yaml")
    if not dataset_root.is_dir():
        raise ProtocolValidationError(f"Prepared dataset does not exist: {dataset_root}")
    if not data_yaml.is_file():
        raise ProtocolValidationError(f"Prepared data.yaml does not exist: {data_yaml}")

    yaml_data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(yaml_data, dict):
        raise ProtocolValidationError("Prepared data.yaml must contain a YAML mapping.")
    names = yaml_data.get("names")
    if isinstance(names, dict):
        try:
            class_names = tuple(names[index] for index in range(len(names)))
        except (KeyError, TypeError) as error:
            raise ProtocolValidationError("Prepared data.yaml class IDs must be consecutive from zero.") from error
    elif isinstance(names, list):
        class_names = tuple(names)
    else:
        raise ProtocolValidationError("Prepared data.yaml must define class names.")
    if class_names != CLASS_NAMES:
        raise ProtocolValidationError("Prepared data.yaml does not contain the six approved classes.")

    expected_paths = {"train": "train/images", "val": "valid/images", "test": "test/images"}
    for key, expected in expected_paths.items():
        if yaml_data.get(key) != expected:
            raise ProtocolValidationError(f"Prepared data.yaml {key!r} path must be {expected!r}.")
    for split in SPLITS:
        for folder in ("images", "labels"):
            path = dataset_root / split / folder
            if not path.is_dir():
                raise ProtocolValidationError(f"Prepared dataset directory is missing: {path}")

    fingerprint = fingerprint_dataset(dataset_root)
    expected_fingerprint = dataset.get("expected_fingerprint")
    if fingerprint.digest != expected_fingerprint:
        raise ProtocolValidationError(
            "Prepared dataset fingerprint does not match the Phase 2B approval: "
            f"{fingerprint.digest} != {expected_fingerprint}"
        )
    return {
        "valid": True,
        "root": _display_path(project_root, dataset_root),
        "data_yaml": _display_path(project_root, data_yaml),
        "class_count": len(class_names),
        "class_names": list(class_names),
        "file_count": fingerprint.file_count,
        "fingerprint": fingerprint.digest,
        "phase2b_fingerprint_match": True,
    }


def verify_ultralytics_semantics(config: Mapping[str, object]) -> dict[str, object]:
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
    from ultralytics.utils.metrics import Metric
    from ultralytics.utils.torch_utils import EarlyStopping

    framework = _mapping(config, "framework")
    if ultralytics.__version__ != str(framework.get("ultralytics_version")):
        raise ProtocolValidationError(
            f"Ultralytics version drift: {ultralytics.__version__} != {framework.get('ultralytics_version')}"
        )
    if not torch.__version__.startswith(f"{framework.get('pytorch_version')}+"):
        raise ProtocolValidationError(
            f"PyTorch version drift: {torch.__version__} != {framework.get('pytorch_version')} with CUDA build"
        )

    missing_keys = sorted(name for name in AUGMENTATION_OFF_SETTINGS if name not in DEFAULT_CFG_DICT)
    if missing_keys:
        raise ProtocolValidationError(f"Installed Ultralytics does not support settings: {missing_keys}")

    # Let Ultralytics parse the exact overrides so accepted names and types are checked by the installed version.
    parsed = get_cfg(
        overrides={
            "task": "detect",
            "mode": "train",
            **dict(_mapping(config, "training")),
            **dict(_mapping(config, "augmentation")),
        }
    )
    for name, expected in AUGMENTATION_OFF_SETTINGS.items():
        if getattr(parsed, name) != expected:
            raise ProtocolValidationError(f"Ultralytics changed the configured value for {name}.")

    stopper = EarlyStopping(patience=0)
    if not math.isinf(stopper.patience):
        raise ProtocolValidationError("Installed Ultralytics does not interpret patience=0 as disabled.")

    metric = Metric()
    metric.p = np.array([0.11, 0.12])
    metric.r = np.array([0.21, 0.22])
    metric.all_ap = np.array([[0.31] + [0.41] * 9, [0.32] + [0.42] * 9])
    if not math.isclose(metric.fitness(), metric.map, rel_tol=0.0, abs_tol=1e-12):
        raise ProtocolValidationError("Ultralytics best-checkpoint fitness is not exactly detection mAP50-95.")

    defaults = {name: DEFAULT_CFG_DICT[name] for name in AUGMENTATION_OFF_SETTINGS}
    return {
        "valid": True,
        "ultralytics_version": ultralytics.__version__,
        "pytorch_version": torch.__version__,
        "installed_config_parser_accepted_overrides": True,
        "supported_augmentation_keys": sorted(AUGMENTATION_OFF_SETTINGS),
        "installed_augmentation_defaults": defaults,
        "patience_zero_disables_early_stopping": True,
        "best_pt_fitness_equals_map50_95": True,
    }


def instantiate_pretrained_model(project_root: Path, config: Mapping[str, object]) -> dict[str, object]:
    from ultralytics import YOLO

    model_config = _mapping(config, "model")
    weights_name = str(model_config["weights"])
    with _working_directory(project_root):
        model = YOLO(weights_name)
    weights_path = Path(model.ckpt_path)
    if not weights_path.is_absolute():
        weights_path = project_root / weights_path
    weights_path = weights_path.resolve()
    if model.task != "detect" or not weights_path.is_file():
        raise ProtocolValidationError("YOLOv8s detection model or pretrained weights could not be initialized.")
    return {
        "valid": True,
        "task": model.task,
        "weights_file": weights_path.name,
        "weights_path": _display_path(project_root, weights_path),
        "weights_size_bytes": weights_path.stat().st_size,
        "weights_sha256": sha256_file(weights_path),
    }


def collect_environment_metadata(
    project_root: Path,
    config: Mapping[str, object],
    weights_path: Path,
) -> dict[str, object]:
    _configure_ultralytics(project_root)
    import cv2
    import numpy as np
    import PIL
    import torch
    import ultralytics

    cuda_available = torch.cuda.is_available()
    gpu: dict[str, object] | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "vram_bytes": properties.total_memory,
            "vram_gib": round(properties.total_memory / (1024**3), 3),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "driver_version": _nvidia_driver_version(),
        }

    training = _mapping(config, "training")
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "libraries": {
            "pytorch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "pyyaml": _package_version("PyYAML"),
        },
        "cuda": {
            "available": cuda_available,
            "runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
        },
        "gpu": gpu,
        "determinism": {
            "protocol_enabled": training["deterministic"],
            "seed": training["seed"],
            "torch_deterministic_algorithms_currently_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic_currently_enabled": torch.backends.cudnn.deterministic,
            "cudnn_benchmark_currently_enabled": torch.backends.cudnn.benchmark,
        },
        "planned_batch_size": training["batch"],
        "batch_size_status": "planned_not_smoke_tested",
        "weights": {
            "filename": weights_path.name,
            "path": _display_path(project_root, weights_path),
            "size_bytes": weights_path.stat().st_size,
            "sha256": sha256_file(weights_path),
        },
    }


def run_protocol_validation(project_root: Path, config_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    _configure_ultralytics(project_root)
    config = load_protocol_config(config_path)
    config_result = validate_protocol_config(config)
    dataset_result = validate_prepared_dataset(project_root, config)
    framework_result = verify_ultralytics_semantics(config)
    model_result = instantiate_pretrained_model(project_root, config)
    weights_path = (project_root / str(model_result["weights_path"])).resolve()
    environment = collect_environment_metadata(project_root, config, weights_path)
    result = {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "success": True,
        "official_training_performed": False,
        "config_path": _display_path(project_root, config_path),
        "config": config_result,
        "dataset": dataset_result,
        "framework": framework_result,
        "model": model_result,
        "test_set_isolated": True,
        "batch_size_8_status": "configuration_and_gpu_detected; VRAM stability deferred to Phase 2F smoke test",
    }
    return result, environment


def write_evidence(
    result: Mapping[str, object],
    environment: Mapping[str, object],
    output_directories: list[Path],
) -> None:
    for directory in output_directories:
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "protocol_validation.json", result)
        _write_json(directory / "training_environment.json", environment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the frozen Phase 2C YOLOv8 baseline protocol.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/training/baseline.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/phase2/phase2c"))
    parser.add_argument("--canonical-output", type=Path, default=Path("docs/evidence/phase2c"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    output = args.output if args.output.is_absolute() else project_root / args.output
    canonical_output = (
        args.canonical_output if args.canonical_output.is_absolute() else project_root / args.canonical_output
    )
    try:
        result, environment = run_protocol_validation(project_root, config_path)
        write_evidence(result, environment, [output, canonical_output])
    except ProtocolValidationError as error:
        print(f"Protocol validation failed: {error}")
        return 1
    print(f"Phase 2C protocol validation passed. Evidence written to {output} and {canonical_output}.")
    return 0


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"Configuration section {key!r} must be a mapping.")
    return value


def _require_values(actual: Mapping[str, object], expected: Mapping[str, object], section: str) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise ProtocolValidationError(f"Missing {section}.{key}.")
        actual_value = actual[key]
        if actual_value != expected_value or type(actual_value) is not type(expected_value):
            raise ProtocolValidationError(
                f"Unexpected {section}.{key}: {actual_value!r}; expected {expected_value!r}."
            )


def _resolve_project_path(project_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProtocolValidationError(f"{label} must be a non-empty path string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _configure_ultralytics(project_root: Path) -> None:
    config_directory = project_root / "outputs" / "phase2" / "phase2c" / "ultralytics_config"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_directory))


def _nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else None


def _package_version(distribution_name: str) -> str:
    from importlib.metadata import version

    return version(distribution_name)


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
