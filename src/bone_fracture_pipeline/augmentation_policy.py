from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence
from unittest.mock import patch

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from bone_fracture_audit.audit import _read_dataset_config, sha256_file
from bone_fracture_audit.yolo import Annotation, parse_yolo_label
from bone_fracture_pipeline.prepare_dataset import (
    CANONICAL_EXPECTATIONS,
    CLASS_NAMES,
    SPLITS,
    DatasetExpectations,
    fingerprint_dataset,
)
from bone_fracture_pipeline.preprocess_dataset import (
    _display_path,
    _finalize_directories,
    _remove_build_directory,
    _repository_commit,
    _write_csv,
    _write_json,
)
from bone_fracture_pipeline.preprocessing_core import PreprocessingError, channels_are_identical


TOOL_VERSION = "1.0.0"
APPROVED_CLAHE_FINGERPRINT = "ca8286b35d3d31f0b8074aa387a44ca6392178926c23bde61ea6d99be70aba5f"
AUGMENTATION_SETTINGS = {
    "augment": False,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.15,
    "degrees": 10.0,
    "translate": 0.05,
    "scale": 0.10,
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


@dataclass(frozen=True)
class MatchedSample:
    split: str
    stem: str
    control_image: Path
    clahe_image: Path
    control_label: Path
    clahe_label: Path
    width: int
    height: int
    annotations: tuple[Annotation, ...]

    @property
    def sample_id(self) -> str:
        return f"{self.split}/{self.stem}"

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(sorted({annotation.class_id for annotation in self.annotations}))

    @property
    def smallest_box_area(self) -> float:
        if not self.annotations:
            return math.inf
        return min(annotation.width * annotation.height for annotation in self.annotations)

    @property
    def closest_boundary(self) -> float:
        if not self.annotations:
            return math.inf
        distances = []
        for annotation in self.annotations:
            left = annotation.x_center - annotation.width / 2
            top = annotation.y_center - annotation.height / 2
            right = annotation.x_center + annotation.width / 2
            bottom = annotation.y_center + annotation.height / 2
            distances.append(min(left, top, 1 - right, 1 - bottom))
        return min(distances)


@dataclass(frozen=True)
class PreviewSpec:
    name: str
    reason: str
    sample: MatchedSample
    angle: float = 0.0
    translate_x: float = 0.0
    translate_y: float = 0.0
    scale: float = 1.0
    intensity_gain: float = 0.0


def load_augmentation_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise PreprocessingError(f"Augmentation configuration does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise PreprocessingError("Augmentation configuration must contain a YAML mapping.")
    return loaded


# Freezes only the experimental augmentation factor and rejects accidental baseline drift.
def validate_augmentation_config(config: Mapping[str, object]) -> dict[str, object]:
    if config.get("policy_version") != 1:
        raise PreprocessingError("policy_version must be 1.")
    if config.get("inherits") != "configs/training/baseline.yaml":
        raise PreprocessingError("The conservative policy must inherit the frozen baseline.")
    if config.get("applies_to_experiments") != ["C", "D"]:
        raise PreprocessingError("The augmentation-on policy must apply only to Experiments C and D.")

    split_policy = _mapping(config, "split_policy")
    if split_policy != {"train": True, "valid": False, "test": False}:
        raise PreprocessingError("Augmentation must be enabled only for the training split.")
    augmentation = _mapping(config, "augmentation")
    if dict(augmentation) != AUGMENTATION_SETTINGS:
        raise PreprocessingError("Augmentation values differ from the approved conservative policy.")

    reproducibility = _mapping(config, "reproducibility")
    expected_reproducibility = {
        "seed": 42,
        "deterministic": True,
        "workers": 8,
        "cache": False,
        "rect": False,
        "require_fresh_run": True,
        "stable_sample_identity": "split_and_source_stem",
        "require_matching_sample_order_for_c_and_d": True,
        "require_albumentations_absent": True,
    }
    if dict(reproducibility) != expected_reproducibility:
        raise PreprocessingError("Augmentation reproducibility settings differ from the frozen baseline policy.")

    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "valid": True,
        "config_digest": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "training_only": True,
        "settings_checked": sorted(AUGMENTATION_SETTINGS),
    }


# Compares every sample by split and stem so path extensions cannot hide identity drift.
def compare_dataset_identities(
    control_root: Path,
    clahe_root: Path,
    *,
    expectations: DatasetExpectations = CANONICAL_EXPECTATIONS,
    expected_clahe_fingerprint: str = APPROVED_CLAHE_FINGERPRINT,
) -> tuple[list[MatchedSample], dict[str, object]]:
    control_root = control_root.resolve()
    clahe_root = clahe_root.resolve()
    for label, root in (("PNG control", control_root), ("CLAHE", clahe_root)):
        if not root.is_dir():
            raise PreprocessingError(f"{label} dataset does not exist: {root}")
        if _read_dataset_config(root / "data.yaml") != expectations.class_names:
            raise PreprocessingError(f"{label} data.yaml changed the approved class mapping.")

    clahe_fingerprint = fingerprint_dataset(clahe_root)
    if clahe_fingerprint.digest != expected_clahe_fingerprint:
        raise PreprocessingError("CLAHE dataset fingerprint no longer matches the Phase 2D approval.")
    control_fingerprint = fingerprint_dataset(control_root)

    matched: list[MatchedSample] = []
    split_counts: dict[str, int] = {}
    valid_class_ids = set(range(len(expectations.class_names)))
    for split in SPLITS:
        control_images = _stem_map(control_root / split / "images", ".png")
        clahe_images = _stem_map(clahe_root / split / "images", ".png")
        control_labels = _stem_map(control_root / split / "labels", ".txt")
        clahe_labels = _stem_map(clahe_root / split / "labels", ".txt")
        expected_keys = set(control_images)
        if not expected_keys or any(
            set(mapping) != expected_keys
            for mapping in (clahe_images, control_labels, clahe_labels)
        ):
            raise PreprocessingError(f"Control and CLAHE sample identities differ in {split}.")
        if len(expected_keys) != expectations.splits[split].images:
            raise PreprocessingError(f"Unexpected sample count in {split}.")

        annotations = 0
        empty_labels = 0
        class_counts: Counter[int] = Counter()
        for stem in sorted(expected_keys):
            control_label = control_labels[stem]
            clahe_label = clahe_labels[stem]
            if sha256_file(control_label) != sha256_file(clahe_label):
                raise PreprocessingError(f"Control and CLAHE labels differ for {split}/{stem}.")
            parsed = parse_yolo_label(control_label, valid_class_ids)
            if parsed.issues or any(annotation.annotation_type != "box" for annotation in parsed.annotations):
                raise PreprocessingError(f"Invalid detection label for {split}/{stem}.")
            annotations += len(parsed.annotations)
            empty_labels += int(parsed.is_empty)
            class_counts.update(annotation.class_id for annotation in parsed.annotations)

            control_image = cv2.imread(str(control_images[stem]), cv2.IMREAD_UNCHANGED)
            clahe_image = cv2.imread(str(clahe_images[stem]), cv2.IMREAD_UNCHANGED)
            if control_image is None or clahe_image is None:
                raise PreprocessingError(f"Could not decode matched sample {split}/{stem}.")
            if control_image.shape != clahe_image.shape or control_image.dtype != clahe_image.dtype:
                raise PreprocessingError(f"Matched image geometry or dtype differs for {split}/{stem}.")
            if control_image.dtype != np.uint8 or control_image.ndim != 3 or control_image.shape[2] != 3:
                raise PreprocessingError(f"PNG control has an unexpected image profile for {split}/{stem}.")
            if not channels_are_identical(clahe_image):
                raise PreprocessingError(f"CLAHE channels differ for {split}/{stem}.")
            height, width = control_image.shape[:2]
            matched.append(
                MatchedSample(
                    split=split,
                    stem=stem,
                    control_image=control_images[stem],
                    clahe_image=clahe_images[stem],
                    control_label=control_label,
                    clahe_label=clahe_label,
                    width=width,
                    height=height,
                    annotations=parsed.annotations,
                )
            )
        expected = expectations.splits[split]
        actual_classes = tuple(class_counts[index] for index in range(len(expectations.class_names)))
        if (
            annotations != expected.annotations
            or empty_labels != expected.empty_labels
            or actual_classes != expected.class_counts
        ):
            raise PreprocessingError(f"Matched labels changed approved counts in {split}.")
        split_counts[split] = len(expected_keys)

    if len(matched) != expectations.images:
        raise PreprocessingError("Matched control/CLAHE sample total is incorrect.")
    return matched, {
        "stable_identity": "<split>/<source stem>",
        "matched_samples": len(matched),
        "split_counts": split_counts,
        "missing_or_extra_identities": 0,
        "label_hash_mismatches": 0,
        "dimension_mismatches": 0,
        "dtype_mismatches": 0,
        "ordered_identities_identical": True,
        "control_dataset_fingerprint": control_fingerprint.digest,
        "control_dataset_file_count": control_fingerprint.file_count,
        "clahe_dataset_fingerprint": clahe_fingerprint.digest,
        "clahe_dataset_file_count": clahe_fingerprint.file_count,
    }


# Uses behavioral checks against the pinned installation instead of trusting option names.
def verify_framework_semantics(project_root: Path, config: Mapping[str, object]) -> dict[str, object]:
    _configure_ultralytics(project_root)
    import torch
    import ultralytics
    from ultralytics.cfg import get_cfg
    from ultralytics.data.augment import RandomHSV, RandomPerspective
    from ultralytics.data.build import build_yolo_dataset

    if ultralytics.__version__ != "8.4.155":
        raise PreprocessingError(f"Ultralytics version drift: {ultralytics.__version__} != 8.4.155")
    parsed = get_cfg(overrides={"task": "detect", "mode": "train", **AUGMENTATION_SETTINGS})
    for name, expected in AUGMENTATION_SETTINGS.items():
        if getattr(parsed, name) != expected:
            raise PreprocessingError(f"Ultralytics changed the configured value for {name}.")

    # Capture the installed affine sampler bounds to prove each configured range directly.
    bounds: list[tuple[float, float]] = []

    def midpoint(lower: float, upper: float) -> float:
        bounds.append((float(lower), float(upper)))
        return (lower + upper) / 2

    affine = RandomPerspective(degrees=10.0, translate=0.05, scale=0.10, shear=0.0, perspective=0.0)
    with patch("ultralytics.data.augment.random.uniform", side_effect=midpoint):
        affine._compute_affine_matrix(np.zeros((100, 200, 3), dtype=np.uint8), (200, 100))
    expected_bounds = [
        (-0.0, 0.0),
        (-0.0, 0.0),
        (-10.0, 10.0),
        (0.9, 1.1),
        (-0.0, 0.0),
        (-0.0, 0.0),
        (0.45, 0.55),
        (0.45, 0.55),
    ]
    if bounds != expected_bounds:
        raise PreprocessingError(f"Installed affine sampling semantics changed: {bounds!r}")

    hsv = RandomHSV(hgain=0.0, sgain=0.0, vgain=0.15)
    bright = {"img": np.full((4, 4, 3), 100, dtype=np.uint8)}
    dark = {"img": np.full((4, 4, 3), 100, dtype=np.uint8)}
    with patch("ultralytics.data.augment.np.random.uniform", return_value=np.array([0.0, 0.0, 1.0])):
        hsv.apply_image(bright)
    with patch("ultralytics.data.augment.np.random.uniform", return_value=np.array([0.0, 0.0, -1.0])):
        hsv.apply_image(dark)
    expected_bright = int(np.clip(np.array([100.0]) * 1.15, 0, 255).astype(np.uint8)[0])
    expected_dark = int(np.clip(np.array([100.0]) * 0.85, 0, 255).astype(np.uint8)[0])
    if not np.all(bright["img"] == expected_bright) or not np.all(dark["img"] == expected_dark):
        raise PreprocessingError("Installed hsv_v behavior is not the expected 0.85-1.15 value multiplier.")

    # Dataset mode, not the prediction-time `augment` flag, isolates training transforms from validation and test.
    calls: list[bool] = []

    class DatasetProbe:
        def __init__(self, **kwargs: object) -> None:
            calls.append(bool(kwargs["augment"]))

    probe_cfg = SimpleNamespace(
        rect=False,
        task="detect",
        fraction=1.0,
        imgsz=640,
        cache=False,
        single_cls=False,
        classes=None,
    )
    with patch("ultralytics.data.build.YOLODataset", DatasetProbe):
        for mode in ("train", "val", "test"):
            build_yolo_dataset(probe_cfg, "unused", 1, {}, mode=mode)
    if calls != [True, False, False]:
        raise PreprocessingError("Installed dataset builder does not isolate augmentation to training mode.")

    albumentations_installed = importlib.util.find_spec("albumentations") is not None
    if albumentations_installed:
        raise PreprocessingError(
            "Albumentations is installed and could activate unapproved default transforms; remove it or define an explicit policy."
        )

    augment_source = Path(importlib.util.find_spec("ultralytics.data.augment").origin)
    build_source = Path(importlib.util.find_spec("ultralytics.data.build").origin)
    return {
        "valid": True,
        "ultralytics_version": ultralytics.__version__,
        "pytorch_version": torch.__version__,
        "config_parser_accepted_values": True,
        "degrees_semantics": "uniform angle in [-10.0, 10.0] degrees",
        "translate_semantics": "independent x/y center translation uniformly within +/-5% of output size",
        "scale_semantics": "uniform multiplicative scale in [0.90, 1.10]",
        "hsv_v_semantics": "uniform multiplicative value gain in [0.85, 1.15] with uint8 clipping",
        "hsv_h_and_hsv_s_disabled": True,
        "augment_flag_semantics": "prediction-time augmentation; retained false",
        "training_transform_activation": "dataset mode train only",
        "validation_augmentation": False,
        "test_augmentation": False,
        "albumentations_installed": False,
        "unapproved_default_albumentations_active": False,
        "augment_source_sha256": sha256_file(augment_source),
        "dataloader_source_sha256": sha256_file(build_source),
    }


# Confirms matching global and worker seeds produce the same geometry sequence for C and D samples.
def verify_paired_reproducibility(
    samples: Sequence[MatchedSample],
    config: Mapping[str, object],
) -> dict[str, object]:
    _configure_ultralytics(Path.cwd())
    import torch
    from ultralytics.data.augment import v8_transforms
    from ultralytics.data.build import seed_worker
    from ultralytics.utils.torch_utils import init_seeds

    selected = _reproducibility_samples(samples)

    def run_sequence(image_attribute: str) -> tuple[list[np.ndarray], list[str], list[str]]:
        init_seeds(42, deterministic=True)
        transforms = v8_transforms(_DatasetStub(), imgsz=640, hyp=_framework_hyp())
        boxes: list[np.ndarray] = []
        python_states: list[str] = []
        numpy_states: list[str] = []
        for sample in selected:
            image = cv2.imread(str(getattr(sample, image_attribute)), cv2.IMREAD_COLOR)
            if image is None:
                raise PreprocessingError(f"Could not decode reproducibility sample {sample.sample_id}.")
            image = _resize_for_training(image, 640)
            transformed = transforms(_framework_labels(image, sample.annotations))
            boxes.append(transformed["instances"].bboxes.copy())
            python_states.append(_state_digest(random.getstate()))
            numpy_states.append(_state_digest(np.random.get_state()))
        return boxes, python_states, numpy_states

    control_boxes, control_python, control_numpy = run_sequence("control_image")
    clahe_boxes, clahe_python, clahe_numpy = run_sequence("clahe_image")
    geometry_equal = all(np.array_equal(left, right) for left, right in zip(control_boxes, clahe_boxes))
    if not geometry_equal or control_python != clahe_python or control_numpy != clahe_numpy:
        raise PreprocessingError("Matched C/D samples did not produce the same stochastic geometry sequence.")

    # The installed worker initializer is checked separately because official runs use eight workers.
    def worker_draws() -> tuple[float, float]:
        torch.manual_seed(123456)
        seed_worker(0)
        return random.random(), float(np.random.random())

    worker_seeding_equal = worker_draws() == worker_draws()
    generator_one = torch.Generator().manual_seed(42)
    generator_two = torch.Generator().manual_seed(42)
    order_equal = torch.equal(
        torch.randperm(len(samples), generator=generator_one),
        torch.randperm(len(samples), generator=generator_two),
    )
    if not worker_seeding_equal or not order_equal:
        raise PreprocessingError("Installed worker or ordering seed behavior was not reproducible.")

    return {
        "success": True,
        "seed": 42,
        "samples_checked": len(selected),
        "sample_ids": [sample.sample_id for sample in selected],
        "transformed_geometry_identical": True,
        "python_rng_states_identical": True,
        "numpy_rng_states_identical": True,
        "shuffle_order_reproducible": True,
        "worker_seeding_reproducible": True,
        "workers": int(_mapping(config, "reproducibility")["workers"]),
        "pairing_conditions": [
            "fresh runs from seed 42",
            "Ultralytics 8.4.155 and unchanged dependency versions",
            "matching split/stem sample order and cardinality",
            "matching labels, dimensions, batch size, workers, cache, and rect settings",
            "no resume from a checkpoint with advanced RNG state",
        ],
        "limitation": (
            "Pairing depends on the frozen loader/version/runtime conditions and must be rechecked in Phase 2F; "
            "it is not guaranteed after worker-count, ordering, resume-state, or framework changes."
        ),
    }


def validate_augmentation_policy(
    project_root: Path,
    config_path: Path,
    control_root: Path,
    clahe_root: Path,
    output: Path,
    canonical_evidence: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    control_root = control_root.resolve()
    clahe_root = clahe_root.resolve()
    output = output.resolve()
    canonical_evidence = canonical_evidence.resolve()
    canonical_files = (
        canonical_evidence / "augmentation_policy_summary.json",
        canonical_evidence / "sample_identity_summary.json",
        canonical_evidence / "preview_index.csv",
    )
    if output.exists() and not overwrite:
        raise PreprocessingError(f"Output already exists: {output}. Use --overwrite to rebuild safely.")
    if any(path.exists() for path in canonical_files) and not overwrite:
        raise PreprocessingError("Canonical Phase 2E evidence exists. Use --overwrite to rebuild safely.")

    config = load_augmentation_config(config_path)
    config_result = validate_augmentation_config(config)
    samples, identity_result = compare_dataset_identities(control_root, clahe_root)
    framework_result = verify_framework_semantics(project_root, config)
    reproducibility_result = verify_paired_reproducibility(samples, config)

    output_build = output.parent / f".{output.name}_building"
    output.parent.mkdir(parents=True, exist_ok=True)
    _remove_build_directory(output_build)
    try:
        output_build.mkdir(parents=True)
        preview_rows = generate_preview_package(samples, output_build / "preview", output / "preview", project_root)
        summary = {
            "schema_version": 1,
            "tool_version": TOOL_VERSION,
            "validated_at_utc": datetime.now(UTC).isoformat(),
            "repository_commit_sha": _repository_commit(project_root),
            "config_path": _display_path(config_path, project_root),
            "config": config_result,
            "augmentation": dict(_mapping(config, "augmentation")),
            "split_policy": dict(_mapping(config, "split_policy")),
            "dataset_identity": identity_result,
            "framework_semantics": framework_result,
            "paired_reproducibility": reproducibility_result,
            "preview": {
                "images": len(preview_rows),
                "split_scope": ["train"],
                "test_images": 0,
                "all_boxes_valid": True,
                "invalid_boxes": 0,
                "package_generated": True,
            },
            "unwanted_transforms_disabled": [
                "horizontal_flip",
                "vertical_flip",
                "shear",
                "perspective",
                "mosaic",
                "mixup",
                "cutmix",
                "copy_paste",
                "bgr_channel_swap",
                "auto_augment",
                "random_erasing",
                "third_party_albumentations_defaults",
            ],
            "official_training_performed": False,
            "technically_ready_for_manual_preview_review": True,
        }
        _write_csv(output_build / "preview_index.csv", preview_rows)
        _write_json(output_build / "sample_identity_summary.json", identity_result)
        _write_json(output_build / "augmentation_policy_summary.json", summary)
        _finalize_directories(((output_build, output),), overwrite=overwrite)

        canonical_evidence.mkdir(parents=True, exist_ok=True)
        _write_json(canonical_files[0], summary)
        _write_json(canonical_files[1], identity_result)
        shutil.copyfile(output / "preview_index.csv", canonical_files[2])
        return summary
    except Exception:
        _remove_build_directory(output_build)
        raise


# Renders fixed endpoint cases to make clipping and box alignment easy to inspect.
def generate_preview_package(
    samples: Sequence[MatchedSample],
    preview_build: Path,
    preview_final: Path,
    project_root: Path,
) -> list[dict[str, object]]:
    preview_build.mkdir(parents=True)
    specs = _select_preview_specs(samples)
    rows: list[dict[str, object]] = []
    rendered: list[Path] = []
    for index, spec in enumerate(specs, start=1):
        output = preview_build / f"{index:02d}_{spec.name}_{spec.sample.stem}.png"
        result = _render_preview(spec, output)
        rendered.append(output)
        rows.append(
            {
                "review_order": index,
                "sample_id": spec.sample.sample_id,
                "split": spec.sample.split,
                "preview_case": spec.name,
                "selection_reason": spec.reason,
                "angle_degrees": spec.angle,
                "translate_x_fraction": spec.translate_x,
                "translate_y_fraction": spec.translate_y,
                "scale_factor": spec.scale,
                "intensity_gain": spec.intensity_gain,
                "source_boxes": len(spec.sample.annotations),
                "output_boxes": result["output_boxes"],
                "clipped_boxes": result["clipped_boxes"],
                "invalid_boxes": 0,
                "preview_path": _display_path(preview_final / output.name, project_root),
            }
        )
    _render_contact_sheet(rendered, preview_build / "contact_sheet.png")
    return rows


def _select_preview_specs(samples: Sequence[MatchedSample]) -> list[PreviewSpec]:
    eligible = sorted(
        (sample for sample in samples if sample.split == "train" and sample.annotations),
        key=lambda sample: sample.sample_id,
    )
    by_class = {
        class_id: next(sample for sample in eligible if class_id in sample.class_ids)
        for class_id in range(len(CLASS_NAMES))
    }
    smallest = min(eligible, key=lambda sample: (sample.smallest_box_area, sample.sample_id))
    boundary = min(eligible, key=lambda sample: (sample.closest_boundary, sample.sample_id))
    multi = max(eligible, key=lambda sample: (len(sample.annotations), tuple(-ord(c) for c in sample.sample_id)))
    multi_class = next((sample for sample in eligible if len(sample.class_ids) > 1), None)
    specs = [
        PreviewSpec("rotation_positive", "class_0_representative", by_class[0], angle=10.0),
        PreviewSpec("rotation_negative", "class_1_representative", by_class[1], angle=-10.0),
        PreviewSpec("translation_positive", "class_2_representative", by_class[2], translate_x=0.05, translate_y=-0.05),
        PreviewSpec("translation_boundary", "box_near_boundary", boundary, translate_x=-0.05, translate_y=0.05),
        PreviewSpec("zoom_in", "small_box", smallest, scale=1.10),
        PreviewSpec("zoom_out", "class_3_representative", by_class[3], scale=0.90),
        PreviewSpec("intensity_positive", "class_4_representative", by_class[4], intensity_gain=0.15),
        PreviewSpec("intensity_negative", "class_5_representative", by_class[5], intensity_gain=-0.15),
        PreviewSpec(
            "combined_policy",
            "multi_annotation",
            multi,
            angle=7.0,
            translate_x=0.03,
            translate_y=-0.04,
            scale=1.06,
            intensity_gain=0.10,
        ),
    ]
    if multi_class is not None:
        specs.append(
            PreviewSpec(
                "combined_multi_class",
                "multiple_classes",
                multi_class,
                angle=-6.0,
                translate_x=-0.03,
                translate_y=0.02,
                scale=0.95,
                intensity_gain=-0.08,
            )
        )
    return specs


def _render_preview(spec: PreviewSpec, output: Path) -> dict[str, int]:
    _configure_ultralytics(Path.cwd())
    from ultralytics.data.augment import RandomPerspective

    image = cv2.imread(str(spec.sample.control_image), cv2.IMREAD_COLOR)
    if image is None:
        raise PreprocessingError(f"Could not decode preview source {spec.sample.control_image}.")
    height, width = image.shape[:2]
    labels = _framework_labels(image.copy(), spec.sample.annotations)
    transform = RandomPerspective(
        degrees=10.0,
        translate=0.05,
        scale=0.10,
        shear=0.0,
        perspective=0.0,
        size=(width, height),
    )
    matrix = _fixed_affine_matrix(
        width,
        height,
        angle=spec.angle,
        translate_x=spec.translate_x,
        translate_y=spec.translate_y,
        scale=spec.scale,
    )
    params = {"M": matrix, "scale": spec.scale, "orig_shape": (height, width), "size": (width, height)}
    transform.apply_image(labels, params)
    transform.apply_instances(labels, params)
    if spec.intensity_gain:
        labels["img"] = _apply_fixed_value_gain(labels["img"], spec.intensity_gain)

    transformed_boxes = labels["instances"].bboxes
    _validate_absolute_boxes(transformed_boxes, width, height)
    source_boxes = _normalized_boxes_to_absolute(spec.sample.annotations, width, height)
    clipped = sum(
        int(left <= 0 or top <= 0 or right >= width or bottom >= height)
        for left, top, right, bottom in transformed_boxes
    )
    _draw_preview_canvas(spec, image, labels["img"], source_boxes, transformed_boxes, output)
    return {"output_boxes": len(transformed_boxes), "clipped_boxes": clipped}


def _fixed_affine_matrix(
    width: int,
    height: int,
    *,
    angle: float,
    translate_x: float,
    translate_y: float,
    scale: float,
) -> np.ndarray:
    center = np.eye(3, dtype=np.float32)
    center[0, 2] = -width / 2
    center[1, 2] = -height / 2
    rotation = np.eye(3, dtype=np.float32)
    rotation[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale)
    translation = np.eye(3, dtype=np.float32)
    translation[0, 2] = (0.5 + translate_x) * width
    translation[1, 2] = (0.5 + translate_y) * height
    return translation @ rotation @ center


# Mirrors the pinned RandomHSV value LUT with a fixed QA gain instead of a random draw.
def _apply_fixed_value_gain(image: np.ndarray, gain: float) -> np.ndarray:
    if not -0.15 <= gain <= 0.15:
        raise PreprocessingError("Preview intensity gain exceeds the approved range.")
    x = np.arange(256, dtype=np.float64)
    lut = np.clip(x * (1.0 + gain), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.LUT(hsv[:, :, 2], lut)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _draw_preview_canvas(
    spec: PreviewSpec,
    original: np.ndarray,
    augmented: np.ndarray,
    source_boxes: np.ndarray,
    transformed_boxes: np.ndarray,
    output: Path,
) -> None:
    original_pil = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    augmented_pil = Image.fromarray(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
    max_width, max_height = 680, 680
    scale = min(max_width / original_pil.width, max_height / original_pil.height, 1.0)
    display_size = (max(1, round(original_pil.width * scale)), max(1, round(original_pil.height * scale)))
    if original_pil.size != display_size:
        original_pil = original_pil.resize(display_size, Image.Resampling.LANCZOS)
        augmented_pil = augmented_pil.resize(display_size, Image.Resampling.LANCZOS)

    header = 88
    footer = 24
    gap = 12
    canvas = Image.new("RGB", (display_size[0] * 2 + gap, display_size[1] + header + footer), "black")
    canvas.paste(original_pil, (0, header))
    canvas.paste(augmented_pil, (display_size[0] + gap, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 6), f"{spec.sample.sample_id} | {spec.name}", fill="white", font=font)
    draw.text((8, 24), f"reason: {spec.reason}", fill="white", font=font)
    draw.text(
        (8, 42),
        f"angle={spec.angle:+.1f}  tx={spec.translate_x:+.2f}  ty={spec.translate_y:+.2f}  "
        f"scale={spec.scale:.2f}  value={spec.intensity_gain:+.2f}",
        fill="white",
        font=font,
    )
    draw.text((8, 66), "Original PNG control", fill=(160, 220, 255), font=font)
    draw.text((display_size[0] + gap + 8, 66), "Augmented", fill=(180, 255, 180), font=font)
    _draw_boxes(draw, source_boxes * scale, 0, header, (0, 255, 255))
    _draw_boxes(draw, transformed_boxes * scale, display_size[0] + gap, header, (255, 80, 220))
    draw.text((8, header + display_size[1] + 5), "Technical alignment QA only", fill="white", font=font)
    canvas.save(output, format="PNG", optimize=True)


def _draw_boxes(
    draw: ImageDraw.ImageDraw,
    boxes: np.ndarray,
    x_offset: int,
    y_offset: int,
    color: tuple[int, int, int],
) -> None:
    for left, top, right, bottom in boxes:
        draw.rectangle(
            (left + x_offset, top + y_offset, right + x_offset, bottom + y_offset),
            outline=color,
            width=2,
        )


def _render_contact_sheet(paths: Sequence[Path], output: Path) -> None:
    columns = 2
    cell_width, cell_height = 520, 300
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (24, 24, 24))
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            thumbnail = opened.convert("RGB")
            thumbnail.thumbnail((cell_width - 10, cell_height - 10), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_width + (cell_width - thumbnail.width) // 2
        y = (index // columns) * cell_height + (cell_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (x, y))
    sheet.save(output, format="PNG", optimize=True)


def _framework_labels(image: np.ndarray, annotations: Sequence[Annotation]) -> dict[str, object]:
    _configure_ultralytics(Path.cwd())
    from ultralytics.utils.instance import Instances

    boxes = np.array(
        [[a.x_center, a.y_center, a.width, a.height] for a in annotations],
        dtype=np.float32,
    ).reshape(-1, 4)
    classes = np.array([a.class_id for a in annotations], dtype=np.float32).reshape(-1, 1)
    instances = Instances(
        bboxes=boxes,
        segments=np.zeros((0, 1000, 2), dtype=np.float32),
        keypoints=None,
        bbox_format="xywh",
        normalized=True,
    )
    return {"img": image, "cls": classes, "instances": instances}


def _framework_hyp() -> SimpleNamespace:
    return SimpleNamespace(
        **AUGMENTATION_SETTINGS,
        copy_paste_mode="flip",
        augmentations=None,
    )


class _DatasetStub:
    data = {"flip_idx": []}
    use_keypoints = False
    use_obb = False
    cache = None
    buffer: list[int] = []


def _reproducibility_samples(samples: Sequence[MatchedSample]) -> list[MatchedSample]:
    annotated = [sample for sample in samples if sample.split == "train" and sample.annotations]
    selected: dict[str, MatchedSample] = {}
    for class_id in range(len(CLASS_NAMES)):
        sample = next(sample for sample in annotated if class_id in sample.class_ids)
        selected[sample.sample_id] = sample
    for sample in sorted(annotated, key=lambda value: value.sample_id)[:6]:
        selected[sample.sample_id] = sample
    return [selected[key] for key in sorted(selected)]


def _resize_for_training(image: np.ndarray, image_size: int) -> np.ndarray:
    ratio = image_size / max(image.shape[:2])
    if ratio == 1.0:
        return image
    width = max(1, round(image.shape[1] * ratio))
    height = max(1, round(image.shape[0] * ratio))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _normalized_boxes_to_absolute(
    annotations: Sequence[Annotation],
    width: int,
    height: int,
) -> np.ndarray:
    return np.array(
        [
            [
                (a.x_center - a.width / 2) * width,
                (a.y_center - a.height / 2) * height,
                (a.x_center + a.width / 2) * width,
                (a.y_center + a.height / 2) * height,
            ]
            for a in annotations
        ],
        dtype=np.float32,
    ).reshape(-1, 4)


def _validate_absolute_boxes(boxes: np.ndarray, width: int, height: int) -> None:
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise PreprocessingError("Augmentation produced malformed bounding boxes.")
    for left, top, right, bottom in boxes:
        if left < 0 or top < 0 or right > width or bottom > height or right <= left or bottom <= top:
            raise PreprocessingError("Augmentation produced an invalid clipped bounding box.")


def _stem_map(directory: Path, suffix: str) -> dict[str, Path]:
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == suffix]
    mapping = {path.stem.casefold(): path for path in paths}
    if len(mapping) != len(paths):
        raise PreprocessingError(f"Ambiguous case-insensitive stems in {directory}.")
    return mapping


def _state_digest(state: object) -> str:
    return hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()


def _configure_ultralytics(project_root: Path) -> None:
    config_directory = project_root / "outputs" / "phase2" / "phase2e" / "ultralytics_config"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_directory)


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise PreprocessingError(f"Configuration section {key!r} must be a mapping.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and freeze the Phase 2E augmentation policy.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/augmentation_conservative.yaml"),
    )
    parser.add_argument("--control", type=Path, default=Path("data/prepared/v3_detection_png"))
    parser.add_argument("--clahe", type=Path, default=Path("data/prepared/v3_detection_clahe"))
    parser.add_argument("--output", type=Path, default=Path("outputs/phase2/phase2e/augmentation"))
    parser.add_argument("--canonical-evidence", type=Path, default=Path("docs/evidence/phase2e"))
    parser.add_argument("--overwrite", action="store_true", help="Safely rebuild and replace existing evidence.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = arguments.project_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()

    try:
        summary = validate_augmentation_policy(
            project_root,
            resolve(arguments.config),
            resolve(arguments.control),
            resolve(arguments.clahe),
            resolve(arguments.output),
            resolve(arguments.canonical_evidence),
            overwrite=arguments.overwrite,
        )
    except PreprocessingError as error:
        print(f"Phase 2E augmentation validation failed: {error}")
        return 1
    print("Phase 2E augmentation validation completed without training.")
    print(f"Matched C/D samples: {summary['dataset_identity']['matched_samples']}")
    print(f"Preview images: {summary['preview']['images']} (train only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
