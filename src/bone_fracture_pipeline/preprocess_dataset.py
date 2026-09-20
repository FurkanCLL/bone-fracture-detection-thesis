from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from bone_fracture_audit.audit import (
    IMAGE_EXTENSIONS,
    _read_dataset_config,
    match_images_and_labels,
    sha256_file,
)
from bone_fracture_audit.yolo import Annotation, ParsedLabel, parse_yolo_label
from bone_fracture_pipeline.prepare_dataset import (
    CANONICAL_EXPECTATIONS,
    CLASS_NAMES,
    SPLITS,
    DatasetExpectations,
    fingerprint_dataset,
)
from bone_fracture_pipeline.preprocessing_core import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    PNG_COMPRESSION,
    PNG_SIGNATURE,
    PreprocessingError,
    ProcessedImageRecord,
    channels_are_identical,
    decode_without_orientation_transform as _decode_without_orientation_transform,
    preprocess_image_array,
    validate_processed_array,
)
from bone_fracture_pipeline.preprocessing_review import (
    render_visual_review as _render_visual_review,
    select_visual_review,
)


TOOL_VERSION = "1.0.0"
APPROVED_SOURCE_FINGERPRINT = "c5031d9937e2a1d917a2369f327b38a59a4b5e8bd40ae2da659451840c7d41b1"


@dataclass(frozen=True)
class DetectionPair:
    split: str
    image_path: Path
    label_path: Path
    parsed_label: ParsedLabel




def preprocess_dataset(
    source: Path,
    output: Path,
    artifacts: Path,
    canonical_evidence: Path,
    *,
    overwrite: bool = False,
    expectations: DatasetExpectations = CANONICAL_EXPECTATIONS,
    expected_source_fingerprint: str = APPROVED_SOURCE_FINGERPRINT,
    project_root: Path | None = None,
) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    artifacts = artifacts.resolve()
    canonical_evidence = canonical_evidence.resolve()
    project_root = (project_root or Path.cwd()).resolve()
    _validate_locations(source, output, artifacts, canonical_evidence)
    _validate_expected_shape(expectations)

    finals = (output, artifacts, canonical_evidence)
    for path in finals:
        if path.exists() and not overwrite:
            raise PreprocessingError(f"Output already exists: {path}. Use --overwrite to rebuild safely.")

    source_before = fingerprint_dataset(source)
    if source_before.digest != expected_source_fingerprint:
        raise PreprocessingError(
            "Approved detection dataset fingerprint changed: "
            f"{source_before.digest} != {expected_source_fingerprint}"
        )
    pairs, source_split_summary = _inspect_source_dataset(source, expectations)

    output_build = output.parent / f".{output.name}_building"
    artifacts_build = artifacts.parent / f".{artifacts.name}_building"
    canonical_build = canonical_evidence.parent / f".{canonical_evidence.name}_building"
    builds = (output_build, artifacts_build, canonical_build)
    _prepare_build_locations(builds, finals)

    cv2.setNumThreads(1)
    try:
        output_build.mkdir(parents=True)
        artifacts_build.mkdir(parents=True)
        canonical_build.mkdir(parents=True)

        records = _build_preprocessed_dataset(pairs, output_build)
        _write_data_yaml(output_build / "data.yaml", expectations.class_names)
        output_validation = _validate_output_dataset(output_build, records, expectations)
        output_fingerprint = fingerprint_dataset(output_build)

        manifest_rows = _manifest_rows(records, source, output, project_root)
        intensity_rows = _intensity_summary_rows(records)
        selected_review = select_visual_review(records, class_count=len(expectations.class_names))
        visual_rows = _render_visual_review(
            source,
            output_build,
            artifacts_build / "visual_review",
            artifacts / "visual_review",
            selected_review,
            expectations.class_names,
            project_root,
        )

        reproduction = _verify_reproducibility(
            output.parent,
            pairs,
            expectations,
            output_fingerprint.digest,
        )
        source_after = fingerprint_dataset(source)
        if source_after.digest != source_before.digest:
            raise PreprocessingError("Approved detection dataset changed during preprocessing.")

        summary = _build_summary(
            source,
            output,
            artifacts,
            source_before.digest,
            source_after.digest,
            output_fingerprint.digest,
            output_fingerprint.file_count,
            source_split_summary,
            output_validation,
            intensity_rows,
            records,
            visual_rows,
            reproduction,
            expectations,
            project_root,
        )

        _write_csv(artifacts_build / "preprocessing_manifest.csv", manifest_rows)
        _write_csv(artifacts_build / "intensity_summary.csv", intensity_rows)
        _write_csv(artifacts_build / "visual_review_index.csv", visual_rows)
        _write_json(artifacts_build / "reproducibility.json", reproduction)
        _write_json(artifacts_build / "preprocessing_summary.json", summary)

        _write_csv(canonical_build / "intensity_summary.csv", intensity_rows)
        _write_csv(canonical_build / "visual_review_index.csv", visual_rows)
        _write_json(canonical_build / "preprocessing_summary.json", summary)

        _finalize_directories(
            ((output_build, output), (artifacts_build, artifacts), (canonical_build, canonical_evidence)),
            overwrite=overwrite,
        )
        return summary
    except Exception:
        for build in builds:
            _remove_build_directory(build)
        raise


def _inspect_source_dataset(
    source: Path,
    expectations: DatasetExpectations,
) -> tuple[tuple[DetectionPair, ...], dict[str, dict[str, object]]]:
    data_yaml = source / "data.yaml"
    if not data_yaml.is_file() or _read_dataset_config(data_yaml) != expectations.class_names:
        raise PreprocessingError("Source data.yaml does not contain the approved class mapping.")
    _validate_yaml_paths(data_yaml)

    valid_class_ids = set(range(len(expectations.class_names)))
    pairs: list[DetectionPair] = []
    split_summary: dict[str, dict[str, object]] = {}
    total_classes: Counter[int] = Counter()
    for split in SPLITS:
        image_dir = source / split / "images"
        label_dir = source / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise PreprocessingError(f"Source split directories are missing for {split}.")
        _reject_unexpected_entries(image_dir, IMAGE_EXTENSIONS)
        _reject_unexpected_entries(label_dir, {".txt"})
        matching = match_images_and_labels(image_dir, label_dir)
        _require_complete_matching(matching, split)

        annotations = 0
        empty_labels = 0
        class_counts: Counter[int] = Counter()
        split_pairs: list[DetectionPair] = []
        for image_path, label_path in sorted(
            matching["pairs"], key=lambda pair: (pair[0].name.casefold(), pair[0].name)
        ):
            parsed = parse_yolo_label(label_path, valid_class_ids)
            if parsed.issues:
                raise PreprocessingError(f"Invalid source label: {label_path}")
            for annotation in parsed.annotations:
                if annotation.annotation_type != "box":
                    raise PreprocessingError(f"Expected detection boxes in {label_path}.")
                _validate_box(annotation)
                class_counts[annotation.class_id] += 1
            annotations += len(parsed.annotations)
            empty_labels += int(parsed.is_empty)
            split_pairs.append(DetectionPair(split, image_path, label_path, parsed))

        expected = expectations.splits[split]
        actual_classes = tuple(class_counts[index] for index in range(len(expectations.class_names)))
        if (
            len(split_pairs) != expected.images
            or annotations != expected.annotations
            or empty_labels != expected.empty_labels
            or actual_classes != expected.class_counts
        ):
            raise PreprocessingError(f"Source split {split!r} no longer matches approved counts.")
        pairs.extend(split_pairs)
        total_classes.update(class_counts)
        split_summary[split] = {
            "images": len(split_pairs),
            "labels": len(split_pairs),
            "annotations": annotations,
            "empty_labels": empty_labels,
            "per_class_annotations": {
                str(index): class_counts[index] for index in range(len(expectations.class_names))
            },
        }

    if len(pairs) != expectations.images:
        raise PreprocessingError("Source image total does not match the approved dataset.")
    if tuple(total_classes[index] for index in range(len(expectations.class_names))) != expectations.class_counts:
        raise PreprocessingError("Source class totals do not match the approved dataset.")
    return tuple(pairs), split_summary


def _build_preprocessed_dataset(
    pairs: Sequence[DetectionPair],
    output_build: Path,
) -> list[ProcessedImageRecord]:
    for split in SPLITS:
        (output_build / split / "images").mkdir(parents=True)
        (output_build / split / "labels").mkdir(parents=True)

    records: list[ProcessedImageRecord] = []
    for pair in pairs:
        image, orientation = _decode_without_orientation_transform(pair.image_path)
        source_channels = image.shape[2] if image.ndim == 3 else 1
        grayscale, processed = preprocess_image_array(image)
        height, width = grayscale.shape
        validate_processed_array(processed, width, height)

        processed_image = output_build / pair.split / "images" / f"{pair.image_path.stem}.png"
        processed_label = output_build / pair.split / "labels" / pair.label_path.name
        if not cv2.imwrite(
            str(processed_image),
            processed,
            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
        ):
            raise PreprocessingError(f"OpenCV could not write {processed_image}.")
        shutil.copyfile(pair.label_path, processed_label)

        decoded_output = cv2.imread(str(processed_image), cv2.IMREAD_UNCHANGED)
        if decoded_output is None:
            raise PreprocessingError(f"Processed PNG cannot be decoded: {processed_image}")
        validate_processed_array(decoded_output, width, height)
        if processed_image.read_bytes()[:8] != PNG_SIGNATURE:
            raise PreprocessingError(f"Processed image is not a PNG file: {processed_image}")

        source_label_hash = sha256_file(pair.label_path)
        processed_label_hash = sha256_file(processed_label)
        if source_label_hash != processed_label_hash:
            raise PreprocessingError(f"Label bytes changed for {pair.label_path}.")

        original_float = grayscale.astype(np.float64)
        processed_gray = decoded_output[:, :, 0]
        processed_float = processed_gray.astype(np.float64)
        records.append(
            ProcessedImageRecord(
                split=pair.split,
                source_filename=pair.image_path.name,
                processed_filename=processed_image.name,
                width=width,
                height=height,
                source_suffix=pair.image_path.suffix.lower(),
                source_dtype=str(image.dtype),
                source_channels=source_channels,
                exif_orientation=orientation,
                original_mean=float(original_float.mean()),
                processed_mean=float(processed_float.mean()),
                original_std=float(original_float.std()),
                processed_std=float(processed_float.std()),
                original_min=int(grayscale.min()),
                original_max=int(grayscale.max()),
                processed_min=int(processed_gray.min()),
                processed_max=int(processed_gray.max()),
                pixel_count=int(grayscale.size),
                original_sum=float(original_float.sum()),
                original_square_sum=float(np.square(original_float).sum()),
                processed_sum=float(processed_float.sum()),
                processed_square_sum=float(np.square(processed_float).sum()),
                original_zero_count=int(np.count_nonzero(grayscale == 0)),
                original_full_count=int(np.count_nonzero(grayscale == 255)),
                processed_zero_count=int(np.count_nonzero(processed_gray == 0)),
                processed_full_count=int(np.count_nonzero(processed_gray == 255)),
                source_image_sha256=sha256_file(pair.image_path),
                processed_image_sha256=sha256_file(processed_image),
                source_label_sha256=source_label_hash,
                processed_label_sha256=processed_label_hash,
                annotations=pair.parsed_label.annotations,
            )
        )
    return records


def _validate_output_dataset(
    output: Path,
    records: Sequence[ProcessedImageRecord],
    expectations: DatasetExpectations,
) -> dict[str, object]:
    if _read_dataset_config(output / "data.yaml") != expectations.class_names:
        raise PreprocessingError("Processed data.yaml class mapping changed.")
    _validate_yaml_paths(output / "data.yaml")
    record_by_key = {(record.split, Path(record.processed_filename).stem.casefold()): record for record in records}
    split_summary: dict[str, dict[str, object]] = {}
    total_annotations = 0
    total_empty = 0
    total_classes: Counter[int] = Counter()

    for split in SPLITS:
        image_dir = output / split / "images"
        label_dir = output / split / "labels"
        _reject_unexpected_entries(image_dir, {".png"})
        _reject_unexpected_entries(label_dir, {".txt"})
        matching = match_images_and_labels(image_dir, label_dir)
        _require_complete_matching(matching, f"processed {split}")
        expected = expectations.splits[split]
        if len(matching["pairs"]) != expected.images:
            raise PreprocessingError(f"Processed split {split!r} has an unexpected pair count.")

        annotations = 0
        empty_labels = 0
        class_counts: Counter[int] = Counter()
        for image_path, label_path in matching["pairs"]:
            record = record_by_key.get((split, image_path.stem.casefold()))
            if record is None:
                raise PreprocessingError(f"Processed image is missing from the manifest: {image_path}")
            decoded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if decoded is None:
                raise PreprocessingError(f"Processed image cannot be decoded: {image_path}")
            validate_processed_array(decoded, record.width, record.height)
            if image_path.suffix.lower() != ".png" or image_path.read_bytes()[:8] != PNG_SIGNATURE:
                raise PreprocessingError(f"Processed image is not lossless PNG: {image_path}")
            if sha256_file(image_path) != record.processed_image_sha256:
                raise PreprocessingError(f"Processed image hash changed during validation: {image_path}")
            if sha256_file(label_path) != record.source_label_sha256:
                raise PreprocessingError(f"Processed label bytes differ from source: {label_path}")

            parsed = parse_yolo_label(label_path, set(range(len(expectations.class_names))))
            if parsed.issues or any(annotation.annotation_type != "box" for annotation in parsed.annotations):
                raise PreprocessingError(f"Processed label failed detection validation: {label_path}")
            annotations += len(parsed.annotations)
            empty_labels += int(parsed.is_empty)
            class_counts.update(annotation.class_id for annotation in parsed.annotations)

        actual_classes = tuple(class_counts[index] for index in range(len(expectations.class_names)))
        if (
            annotations != expected.annotations
            or empty_labels != expected.empty_labels
            or actual_classes != expected.class_counts
        ):
            raise PreprocessingError(f"Processed split {split!r} does not preserve approved labels.")
        total_annotations += annotations
        total_empty += empty_labels
        total_classes.update(class_counts)
        split_summary[split] = {
            "images": expected.images,
            "labels": expected.images,
            "annotations": annotations,
            "empty_labels": empty_labels,
            "per_class_annotations": {
                str(index): class_counts[index] for index in range(len(expectations.class_names))
            },
        }

    if len(records) != expectations.images or total_annotations != expectations.annotations:
        raise PreprocessingError("Processed dataset totals do not match approved expectations.")
    if total_empty != expectations.empty_labels:
        raise PreprocessingError("Processed empty-label total changed.")
    if tuple(total_classes[index] for index in range(len(expectations.class_names))) != expectations.class_counts:
        raise PreprocessingError("Processed class totals changed.")
    return {
        "images": len(records),
        "labels": len(records),
        "annotations": total_annotations,
        "empty_labels": total_empty,
        "split_summary": split_summary,
        "dimension_mismatches": 0,
        "label_hash_mismatches": 0,
        "decode_failures": 0,
        "non_png_images": 0,
        "non_three_channel_images": 0,
        "nonidentical_channel_images": 0,
    }


def _verify_reproducibility(
    temporary_parent: Path,
    pairs: Sequence[DetectionPair],
    expectations: DatasetExpectations,
    approved_output_fingerprint: str,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="phase2d-reproduction-", dir=temporary_parent) as temp_name:
        reproduced = Path(temp_name) / "dataset"
        reproduced.mkdir()
        reproduced_records = _build_preprocessed_dataset(pairs, reproduced)
        _write_data_yaml(reproduced / "data.yaml", expectations.class_names)
        _validate_output_dataset(reproduced, reproduced_records, expectations)
        fingerprint = fingerprint_dataset(reproduced)
        success = fingerprint.digest == approved_output_fingerprint
        if not success:
            raise PreprocessingError("Independent CLAHE rebuild produced a different dataset fingerprint.")
    return {
        "success": True,
        "first_build_fingerprint": approved_output_fingerprint,
        "reproduced_fingerprint": fingerprint.digest,
        "fingerprints_identical": True,
        "files_compared": fingerprint.file_count,
        "temporary_rebuild_removed": True,
    }


def _intensity_summary_rows(records: Sequence[ProcessedImageRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, Sequence[ProcessedImageRecord]]] = [("overall", records)]
    groups.extend((split, [record for record in records if record.split == split]) for split in SPLITS)
    for scope, group in groups:
        if not group:
            raise PreprocessingError(f"No records available for intensity scope {scope}.")
        pixel_count = sum(record.pixel_count for record in group)
        original_sum = sum(record.original_sum for record in group)
        processed_sum = sum(record.processed_sum for record in group)
        original_square_sum = sum(record.original_square_sum for record in group)
        processed_square_sum = sum(record.processed_square_sum for record in group)
        original_mean = original_sum / pixel_count
        processed_mean = processed_sum / pixel_count
        original_std = math.sqrt(max(0.0, original_square_sum / pixel_count - original_mean**2))
        processed_std = math.sqrt(max(0.0, processed_square_sum / pixel_count - processed_mean**2))
        rows.append(
            {
                "scope": scope,
                "images": len(group),
                "pixels": pixel_count,
                "original_mean": _round(original_mean),
                "processed_mean": _round(processed_mean),
                "original_std": _round(original_std),
                "processed_std": _round(processed_std),
                "std_change": _round(processed_std - original_std),
                "images_with_increased_std": sum(
                    record.processed_std > record.original_std for record in group
                ),
                "original_min": min(record.original_min for record in group),
                "original_max": max(record.original_max for record in group),
                "processed_min": min(record.processed_min for record in group),
                "processed_max": max(record.processed_max for record in group),
                "original_zero_fraction": _round(
                    sum(record.original_zero_count for record in group) / pixel_count
                ),
                "processed_zero_fraction": _round(
                    sum(record.processed_zero_count for record in group) / pixel_count
                ),
                "original_255_fraction": _round(
                    sum(record.original_full_count for record in group) / pixel_count
                ),
                "processed_255_fraction": _round(
                    sum(record.processed_full_count for record in group) / pixel_count
                ),
                "median_image_original_mean": _round(median(record.original_mean for record in group)),
                "median_image_processed_mean": _round(median(record.processed_mean for record in group)),
                "median_image_original_std": _round(median(record.original_std for record in group)),
                "median_image_processed_std": _round(median(record.processed_std for record in group)),
            }
        )
    return rows


def _manifest_rows(
    records: Sequence[ProcessedImageRecord],
    source: Path,
    output: Path,
    project_root: Path,
) -> list[dict[str, object]]:
    return [
        {
            "split": record.split,
            "source_filename": record.source_filename,
            "processed_filename": record.processed_filename,
            "source_image_path": _display_path(
                source / record.split / "images" / record.source_filename, project_root
            ),
            "processed_image_path": _display_path(
                output / record.split / "images" / record.processed_filename, project_root
            ),
            "width": record.width,
            "height": record.height,
            "source_dtype": record.source_dtype,
            "source_channels": record.source_channels,
            "exif_orientation": record.exif_orientation,
            "original_mean": _round(record.original_mean),
            "processed_mean": _round(record.processed_mean),
            "original_std": _round(record.original_std),
            "processed_std": _round(record.processed_std),
            "original_min": record.original_min,
            "original_max": record.original_max,
            "processed_min": record.processed_min,
            "processed_max": record.processed_max,
            "source_image_sha256": record.source_image_sha256,
            "processed_image_sha256": record.processed_image_sha256,
            "source_label_sha256": record.source_label_sha256,
            "processed_label_sha256": record.processed_label_sha256,
            "label_sha256_equal": record.source_label_sha256 == record.processed_label_sha256,
            "annotation_count": len(record.annotations),
            "class_ids": ";".join(str(value) for value in record.class_ids),
        }
        for record in records
    ]


def _build_summary(
    source: Path,
    output: Path,
    artifacts: Path,
    source_before: str,
    source_after: str,
    output_fingerprint: str,
    output_file_count: int,
    source_split_summary: Mapping[str, object],
    output_validation: Mapping[str, object],
    intensity_rows: Sequence[Mapping[str, object]],
    records: Sequence[ProcessedImageRecord],
    visual_rows: Sequence[Mapping[str, object]],
    reproduction: Mapping[str, object],
    expectations: DatasetExpectations,
    project_root: Path,
) -> dict[str, object]:
    source_encodings = Counter(
        (record.source_suffix, record.source_dtype, record.source_channels, record.exif_orientation)
        for record in records
    )
    return {
        "tool_version": TOOL_VERSION,
        "preprocessing_timestamp_utc": datetime.now(UTC).isoformat(),
        "repository_commit_sha": _repository_commit(project_root),
        "source_dataset_path": _display_path(source, project_root),
        "processed_dataset_path": _display_path(output, project_root),
        "phase2d_output_path": _display_path(artifacts, project_root),
        "source_dataset_fingerprint_before": source_before,
        "source_dataset_fingerprint_after": source_after,
        "source_unchanged": source_before == source_after,
        "processed_dataset_fingerprint": output_fingerprint,
        "processed_dataset_file_count": output_file_count,
        "algorithm": {
            "steps": [
                "decode stored pixels without orientation transform",
                "standardize to 8-bit single-channel grayscale",
                "apply OpenCV CLAHE",
                "replicate processed grayscale into three identical channels",
                "encode losslessly as PNG",
            ],
            "grayscale_conversion": "OpenCV BGR/BGRA to grayscale; existing single-channel input retained",
            "uint8_standardization": "fixed full-dtype-range linear mapping; current source is already uint8",
            "clahe_clip_limit": CLAHE_CLIP_LIMIT,
            "clahe_tile_grid_size": list(CLAHE_TILE_GRID_SIZE),
            "png_compression": PNG_COMPRESSION,
            "opencv_version": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
        },
        "source_encoding_profiles": [
            {
                "suffix": key[0],
                "dtype": key[1],
                "channels": key[2],
                "exif_orientation": key[3],
                "images": count,
            }
            for key, count in sorted(source_encodings.items())
        ],
        "class_mapping": {str(index): name for index, name in enumerate(expectations.class_names)},
        "source_splits": source_split_summary,
        "validation": dict(output_validation),
        "label_integrity": {
            "labels_compared": expectations.images,
            "byte_identical_labels": expectations.images,
            "hash_mismatches": 0,
            "empty_labels_preserved": expectations.empty_labels,
        },
        "geometry_integrity": {
            "images_compared": expectations.images,
            "dimension_mismatches": 0,
            "orientation_changes": 0,
            "crops": 0,
            "resizes": 0,
        },
        "intensity_statistics": {str(row["scope"]): dict(row) for row in intensity_rows},
        "visual_review": {
            "images": len(visual_rows),
            "split_scope": ["train", "valid"],
            "test_images": 0,
            "selection_reasons": sorted(
                {reason for row in visual_rows for reason in str(row["selection_reasons"]).split(";")}
            ),
            "package_generated": True,
        },
        "reproducibility": dict(reproduction),
        "automated_validation_passed": True,
        "official_training_performed": False,
        "technically_ready_for_manual_visual_review": True,
    }


def _validate_box(annotation: Annotation) -> None:
    values = (annotation.x_center, annotation.y_center, annotation.width, annotation.height)
    if not all(math.isfinite(value) for value in values):
        raise PreprocessingError("Detection box contains a non-finite value.")
    left = annotation.x_center - annotation.width / 2
    top = annotation.y_center - annotation.height / 2
    right = annotation.x_center + annotation.width / 2
    bottom = annotation.y_center + annotation.height / 2
    tolerance = 1e-10
    if (
        annotation.width <= 0
        or annotation.height <= 0
        or left < -tolerance
        or top < -tolerance
        or right > 1 + tolerance
        or bottom > 1 + tolerance
    ):
        raise PreprocessingError(f"Invalid source detection box at line {annotation.line_number}.")


def _validate_expected_shape(expectations: DatasetExpectations) -> None:
    if tuple(expectations.splits) != SPLITS:
        raise PreprocessingError(f"Expected split order {SPLITS!r}.")
    class_count = len(expectations.class_names)
    if class_count == 0 or any(len(split.class_counts) != class_count for split in expectations.splits.values()):
        raise PreprocessingError("Dataset expectations contain inconsistent class counts.")


def _validate_locations(source: Path, output: Path, artifacts: Path, canonical: Path) -> None:
    if not source.is_dir():
        raise PreprocessingError(f"Approved source dataset does not exist: {source}")
    for name, path in (("processed dataset", output), ("artifacts", artifacts), ("canonical evidence", canonical)):
        if path == source or path.is_relative_to(source):
            raise PreprocessingError(f"{name.capitalize()} must not be inside the approved source dataset: {path}")
    paths = (output, artifacts, canonical)
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            if first == second or first.is_relative_to(second) or second.is_relative_to(first):
                raise PreprocessingError("Processed data, generated artifacts, and canonical evidence must be separate.")


def _validate_yaml_paths(path: Path) -> None:
    expected = {"train": "train/images", "val": "valid/images", "test": "test/images"}
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        for key in expected:
            if stripped.startswith(f"{key}:"):
                found[key] = stripped.split(":", 1)[1].strip().strip("'\"")
    if found != expected:
        raise PreprocessingError(f"Unexpected data.yaml split paths: {found!r}")


def _require_complete_matching(matching: Mapping[str, object], context: str) -> None:
    problem_keys = (
        "images_without_labels",
        "labels_without_images",
        "ambiguous_image_stems",
        "ambiguous_label_stems",
    )
    if any(matching[key] for key in problem_keys):
        raise PreprocessingError(f"Unmatched or ambiguous image-label pairs in {context}.")


def _reject_unexpected_entries(directory: Path, allowed_extensions: set[str]) -> None:
    unexpected = [
        path.name
        for path in directory.iterdir()
        if not path.is_file() or path.suffix.lower() not in allowed_extensions
    ]
    if unexpected:
        raise PreprocessingError(f"Unexpected entries in {directory}: {sorted(unexpected, key=str.casefold)!r}")


def _write_data_yaml(path: Path, class_names: Sequence[str]) -> None:
    lines = ["path: .", "train: train/images", "val: valid/images", "test: test/images", "names:"]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise PreprocessingError(f"Required CSV would be empty: {path}")
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _round(value: float) -> float:
    return round(value, 12)


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _repository_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _prepare_build_locations(builds: Iterable[Path], finals: Iterable[Path]) -> None:
    for final in finals:
        final.parent.mkdir(parents=True, exist_ok=True)
    for build in builds:
        _remove_build_directory(build)


def _remove_build_directory(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir() or not path.name.startswith(".") or not path.name.endswith("_building"):
        raise PreprocessingError(f"Refusing to remove unsafe build path: {path}")
    shutil.rmtree(path)


# Replaces the dataset and its evidence together so overwrite failures can be rolled back.
def _finalize_directories(staged_and_final: Sequence[tuple[Path, Path]], *, overwrite: bool) -> None:
    backups: list[tuple[Path, Path]] = []
    finalized: list[Path] = []
    try:
        for _, final in staged_and_final:
            if not final.exists():
                continue
            if not overwrite:
                raise PreprocessingError(f"Output appeared during preprocessing: {final}")
            backup = final.parent / f".{final.name}_backup"
            if backup.exists():
                raise PreprocessingError(f"Stale backup blocks safe replacement: {backup}")
            final.rename(backup)
            backups.append((backup, final))
        for staged, final in staged_and_final:
            staged.rename(final)
            finalized.append(final)
    except Exception:
        for final in reversed(finalized):
            if final.is_dir() and not final.is_symlink():
                shutil.rmtree(final)
        for backup, final in reversed(backups):
            if backup.exists():
                backup.rename(final)
        raise
    else:
        for backup, _ in backups:
            shutil.rmtree(backup)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate the deterministic grayscale plus CLAHE detection dataset."
    )
    parser.add_argument("--source", type=Path, default=Path("data/prepared/v3_detection"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared/v3_detection_clahe"))
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/phase2/phase2d"))
    parser.add_argument("--canonical-evidence", type=Path, default=Path("docs/evidence/phase2d"))
    parser.add_argument("--overwrite", action="store_true", help="Safely rebuild and replace existing outputs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = preprocess_dataset(
            arguments.source,
            arguments.output,
            arguments.artifacts,
            arguments.canonical_evidence,
            overwrite=arguments.overwrite,
        )
    except PreprocessingError as error:
        print(f"Phase 2D preprocessing failed: {error}")
        return 1
    print("Phase 2D preprocessing completed without training.")
    print(f"Processed dataset: {summary['processed_dataset_path']}")
    print(f"Dataset fingerprint: {summary['processed_dataset_fingerprint']}")
    print(f"Visual QA images: {summary['visual_review']['images']} (train/valid only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
