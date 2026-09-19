from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from bone_fracture_audit.audit import _read_dataset_config, match_images_and_labels, sha256_file
from bone_fracture_audit.yolo import Annotation, parse_yolo_label
from bone_fracture_pipeline.prepare_dataset import (
    CANONICAL_EXPECTATIONS,
    CLASS_NAMES,
    SPLITS,
    DatasetExpectations,
    fingerprint_dataset,
    prepare_dataset,
)


TOOL_VERSION = "1.0.0"
SERIALIZATION_TOLERANCE = 1e-10
REVIEW_RANK_COUNT = 10
VISUAL_CATEGORY_LIMIT = 4


class ValidationError(RuntimeError):
    """Raised when the prepared dataset fails an independent Phase 2B check."""


@dataclass(frozen=True)
class ValidatedAnnotation:
    split: str
    filename: str
    source_line_number: int
    prepared_line_number: int
    class_id: int
    polygon_points: tuple[tuple[float, float], ...]
    expected_box: tuple[float, float, float, float]
    prepared_box: tuple[float, float, float, float]
    polygon_area: float
    bounding_box_area: float
    occupancy_ratio: float
    border_distance: float
    image_width: int
    image_height: int
    image_annotation_count: int

    @property
    def key(self) -> tuple[str, str, int]:
        return self.split, self.filename, self.source_line_number

    @property
    def image_key(self) -> tuple[str, str]:
        return self.split, self.filename

    @property
    def box_area(self) -> float:
        return self.prepared_box[2] * self.prepared_box[3]

    @property
    def aspect_extremeness(self) -> float:
        return abs(math.log(self.image_width / self.image_height))


@dataclass(frozen=True)
class ReviewCandidate:
    annotation: ValidatedAnnotation
    reasons: tuple[str, ...]


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        raise ValidationError("A polygon needs at least three vertices.")
    doubled_area = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, (*points[1:], points[0]))
    )
    return abs(doubled_area) / 2


def minimum_axis_aligned_box(points: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    if len(points) < 3:
        raise ValidationError("A polygon needs at least three vertices.")
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    left, right = min(x_values), max(x_values)
    top, bottom = min(y_values), max(y_values)
    return (left + right) / 2, (top + bottom) / 2, right - left, bottom - top


def vertices_are_contained(
    points: Sequence[tuple[float, float]],
    box: Sequence[float],
    *,
    tolerance: float = SERIALIZATION_TOLERANCE,
) -> bool:
    left, top, right, bottom = _box_edges(box)
    return all(
        left - tolerance <= x <= right + tolerance and top - tolerance <= y <= bottom + tolerance
        for x, y in points
    )


def minimum_box_matches(
    points: Sequence[tuple[float, float]],
    box: Sequence[float],
    *,
    tolerance: float = SERIALIZATION_TOLERANCE,
) -> bool:
    expected = minimum_axis_aligned_box(points)
    return all(abs(expected_value - actual_value) <= tolerance for expected_value, actual_value in zip(expected, box))


def occupancy_ratio(points: Sequence[tuple[float, float]], box: Sequence[float]) -> float:
    area = polygon_area(points)
    box_area = box[2] * box[3]
    if area <= 0 or box_area <= 0:
        raise ValidationError("Polygon and bounding-box areas must be positive.")
    ratio = area / box_area
    if ratio > 1 + SERIALIZATION_TOLERANCE:
        raise ValidationError(f"Polygon occupancy cannot exceed one, found {ratio:.12f}.")
    return min(ratio, 1.0)


def validate_annotation_geometry(source: Annotation, prepared: Annotation) -> dict[str, object]:
    if source.annotation_type != "polygon":
        raise ValidationError(f"Source line {source.line_number} is not a polygon.")
    if prepared.annotation_type != "box":
        raise ValidationError(f"Prepared line {prepared.line_number} is not a detection box.")
    if source.class_id != prepared.class_id:
        raise ValidationError(
            f"Class mismatch at source line {source.line_number}: {source.class_id} != {prepared.class_id}."
        )

    box = (prepared.x_center, prepared.y_center, prepared.width, prepared.height)
    _validate_normalized_box(box)
    contained = vertices_are_contained(source.polygon_points, box)
    minimum_match = minimum_box_matches(source.polygon_points, box)
    if not contained:
        raise ValidationError(f"Prepared box does not contain every vertex at source line {source.line_number}.")
    if not minimum_match:
        raise ValidationError(f"Prepared box is not the polygon's minimum bounds at source line {source.line_number}.")

    expected = minimum_axis_aligned_box(source.polygon_points)
    area = polygon_area(source.polygon_points)
    ratio = occupancy_ratio(source.polygon_points, box)
    return {
        "expected_box": expected,
        "prepared_box": box,
        "polygon_area": area,
        "bounding_box_area": box[2] * box[3],
        "occupancy_ratio": ratio,
        "vertices_contained": contained,
        "minimum_box_match": minimum_match,
        "valid_normalized_box": True,
    }


def select_review_candidates(
    annotations: Sequence[ValidatedAnnotation],
    *,
    rank_count: int = REVIEW_RANK_COUNT,
) -> list[ReviewCandidate]:
    eligible = [annotation for annotation in annotations if annotation.split in {"train", "valid"}]
    if not eligible:
        return []

    reasons: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    stable_key = lambda item: (item.split, item.filename.casefold(), item.filename, item.source_line_number)

    ranked_groups = {
        "lowest_occupancy": sorted(eligible, key=lambda item: (item.occupancy_ratio, stable_key(item))),
        "highest_occupancy": sorted(eligible, key=lambda item: (-item.occupancy_ratio, stable_key(item))),
        "smallest_box": sorted(eligible, key=lambda item: (item.box_area, stable_key(item))),
        "largest_box": sorted(eligible, key=lambda item: (-item.box_area, stable_key(item))),
        "closest_to_boundary": sorted(eligible, key=lambda item: (item.border_distance, stable_key(item))),
        "unusual_image_aspect_ratio": sorted(
            eligible, key=lambda item: (-item.aspect_extremeness, stable_key(item))
        ),
    }
    for reason, ranked in ranked_groups.items():
        for annotation in ranked[:rank_count]:
            reasons[annotation.key].add(reason)

    multi_image_keys = sorted(
        {item.image_key for item in eligible if item.image_annotation_count > 1},
        key=lambda key: (key[0], key[1].casefold(), key[1]),
    )[:rank_count]
    for annotation in eligible:
        if annotation.image_key in multi_image_keys:
            reasons[annotation.key].add("multi_annotation_image")

    # Each class receives a representative from both allowed review splits.
    for split in ("train", "valid"):
        for class_id in sorted({annotation.class_id for annotation in eligible if annotation.split == split}):
            class_annotations = [
                annotation
                for annotation in eligible
                if annotation.split == split and annotation.class_id == class_id
            ]
            class_median = median(annotation.occupancy_ratio for annotation in class_annotations)
            representative = min(
                class_annotations,
                key=lambda item: (abs(item.occupancy_ratio - class_median), stable_key(item)),
            )
            reasons[representative.key].add("representative_class")

    by_key = {annotation.key: annotation for annotation in eligible}
    return [
        ReviewCandidate(by_key[key], tuple(sorted(reasons[key])))
        for key in sorted(reasons, key=lambda item: (item[0], item[1].casefold(), item[1], item[2]))
    ]


def select_visual_review_images(
    candidates: Sequence[ReviewCandidate],
    *,
    category_limit: int = VISUAL_CATEGORY_LIMIT,
) -> list[tuple[tuple[str, str], tuple[str, ...]]]:
    selected_reasons: dict[tuple[str, str], set[str]] = defaultdict(set)

    # Class representatives are mandatory before ranked diagnostic categories are added.
    representatives = [candidate for candidate in candidates if "representative_class" in candidate.reasons]
    for candidate in sorted(representatives, key=lambda item: (item.annotation.class_id, item.annotation.key)):
        selected_reasons[candidate.annotation.image_key].add(
            f"representative_{candidate.annotation.split}_class_{candidate.annotation.class_id}"
        )

    categories = (
        "lowest_occupancy",
        "smallest_box",
        "largest_box",
        "closest_to_boundary",
        "multi_annotation_image",
        "unusual_image_aspect_ratio",
    )
    category_sort_keys = {
        "lowest_occupancy": lambda item: (item.annotation.occupancy_ratio, item.annotation.key),
        "smallest_box": lambda item: (item.annotation.box_area, item.annotation.key),
        "largest_box": lambda item: (-item.annotation.box_area, item.annotation.key),
        "closest_to_boundary": lambda item: (item.annotation.border_distance, item.annotation.key),
        "multi_annotation_image": lambda item: (-item.annotation.image_annotation_count, item.annotation.key),
        "unusual_image_aspect_ratio": lambda item: (-item.annotation.aspect_extremeness, item.annotation.key),
    }
    for reason in categories:
        reason_candidates = sorted(
            (candidate for candidate in candidates if reason in candidate.reasons),
            key=category_sort_keys[reason],
        )
        unique_images: list[tuple[str, str]] = []
        for candidate in reason_candidates:
            if candidate.annotation.image_key not in unique_images:
                unique_images.append(candidate.annotation.image_key)
            if len(unique_images) == category_limit:
                break
        for image_key in unique_images:
            selected_reasons[image_key].add(reason)

    return [
        (image_key, tuple(sorted(selected_reasons[image_key])))
        for image_key in sorted(selected_reasons, key=lambda key: (key[0], key[1].casefold(), key[1]))
    ]


def compare_reproduced_datasets(
    approved: Path,
    reproduced: Path,
    *,
    class_count: int,
) -> dict[str, object]:
    approved = approved.resolve()
    reproduced = reproduced.resolve()
    label_hash_mismatches: list[str] = []
    image_hash_mismatches: list[str] = []
    missing_or_extra_files: list[str] = []
    empty_label_mismatches: list[str] = []
    approved_class_counts: Counter[int] = Counter()
    reproduced_class_counts: Counter[int] = Counter()
    approved_annotations = 0
    reproduced_annotations = 0
    label_count = 0
    image_count = 0

    for split in SPLITS:
        for kind, extension_set in (("images", None), ("labels", {".txt"})):
            approved_dir = approved / split / kind
            reproduced_dir = reproduced / split / kind
            approved_files = {
                path.name: path
                for path in approved_dir.iterdir()
                if path.is_file() and (extension_set is None or path.suffix.lower() in extension_set)
            }
            reproduced_files = {
                path.name: path
                for path in reproduced_dir.iterdir()
                if path.is_file() and (extension_set is None or path.suffix.lower() in extension_set)
            }
            if set(approved_files) != set(reproduced_files):
                missing_or_extra_files.append(f"{split}/{kind}")
                continue
            for name in sorted(approved_files, key=str.casefold):
                approved_path = approved_files[name]
                reproduced_path = reproduced_files[name]
                if sha256_file(approved_path) != sha256_file(reproduced_path):
                    target = label_hash_mismatches if kind == "labels" else image_hash_mismatches
                    target.append(f"{split}/{kind}/{name}")
                if kind == "images":
                    image_count += 1
                    continue
                label_count += 1
                if (approved_path.stat().st_size == 0) != (reproduced_path.stat().st_size == 0):
                    empty_label_mismatches.append(f"{split}/labels/{name}")
                approved_label = parse_yolo_label(approved_path, set(range(class_count)))
                reproduced_label = parse_yolo_label(reproduced_path, set(range(class_count)))
                if approved_label.issues or reproduced_label.issues:
                    raise ValidationError(f"Invalid label encountered during reproduction comparison: {name}")
                approved_annotations += len(approved_label.annotations)
                reproduced_annotations += len(reproduced_label.annotations)
                approved_class_counts.update(item.class_id for item in approved_label.annotations)
                reproduced_class_counts.update(item.class_id for item in reproduced_label.annotations)

    approved_fingerprint = fingerprint_dataset(approved).digest
    reproduced_fingerprint = fingerprint_dataset(reproduced).digest
    result = {
        "success": not any(
            (label_hash_mismatches, image_hash_mismatches, missing_or_extra_files, empty_label_mismatches)
        )
        and approved_fingerprint == reproduced_fingerprint
        and approved_annotations == reproduced_annotations
        and approved_class_counts == reproduced_class_counts,
        "approved_prepared_fingerprint": approved_fingerprint,
        "reproduced_prepared_fingerprint": reproduced_fingerprint,
        "fingerprints_identical": approved_fingerprint == reproduced_fingerprint,
        "label_files_compared": label_count,
        "image_files_compared": image_count,
        "label_hash_mismatch_count": len(label_hash_mismatches),
        "image_hash_mismatch_count": len(image_hash_mismatches),
        "missing_or_extra_directory_count": len(missing_or_extra_files),
        "empty_label_mismatch_count": len(empty_label_mismatches),
        "approved_annotation_count": approved_annotations,
        "reproduced_annotation_count": reproduced_annotations,
        "class_counts_identical": approved_class_counts == reproduced_class_counts,
        "label_hash_mismatches": label_hash_mismatches,
        "image_hash_mismatches": image_hash_mismatches,
        "missing_or_extra_directories": missing_or_extra_files,
        "empty_label_mismatches": empty_label_mismatches,
    }
    return result


def run_validation(
    source: Path,
    prepared: Path,
    phase2a_artifacts: Path,
    output: Path,
    *,
    overwrite: bool = False,
    expectations: DatasetExpectations = CANONICAL_EXPECTATIONS,
    project_root: Path | None = None,
) -> dict[str, object]:
    source = source.resolve()
    prepared = prepared.resolve()
    phase2a_artifacts = phase2a_artifacts.resolve()
    output = output.resolve()
    project_root = (project_root or Path.cwd()).resolve()
    _validate_locations(source, prepared, output)
    if output.exists() and not overwrite:
        raise ValidationError(f"Phase 2B output already exists: {output}. Use --overwrite to rebuild it.")

    phase2a_summary_path = phase2a_artifacts / "preparation_summary.json"
    if not phase2a_summary_path.is_file():
        raise ValidationError(f"Phase 2A summary is missing: {phase2a_summary_path}")
    phase2a_summary = json.loads(phase2a_summary_path.read_text(encoding="utf-8"))

    source_before = fingerprint_dataset(source)
    prepared_before = fingerprint_dataset(prepared)
    records, independent_summary, geometry_rows = _validate_dataset(source, prepared, expectations)
    occupancy_rows = _occupancy_statistics(records, expectations.class_names)
    candidates = select_review_candidates(records)
    candidate_rows = _candidate_rows(candidates, expectations.class_names)
    visual_images = select_visual_review_images(candidates)

    build = output.parent / f".{output.name}_building"
    _remove_build_directory(build)
    build.mkdir(parents=True)
    try:
        visual_rows = _render_visual_review(
            source,
            build / "visual_review",
            output / "visual_review",
            records,
            visual_images,
            expectations.class_names,
            project_root,
        )
        _write_csv(build / "geometry_validation.csv", geometry_rows)
        _write_csv(build / "occupancy_statistics.csv", occupancy_rows)
        _write_csv(build / "review_candidates.csv", candidate_rows)
        _write_csv(build / "visual_review_index.csv", visual_rows)

        with tempfile.TemporaryDirectory(prefix="phase2b-reproduction-", dir=output.parent) as temp_name:
            reproduction_root = Path(temp_name)
            reproduction_prepared = reproduction_root / "prepared"
            reproduction_artifacts = reproduction_root / "phase2a_artifacts"
            reproduced_summary = prepare_dataset(
                source,
                reproduction_prepared,
                reproduction_artifacts,
                expectations=expectations,
                project_root=project_root,
            )
            reproduction = compare_reproduced_datasets(
                prepared, reproduction_prepared, class_count=len(expectations.class_names)
            )
            reproduction["source_fingerprint_matches_current"] = (
                reproduced_summary["source_dataset_fingerprint"] == source_before.digest
            )
            reproduction["source_fingerprint_matches_phase2a_summary"] = (
                phase2a_summary.get("source_dataset_fingerprint") == source_before.digest
            )
            reproduction["temporary_outputs_cleaned_after_comparison"] = True
            if not reproduction["success"] or not reproduction["source_fingerprint_matches_current"]:
                raise ValidationError("The independent Phase 2A reproduction differs from the approved dataset.")

        source_after = fingerprint_dataset(source)
        prepared_after = fingerprint_dataset(prepared)
        if source_after.digest != source_before.digest:
            raise ValidationError("Raw source data changed during Phase 2B validation.")
        if prepared_after.digest != prepared_before.digest:
            raise ValidationError("The approved prepared dataset changed during Phase 2B validation.")

        summary = {
            "tool_version": TOOL_VERSION,
            "validation_timestamp_utc": datetime.now(UTC).isoformat(),
            "source_dataset_path": _display_path(source, project_root),
            "prepared_dataset_path": _display_path(prepared, project_root),
            "phase2b_output_path": _display_path(output, project_root),
            "source_dataset_fingerprint": source_before.digest,
            "prepared_dataset_fingerprint": prepared_before.digest,
            "source_unchanged_during_validation": True,
            "prepared_unchanged_during_validation": True,
            "serialization_tolerance": SERIALIZATION_TOLERANCE,
            "independent_consistency": independent_summary,
            "geometry": {
                "annotations_checked": len(records),
                "containment_errors": 0,
                "minimum_box_errors": 0,
                "invalid_box_errors": 0,
            },
            "occupancy": _summary_occupancy(records),
            "review_candidates": len(candidates),
            "visual_review_images": len(visual_rows),
            "visual_review_split_scope": ["train", "valid"],
            "review_selection_policy": {
                "ranked_annotations_per_category": REVIEW_RANK_COUNT,
                "visual_images_per_diagnostic_category": VISUAL_CATEGORY_LIMIT,
                "mandatory_class_representatives_per_split": len(expectations.class_names),
                "mandatory_class_representatives_total": 2 * len(expectations.class_names),
                "categories": [
                    "lowest occupancy",
                    "smallest box",
                    "largest box",
                    "closest to boundary",
                    "multi-annotation image",
                    "unusual image aspect ratio",
                ],
            },
            "reproducibility": reproduction,
            "technically_ready_for_phase2c": True,
            "remaining_limitations": [
                "Visual overlays validate conversion alignment, not medical correctness or clinical utility.",
                "Patient or study-level independence remains unverified because source identifiers are unavailable.",
            ],
        }
        (build / "reproducibility_comparison.json").write_text(
            json.dumps(reproduction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (build / "validation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _finalize_output(build, output, overwrite=overwrite)
        return summary
    except Exception:
        _remove_build_directory(build)
        raise


def _validate_dataset(
    source: Path,
    prepared: Path,
    expectations: DatasetExpectations,
) -> tuple[list[ValidatedAnnotation], dict[str, object], list[dict[str, object]]]:
    if _read_dataset_config(source / "data.yaml") != expectations.class_names:
        raise ValidationError("Raw source class mapping differs from the approved mapping.")
    if _read_dataset_config(prepared / "data.yaml") != expectations.class_names:
        raise ValidationError("Prepared class mapping differs from the approved mapping.")
    _validate_prepared_yaml_paths(prepared / "data.yaml")

    records: list[ValidatedAnnotation] = []
    geometry_rows: list[dict[str, object]] = []
    total_class_counts: Counter[int] = Counter()
    split_summaries: dict[str, dict[str, object]] = {}

    for split in SPLITS:
        source_matching = match_images_and_labels(source / split / "images", source / split / "labels")
        prepared_matching = match_images_and_labels(prepared / split / "images", prepared / split / "labels")
        _require_complete_matching(source_matching, f"raw {split}")
        _require_complete_matching(prepared_matching, f"prepared {split}")

        source_images = {path.name: path for path in source_matching["images"]}
        prepared_images = {path.name: path for path in prepared_matching["images"]}
        source_labels = {path.name: path for path in source_matching["labels"]}
        prepared_labels = {path.name: path for path in prepared_matching["labels"]}
        if set(source_images) != set(prepared_images) or set(source_labels) != set(prepared_labels):
            raise ValidationError(f"Raw and prepared filenames differ in split {split!r}.")

        split_annotations = 0
        split_empty = 0
        split_class_counts: Counter[int] = Counter()
        image_hash_mismatches = 0
        for image_name in sorted(source_images, key=str.casefold):
            source_image = source_images[image_name]
            prepared_image = prepared_images[image_name]
            if sha256_file(source_image) != sha256_file(prepared_image):
                image_hash_mismatches += 1
                raise ValidationError(f"Source/prepared image hash mismatch: {split}/{image_name}")

            label_name = f"{Path(image_name).stem}.txt"
            if label_name not in source_labels or label_name not in prepared_labels:
                raise ValidationError(f"Expected label is missing for {split}/{image_name}.")
            source_label = parse_yolo_label(source_labels[label_name], set(range(len(expectations.class_names))))
            prepared_label = parse_yolo_label(prepared_labels[label_name], set(range(len(expectations.class_names))))
            if source_label.issues or prepared_label.issues:
                raise ValidationError(f"A malformed label was found for {split}/{image_name}.")
            if len(source_label.annotations) != len(prepared_label.annotations):
                raise ValidationError(f"Annotation count mismatch for {split}/{image_name}.")
            if source_label.is_empty != prepared_label.is_empty:
                raise ValidationError(f"Empty-label status mismatch for {split}/{image_name}.")
            if source_label.is_empty and prepared_labels[label_name].stat().st_size != 0:
                raise ValidationError(f"Prepared empty label is not zero-byte: {split}/{label_name}.")

            split_empty += int(source_label.is_empty)
            split_annotations += len(source_label.annotations)
            with Image.open(source_image) as image:
                image_width, image_height = image.size

            for source_annotation, prepared_annotation in zip(
                source_label.annotations, prepared_label.annotations
            ):
                geometry = validate_annotation_geometry(source_annotation, prepared_annotation)
                prepared_box = geometry["prepared_box"]
                left, top, right, bottom = _box_edges(prepared_box)
                record = ValidatedAnnotation(
                    split=split,
                    filename=image_name,
                    source_line_number=source_annotation.line_number,
                    prepared_line_number=prepared_annotation.line_number,
                    class_id=source_annotation.class_id,
                    polygon_points=source_annotation.polygon_points,
                    expected_box=geometry["expected_box"],
                    prepared_box=prepared_box,
                    polygon_area=geometry["polygon_area"],
                    bounding_box_area=geometry["bounding_box_area"],
                    occupancy_ratio=geometry["occupancy_ratio"],
                    border_distance=min(left, top, 1 - right, 1 - bottom),
                    image_width=image_width,
                    image_height=image_height,
                    image_annotation_count=len(source_label.annotations),
                )
                records.append(record)
                split_class_counts[record.class_id] += 1
                geometry_rows.append(_geometry_row(record))

        expected = expectations.splits[split]
        actual_classes = tuple(split_class_counts[index] for index in range(len(expectations.class_names)))
        if (
            len(source_images) != expected.images
            or len(prepared_images) != expected.images
            or split_annotations != expected.annotations
            or split_empty != expected.empty_labels
            or actual_classes != expected.class_counts
        ):
            raise ValidationError(f"Independent count validation failed for split {split!r}.")
        total_class_counts.update(split_class_counts)
        split_summaries[split] = {
            "images": len(source_images),
            "labels": len(source_labels),
            "annotations": split_annotations,
            "empty_labels": split_empty,
            "image_hash_mismatches": image_hash_mismatches,
            "per_class_annotations": {
                str(index): split_class_counts[index] for index in range(len(expectations.class_names))
            },
        }

    total_counts = tuple(total_class_counts[index] for index in range(len(expectations.class_names)))
    if len(records) != expectations.annotations or total_counts != expectations.class_counts:
        raise ValidationError("Independent dataset totals differ from the approved expectations.")
    summary = {
        "images": expectations.images,
        "labels": expectations.images,
        "annotations": len(records),
        "empty_labels": expectations.empty_labels,
        "class_mapping": {str(index): name for index, name in enumerate(expectations.class_names)},
        "per_class_annotations": {
            str(index): total_class_counts[index] for index in range(len(expectations.class_names))
        },
        "splits": split_summaries,
        "source_prepared_image_hash_mismatches": 0,
    }
    return records, summary, geometry_rows


def _geometry_row(record: ValidatedAnnotation) -> dict[str, object]:
    expected = record.expected_box
    prepared = record.prepared_box
    return {
        "split": record.split,
        "filename": record.filename,
        "source_line_number": record.source_line_number,
        "prepared_line_number": record.prepared_line_number,
        "class_id": record.class_id,
        "vertex_count": len(record.polygon_points),
        "expected_x_center": _format(expected[0]),
        "expected_y_center": _format(expected[1]),
        "expected_width": _format(expected[2]),
        "expected_height": _format(expected[3]),
        "prepared_x_center": _format(prepared[0]),
        "prepared_y_center": _format(prepared[1]),
        "prepared_width": _format(prepared[2]),
        "prepared_height": _format(prepared[3]),
        "vertices_contained": True,
        "minimum_box_match": True,
        "valid_normalized_box": True,
        "polygon_area": _format(record.polygon_area),
        "bounding_box_area": _format(record.bounding_box_area),
        "occupancy_ratio": _format(record.occupancy_ratio),
        "border_distance": _format(record.border_distance),
    }


def _occupancy_statistics(
    records: Sequence[ValidatedAnnotation], class_names: Sequence[str]
) -> list[dict[str, object]]:
    groups: list[tuple[str, str, Sequence[ValidatedAnnotation]]] = [("overall", "all", records)]
    groups.extend(("split", split, [item for item in records if item.split == split]) for split in SPLITS)
    groups.extend(
        (
            "class",
            f"{class_id}: {name}",
            [item for item in records if item.class_id == class_id],
        )
        for class_id, name in enumerate(class_names)
    )
    rows = []
    for scope_type, scope_value, group in groups:
        values = sorted(item.occupancy_ratio for item in group)
        if not values:
            continue
        rows.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "annotation_count": len(values),
                "minimum": _format(min(values)),
                "q1": _format(_percentile(values, 0.25)),
                "median": _format(median(values)),
                "mean": _format(mean(values)),
                "q3": _format(_percentile(values, 0.75)),
                "maximum": _format(max(values)),
            }
        )
    return rows


def _summary_occupancy(records: Sequence[ValidatedAnnotation]) -> dict[str, object]:
    values = sorted(record.occupancy_ratio for record in records)
    return {
        "minimum": float(_format(min(values))),
        "q1": float(_format(_percentile(values, 0.25))),
        "median": float(_format(median(values))),
        "mean": float(_format(mean(values))),
        "q3": float(_format(_percentile(values, 0.75))),
        "maximum": float(_format(max(values))),
    }


def _candidate_rows(
    candidates: Sequence[ReviewCandidate], class_names: Sequence[str]
) -> list[dict[str, object]]:
    return [
        {
            "split": candidate.annotation.split,
            "filename": candidate.annotation.filename,
            "source_line_number": candidate.annotation.source_line_number,
            "class_id": candidate.annotation.class_id,
            "class_name": class_names[candidate.annotation.class_id],
            "selection_reasons": ";".join(candidate.reasons),
            "occupancy_ratio": _format(candidate.annotation.occupancy_ratio),
            "bounding_box_area": _format(candidate.annotation.box_area),
            "border_distance": _format(candidate.annotation.border_distance),
            "image_annotation_count": candidate.annotation.image_annotation_count,
            "image_width": candidate.annotation.image_width,
            "image_height": candidate.annotation.image_height,
        }
        for candidate in candidates
    ]


def _render_visual_review(
    source: Path,
    build_output: Path,
    final_output: Path,
    records: Sequence[ValidatedAnnotation],
    selected_images: Sequence[tuple[tuple[str, str], tuple[str, ...]]],
    class_names: Sequence[str],
    project_root: Path,
) -> list[dict[str, object]]:
    build_output.mkdir(parents=True)
    by_image: dict[tuple[str, str], list[ValidatedAnnotation]] = defaultdict(list)
    for record in records:
        by_image[record.image_key].append(record)

    rows: list[dict[str, object]] = []
    rendered_paths: list[Path] = []
    for index, (image_key, reasons) in enumerate(selected_images, start=1):
        split, filename = image_key
        annotations = sorted(by_image[image_key], key=lambda item: item.source_line_number)
        source_image = source / split / "images" / filename
        rendered_name = f"{index:02d}_{split}_{Path(filename).stem}.png"
        rendered_path = build_output / rendered_name
        _render_review_image(source_image, rendered_path, annotations, class_names, reasons)
        rendered_paths.append(rendered_path)
        class_ids = sorted({annotation.class_id for annotation in annotations})
        rows.append(
            {
                "review_order": index,
                "split": split,
                "filename": filename,
                "review_image_path": _display_path(final_output / rendered_name, project_root),
                "selection_reasons": ";".join(reasons),
                "class_ids": ";".join(str(value) for value in class_ids),
                "class_names": ";".join(class_names[value] for value in class_ids),
                "annotation_count": len(annotations),
            }
        )
    _render_contact_sheet(rendered_paths, build_output / "contact_sheet.jpg")
    return rows


def _render_review_image(
    source_image: Path,
    output: Path,
    annotations: Sequence[ValidatedAnnotation],
    class_names: Sequence[str],
    reasons: Sequence[str],
) -> None:
    with Image.open(source_image) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    header_height = max(54, min(90, height // 8))
    canvas = Image.new("RGB", (width, height + header_height), "black")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    line_width = max(2, round(min(width, height) / 250))
    draw.text((8, 6), source_image.name, fill="white", font=font)
    draw.text((8, 22), f"split: {annotations[0].split}", fill="white", font=font)
    draw.text((8, 38), "reasons: " + ", ".join(reasons), fill="white", font=font)

    for annotation in annotations:
        polygon = [(x * width, y * height + header_height) for x, y in annotation.polygon_points]
        draw.line((*polygon, polygon[0]), fill=(0, 255, 255), width=line_width, joint="curve")
        for x, y in polygon:
            radius = max(2, line_width)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 0))
        left, top, right, bottom = _box_edges(annotation.prepared_box)
        rectangle = (left * width, top * height + header_height, right * width, bottom * height + header_height)
        draw.rectangle(rectangle, outline=(255, 0, 255), width=line_width)
        label = f"{annotation.class_id}: {class_names[annotation.class_id]}"
        label_y = max(header_height, rectangle[1] - 14)
        draw.rectangle((rectangle[0], label_y, rectangle[0] + len(label) * 7 + 6, label_y + 14), fill="black")
        draw.text((rectangle[0] + 3, label_y + 1), label, fill=(255, 255, 255), font=font)
    canvas.save(output, format="PNG", optimize=True)


def _render_contact_sheet(paths: Sequence[Path], output: Path) -> None:
    if not paths:
        raise ValidationError("The visual review selection produced no images.")
    thumb_width, thumb_height = 320, 240
    columns = 4
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), (24, 24, 24))
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((thumb_width - 8, thumb_height - 8))
        x = (index % columns) * thumb_width + (thumb_width - thumbnail.width) // 2
        y = (index // columns) * thumb_height + (thumb_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (x, y))
    sheet.save(output, format="JPEG", quality=90)


def _validate_normalized_box(box: Sequence[float]) -> None:
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValidationError("Prepared box contains non-finite or missing values.")
    x_center, y_center, width, height = box
    left, top, right, bottom = _box_edges(box)
    if not 0 <= x_center <= 1 or not 0 <= y_center <= 1 or not 0 < width <= 1 or not 0 < height <= 1:
        raise ValidationError(f"Prepared box has invalid normalized values: {tuple(box)!r}")
    if left < -SERIALIZATION_TOLERANCE or top < -SERIALIZATION_TOLERANCE:
        raise ValidationError(f"Prepared box begins outside the image: {tuple(box)!r}")
    if right > 1 + SERIALIZATION_TOLERANCE or bottom > 1 + SERIALIZATION_TOLERANCE:
        raise ValidationError(f"Prepared box ends outside the image: {tuple(box)!r}")


def _box_edges(box: Sequence[float]) -> tuple[float, float, float, float]:
    x_center, y_center, width, height = box
    return x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValidationError("Cannot calculate a percentile for an empty group.")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _require_complete_matching(matching: Mapping[str, object], context: str) -> None:
    problem_keys = (
        "images_without_labels",
        "labels_without_images",
        "ambiguous_image_stems",
        "ambiguous_label_stems",
    )
    if any(matching[key] for key in problem_keys):
        raise ValidationError(f"Unmatched or ambiguous image-label pairs in {context}.")


def _validate_prepared_yaml_paths(path: Path) -> None:
    expected = {"train": "train/images", "val": "valid/images", "test": "test/images"}
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        for key in expected:
            if stripped.startswith(f"{key}:"):
                found[key] = stripped.split(":", 1)[1].strip().strip("'\"")
    if found != expected:
        raise ValidationError(f"Prepared data.yaml paths are unexpected: {found!r}")


def _validate_locations(source: Path, prepared: Path, output: Path) -> None:
    if not source.is_dir() or not prepared.is_dir():
        raise ValidationError("Raw source and approved prepared dataset directories must exist.")
    for protected in (source, prepared):
        if output == protected or output.is_relative_to(protected):
            raise ValidationError(f"Phase 2B output must not be inside protected dataset data: {output}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValidationError(f"Required Phase 2B artifact would be empty: {path}")
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format(value: float) -> str:
    return f"{value:.12f}"


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _remove_build_directory(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir() or not path.name.startswith(".") or not path.name.endswith("_building"):
        raise ValidationError(f"Refusing to remove unsafe build path: {path}")
    shutil.rmtree(path)


def _finalize_output(build: Path, output: Path, *, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    backup = output.parent / f".{output.name}_backup"
    if output.exists():
        if not overwrite:
            raise ValidationError(f"Phase 2B output appeared during validation: {output}")
        if backup.exists():
            raise ValidationError(f"Stale backup blocks safe replacement: {backup}")
        output.rename(backup)
    try:
        build.rename(output)
    except Exception:
        if backup.exists():
            backup.rename(output)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independently validate the Phase 2A detection conversion.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/bone-fracture-detection/bone-fracture-detection-v3-yolov8"),
    )
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared/v3_detection"))
    parser.add_argument("--phase2a-artifacts", type=Path, default=Path("outputs/phase2a/v3_detection"))
    parser.add_argument("--output", type=Path, default=Path("outputs/phase2b/v3_detection"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        summary = run_validation(
            arguments.source,
            arguments.prepared,
            arguments.phase2a_artifacts,
            arguments.output,
            overwrite=arguments.overwrite,
        )
    except ValidationError as error:
        raise SystemExit(f"Phase 2B validation failed: {error}") from error
    print(
        f"Validated {summary['geometry']['annotations_checked']} conversions; "
        f"generated {summary['visual_review_images']} train/validation review images."
    )
    print(f"Reproducible: {summary['reproducibility']['success']}")
    print(f"Output: {summary['phase2b_output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
