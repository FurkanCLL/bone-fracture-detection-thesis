from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from bone_fracture_audit.audit import IMAGE_EXTENSIONS, _read_dataset_config, match_images_and_labels, sha256_file
from bone_fracture_audit.yolo import parse_yolo_label
from bone_fracture_pipeline.prepare_dataset import CANONICAL_EXPECTATIONS, CLASS_NAMES


ANALYSIS_SPLITS = ("train", "valid")
IMAGE_SIZE = 640
SQUARE_RATIO_TOLERANCE = 0.05
AREA_THRESHOLDS = (0.005, 0.01, 0.02, 0.05, 0.10)
SHORT_SIDE_THRESHOLDS = (8, 16, 32, 64)
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
APPROVED_FULL_FINGERPRINT = "c5031d9937e2a1d917a2369f327b38a59a4b5e8bd40ae2da659451840c7d41b1"


class DifficultyAnalysisError(ValueError):
    """Raised when train/validation inputs do not match the approved detection format."""


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        empty = {key: None for key in ("mean", "median", "std_population", "min", "max")}
        empty.update({f"p{p}": None for p in PERCENTILES})
        return {"count": 0, **empty}
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise DifficultyAnalysisError("A descriptive input is not finite.")
    result: dict[str, float | int | None] = {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std_population": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    result.update({f"p{p}": float(np.percentile(array, p)) for p in PERCENTILES})
    return result


def box_measurements(normalized_width: float, normalized_height: float, image_width: int, image_height: int) -> dict[str, float]:
    if not (0 < normalized_width <= 1 and 0 < normalized_height <= 1 and image_width > 0 and image_height > 0):
        raise DifficultyAnalysisError("Box sizes and image dimensions must be positive and valid.")
    width_px = normalized_width * image_width
    height_px = normalized_height * image_height
    # Letterbox padding moves boxes but does not change their scaled dimensions.
    scale = min(IMAGE_SIZE / image_width, IMAGE_SIZE / image_height)
    effective_width = width_px * scale
    effective_height = height_px * scale
    return {
        "normalized_width": normalized_width,
        "normalized_height": normalized_height,
        "normalized_area": normalized_width * normalized_height,
        "normalized_shorter_side": min(normalized_width, normalized_height),
        "normalized_longer_side": max(normalized_width, normalized_height),
        "width_px": width_px,
        "height_px": height_px,
        "area_px": width_px * height_px,
        "letterbox_scale_640": scale,
        "effective_width_640": effective_width,
        "effective_height_640": effective_height,
        "effective_area_640": effective_width * effective_height,
        "effective_short_side_640": min(effective_width, effective_height),
    }


def orientation(width: int, height: int) -> str:
    ratio = width / height
    if ratio < 1 - SQUARE_RATIO_TOLERANCE:
        return "portrait"
    if ratio > 1 + SQUARE_RATIO_TOLERANCE:
        return "landscape"
    return "approximately_square"


def _paired_files(root: Path, split: str) -> list[tuple[Path, Path]]:
    # Accept only the two review splits, then require a unique label for each image.
    if split not in ANALYSIS_SPLITS:
        raise DifficultyAnalysisError(f"Split {split!r} is outside the train/validation analysis scope.")
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise DifficultyAnalysisError(f"Missing image or label directory for {split}: {root}")
    if any(not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS for path in image_dir.iterdir()):
        raise DifficultyAnalysisError(f"Unexpected image-directory entry in {split}: {root}")
    if any(not path.is_file() or path.suffix.lower() != ".txt" for path in label_dir.iterdir()):
        raise DifficultyAnalysisError(f"Unexpected label-directory entry in {split}: {root}")
    matched = match_images_and_labels(image_dir, label_dir)
    if any(matched[key] for key in (
        "images_without_labels", "labels_without_images", "ambiguous_image_stems", "ambiguous_label_stems"
    )):
        raise DifficultyAnalysisError(f"Unmatched or ambiguous train/validation files in {split}: {root}")
    return sorted(matched["pairs"], key=lambda pair: pair[0].name.casefold())


def _restricted_fingerprint(root: Path, pairs_by_split: Mapping[str, list[tuple[Path, Path]]]) -> dict[str, object]:
    # The digest deliberately excludes every held-out image and label.
    paths = [root / "data.yaml"]
    for split in ANALYSIS_SPLITS:
        paths.extend(path for pair in pairs_by_split[split] for path in pair)
    paths.sort(key=lambda path: (path.relative_to(root).as_posix().casefold(), path.relative_to(root).as_posix()))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(f"{path.relative_to(root).as_posix()}\t{sha256_file(path)}\n".encode("utf-8"))
    return {"sha256": digest.hexdigest(), "file_count": len(paths), "scope": "data.yaml plus train/valid images and labels"}


def _read_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise DifficultyAnalysisError(f"Invalid image dimensions: {path}")
    return width, height


def _collect_split(root: Path, split: str, pairs: list[tuple[Path, Path]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    # Keep one image row even when its label is empty; boxes get separate rows.
    image_rows: list[dict[str, object]] = []
    annotation_rows: list[dict[str, object]] = []
    for image_path, label_path in pairs:
        width, height = _read_dimensions(image_path)
        parsed = parse_yolo_label(label_path, set(range(len(CLASS_NAMES))))
        if parsed.issues or any(annotation.annotation_type != "box" for annotation in parsed.annotations):
            raise DifficultyAnalysisError(f"Invalid canonical detection label: {label_path}")
        if parsed.is_empty and label_path.stat().st_size != 0:
            raise DifficultyAnalysisError(f"An empty canonical label is not zero-byte: {label_path}")
        image_name = image_path.relative_to(root).as_posix()
        image_rows.append({
            "split": split,
            "image": image_name,
            "label": label_path.relative_to(root).as_posix(),
            "width_px": width,
            "height_px": height,
            "pixel_area": width * height,
            "aspect_ratio": width / height,
            "orientation": orientation(width, height),
            "annotation_count": len(parsed.annotations),
            "positive": bool(parsed.annotations),
        })
        for annotation in parsed.annotations:
            annotation_rows.append({
                "split": split,
                "image": image_name,
                "label": label_path.relative_to(root).as_posix(),
                "line_number": annotation.line_number,
                "class_id": annotation.class_id,
                "class_name": CLASS_NAMES[annotation.class_id],
                "x_center": annotation.x_center,
                "y_center": annotation.y_center,
                **box_measurements(annotation.width, annotation.height, width, height),
            })
    return image_rows, annotation_rows


def _thresholds(values: Sequence[float], thresholds: Sequence[float]) -> dict[str, dict[str, float | int]]:
    if not values:
        return {str(threshold): {"count": 0, "fraction": 0.0} for threshold in thresholds}
    counts = {threshold: sum(value < threshold for value in values) for threshold in thresholds}
    return {
        str(threshold): {"count": counts[threshold], "fraction": counts[threshold] / len(values)}
        for threshold in thresholds
    }


def _split_summary(image_rows: list[dict[str, object]], annotation_rows: list[dict[str, object]]) -> dict[str, object]:
    positive_counts = [int(row["annotation_count"]) for row in image_rows if row["positive"]]
    orientations = Counter(str(row["orientation"]) for row in image_rows)
    resolutions = Counter((int(row["width_px"]), int(row["height_px"])) for row in image_rows)
    boxes = {key: describe([float(row[key]) for row in annotation_rows]) for key in (
        "normalized_width", "normalized_height", "normalized_area", "normalized_shorter_side", "normalized_longer_side",
        "width_px", "height_px", "area_px", "effective_width_640", "effective_height_640",
        "effective_area_640", "effective_short_side_640",
    )}
    return {
        "images": len(image_rows),
        "positive_images": len(positive_counts),
        "empty_label_images": len(image_rows) - len(positive_counts),
        "positive_image_percentage": 100 * len(positive_counts) / len(image_rows),
        "empty_label_percentage": 100 * (len(image_rows) - len(positive_counts)) / len(image_rows),
        "annotations": len(annotation_rows),
        "annotations_per_positive_image": describe(positive_counts),
        "dimensions": {
            "width_px": describe([float(row["width_px"]) for row in image_rows]),
            "height_px": describe([float(row["height_px"]) for row in image_rows]),
            "pixel_area": describe([float(row["pixel_area"]) for row in image_rows]),
            "aspect_ratio": describe([float(row["aspect_ratio"]) for row in image_rows]),
            "orientation_counts": {name: orientations[name] for name in ("portrait", "landscape", "approximately_square")},
            "unique_resolution_count": len(resolutions),
            "most_common_resolutions": [
                {"width_px": width, "height_px": height, "images": count}
                for (width, height), count in sorted(resolutions.items(), key=lambda entry: (-entry[1], entry[0]))[:10]
            ],
        },
        "boxes": boxes,
        "normalized_area_below_fraction_of_image": _thresholds(
            [float(row["normalized_area"]) for row in annotation_rows], AREA_THRESHOLDS
        ),
        "effective_short_side_below_px": _thresholds(
            [float(row["effective_short_side_640"]) for row in annotation_rows], SHORT_SIDE_THRESHOLDS
        ),
        "class_annotation_counts": {
            str(class_id): sum(int(row["class_id"]) == class_id for row in annotation_rows)
            for class_id in range(len(CLASS_NAMES))
        },
    }


def _class_rows(annotation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for split in ANALYSIS_SPLITS:
        for class_id, class_name in enumerate(CLASS_NAMES):
            rows = [row for row in annotation_rows if row["split"] == split and row["class_id"] == class_id]
            area = describe([100 * float(row["normalized_area"]) for row in rows])
            short = describe([float(row["effective_short_side_640"]) for row in rows])
            result.append({
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "annotations": len(rows),
                "positive_images": len({str(row["image"]) for row in rows}),
                "median_normalized_area_pct": area["median"],
                "p25_normalized_area_pct": area["p25"],
                "p75_normalized_area_pct": area["p75"],
                "median_effective_width_640": describe([float(row["effective_width_640"]) for row in rows])["median"],
                "median_effective_height_640": describe([float(row["effective_height_640"]) for row in rows])["median"],
                "median_effective_short_side_640": short["median"],
                "p25_effective_short_side_640": short["p25"],
                "p75_effective_short_side_640": short["p75"],
            })
    return result


def _verify_image_conditions(
    source_pairs: Mapping[str, list[tuple[Path, Path]]],
    condition_roots: Mapping[str, Path],
) -> dict[str, object]:
    # Matching label bytes and stored dimensions make source geometry reusable.
    checks: dict[str, object] = {}
    for name, root in sorted(condition_roots.items()):
        if _read_dataset_config(root / "data.yaml") != CLASS_NAMES:
            raise DifficultyAnalysisError(f"{name} class mapping differs from the canonical detection dataset.")
        matched_images = 0
        matched_labels = 0
        matched_dimensions = 0
        for split in ANALYSIS_SPLITS:
            reference = {image.stem.casefold(): (image, label) for image, label in source_pairs[split]}
            condition = {image.stem.casefold(): (image, label) for image, label in _paired_files(root, split)}
            if set(reference) != set(condition):
                raise DifficultyAnalysisError(f"{name} image identities differ in {split}.")
            for stem, (reference_image, reference_label) in reference.items():
                condition_image, condition_label = condition[stem]
                if sha256_file(reference_label) != sha256_file(condition_label):
                    raise DifficultyAnalysisError(f"{name} label geometry differs in {split}/{stem}.")
                if _read_dimensions(reference_image) != _read_dimensions(condition_image):
                    raise DifficultyAnalysisError(f"{name} image dimensions differ in {split}/{stem}.")
                matched_images += 1
                matched_labels += 1
                matched_dimensions += 1
        checks[name] = {
            "root": root.as_posix(), "matched_image_identities": matched_images,
            "matching_label_hashes": matched_labels, "matching_image_dimensions": matched_dimensions,
        }
    return checks


def analyze_dataset(
    source_root: Path,
    *,
    condition_roots: Mapping[str, Path] | None = None,
    expected_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, object]:
    # Fingerprint only the inspected splits before and after reading the source.
    source_root = source_root.resolve()
    if _read_dataset_config(source_root / "data.yaml") != CLASS_NAMES:
        raise DifficultyAnalysisError("Canonical prepared detection classes differ from the approved six classes.")
    pairs_by_split = {split: _paired_files(source_root, split) for split in ANALYSIS_SPLITS}
    before = _restricted_fingerprint(source_root, pairs_by_split)
    image_rows: list[dict[str, object]] = []
    annotation_rows: list[dict[str, object]] = []
    split_summaries: dict[str, object] = {}
    for split in ANALYSIS_SPLITS:
        images, annotations = _collect_split(source_root, split, pairs_by_split[split])
        image_rows.extend(images)
        annotation_rows.extend(annotations)
        summary = _split_summary(images, annotations)
        if expected_counts:
            expected = expected_counts[split]
            for key, expected_value in expected.items():
                if summary[key] != expected_value:
                    raise DifficultyAnalysisError(f"Canonical {split} {key} changed: {summary[key]} != {expected_value}")
        split_summaries[split] = summary
    conditions = _verify_image_conditions(pairs_by_split, condition_roots or {})
    after = _restricted_fingerprint(source_root, pairs_by_split)
    if after != before:
        raise DifficultyAnalysisError("Train/validation source files changed during characterization.")
    return {
        "image_rows": image_rows,
        "annotation_rows": annotation_rows,
        "class_rows": _class_rows(annotation_rows),
        "summary": {
            "schema_version": 1,
            "source_dataset": source_root.as_posix(),
            "historical_approved_full_dataset_fingerprint": APPROVED_FULL_FINGERPRINT,
            "full_dataset_fingerprint_recomputed": False,
            "analyzed_splits": list(ANALYSIS_SPLITS),
            "source_train_valid_fingerprint_before": before,
            "source_train_valid_fingerprint_after": after,
            "source_unchanged": True,
            "condition_geometry_checks": conditions,
            "class_names": list(CLASS_NAMES),
            "method": {
                "box_source": "canonical polygon-extrema detection boxes",
                "image_dimensions": "stored source image width and height; no orientation transform",
                "effective_size": "box pixels multiplied by min(640 / width, 640 / height); padding excluded",
                "square_aspect_ratio_tolerance": SQUARE_RATIO_TOLERANCE,
                "percentile_method": "NumPy linear interpolation",
                "standard_deviation": "population",
                "area_thresholds": list(AREA_THRESHOLDS),
                "short_side_thresholds_px": list(SHORT_SIDE_THRESHOLDS),
            },
            "splits": split_summaries,
            "test_split_opened_or_analyzed": False,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(result: Mapping[str, object], output_dir: Path, figure_dir: Path, evidence_dir: Path) -> list[Path]:
    from bone_fracture_pipeline.dataset_difficulty_figures import generate_figures

    for directory in (output_dir, figure_dir, evidence_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    (output_dir / "dataset_difficulty_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "image_statistics.csv", result["image_rows"])
    _write_csv(output_dir / "annotation_statistics.csv", result["annotation_rows"])
    _write_csv(output_dir / "per_class_statistics.csv", result["class_rows"])
    figures = generate_figures(result["image_rows"], result["annotation_rows"], result["class_rows"], figure_dir)
    compact = {
        "source_dataset": "data/prepared/v3_detection",
        "historical_approved_full_dataset_fingerprint": APPROVED_FULL_FINGERPRINT,
        "full_dataset_fingerprint_recomputed": False,
        "train_valid_fingerprint": summary["source_train_valid_fingerprint_after"],
        "analyzed_splits": list(ANALYSIS_SPLITS),
        "splits": {
            split: {
                **{key: summary["splits"][split][key] for key in (
                    "images", "positive_images", "empty_label_images", "annotations", "positive_image_percentage",
                    "empty_label_percentage", "normalized_area_below_fraction_of_image", "effective_short_side_below_px",
                )},
                "unique_image_resolutions": summary["splits"][split]["dimensions"]["unique_resolution_count"],
                "median_aspect_ratio": summary["splits"][split]["dimensions"]["aspect_ratio"]["median"],
                "median_normalized_box_area_pct": 100 * summary["splits"][split]["boxes"]["normalized_area"]["median"],
                "median_effective_short_side_640_px": summary["splits"][split]["boxes"]["effective_short_side_640"]["median"],
            } for split in ANALYSIS_SPLITS
        },
        "condition_geometry_checks": {
            name: {**check, "root": f"data/prepared/{Path(check['root']).name}"}
            for name, check in summary["condition_geometry_checks"].items()
        },
        "class_statistics": result["class_rows"],
        "source_unchanged": summary["source_unchanged"],
        "test_split_opened_or_analyzed": False,
        "figure_files": [path.name for path in figures],
    }
    (evidence_dir / "dataset_difficulty_summary.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return figures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Characterize canonical detection train and validation data only.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    source = project_root / "data/prepared/v3_detection"
    conditions = {
        "png_control": project_root / "data/prepared/v3_detection_png",
        "clahe": project_root / "data/prepared/v3_detection_clahe",
    }
    expected = {
        split: {
            "images": CANONICAL_EXPECTATIONS.splits[split].images,
            "annotations": CANONICAL_EXPECTATIONS.splits[split].annotations,
            "empty_label_images": CANONICAL_EXPECTATIONS.splits[split].empty_labels,
        } for split in ANALYSIS_SPLITS
    }
    result = analyze_dataset(source, condition_roots=conditions, expected_counts=expected)
    figures = write_analysis(
        result,
        project_root / "outputs/dataset_difficulty",
        project_root / "docs/figures/dataset_difficulty",
        project_root / "docs/evidence/dataset_difficulty",
    )
    print(f"Analyzed train and validation only; wrote {len(figures)} figures and dataset difficulty tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
