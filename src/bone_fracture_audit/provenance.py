from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np


DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_AUDIT_DIR = Path("outputs/dataset_audit")
DEFAULT_OUTPUT_DIR = Path("outputs/dataset_followup/provenance")


@dataclass(frozen=True)
class ImageRecord:
    relative_path: str
    split: str
    source_key: str
    sha256: str
    dhash: str
    annotation_count: int
    is_empty_label: bool


@dataclass
class ImageFeatures:
    grayscale: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


# Loads a bounded grayscale image and ORB features once for repeated pair comparisons.
def extract_features(path: Path, maximum_dimension: int = 640) -> ImageFeatures:
    grayscale = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if grayscale is None:
        raise ValueError(f"Could not decode {path}")
    height, width = grayscale.shape
    scale = min(1.0, maximum_dimension / max(width, height))
    if scale < 1.0:
        grayscale = cv2.resize(grayscale, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    detector = cv2.ORB_create(nfeatures=1800, fastThreshold=8)
    keypoints, descriptors = detector.detectAndCompute(grayscale, None)
    return ImageFeatures(grayscale, keypoints, descriptors)


def _aligned_metrics(
    first: np.ndarray,
    second: np.ndarray,
    matrix: np.ndarray,
) -> tuple[float, float, float, float, float]:
    # Only compare pixels that came from the second image after alignment.
    target_size = (first.shape[1], first.shape[0])
    aligned = cv2.warpAffine(second, matrix, target_size, flags=cv2.INTER_LINEAR, borderValue=0)
    source_mask = np.full(second.shape, 255, dtype=np.uint8)
    overlap_mask = cv2.warpAffine(source_mask, matrix, target_size, flags=cv2.INTER_NEAREST, borderValue=0) > 0
    overlap = float(np.mean(overlap_mask))
    if np.count_nonzero(overlap_mask) < 100:
        return overlap, 0.0, 1.0, 0.0, 0.0

    target_values = first[overlap_mask].astype(np.float32)
    aligned_values = aligned[overlap_mask].astype(np.float32)
    if np.std(target_values) > 1e-6 and np.std(aligned_values) > 1e-6:
        correlation = float(np.corrcoef(target_values, aligned_values)[0, 1])
    else:
        correlation = 0.0
    # A simple linear fit separates exposure changes from structural differences.
    design = np.column_stack((aligned_values, np.ones_like(aligned_values)))
    slope, intercept = np.linalg.lstsq(design, target_values, rcond=None)[0]
    fitted = aligned_values * slope + intercept
    residual_rmse = float(np.sqrt(np.mean((target_values - fitted) ** 2)) / 255.0)
    return overlap, correlation, residual_rmse, float(slope), float(intercept)


# Estimates a rotation/scale/translation relationship and checks whether pixels agree after alignment.
def compare_image_pair(first: ImageFeatures, second: ImageFeatures) -> dict[str, Any]:
    empty_result = {
        "keypoints_first": len(first.keypoints),
        "keypoints_second": len(second.keypoints),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "rotation_degrees": None,
        "scale": None,
        "translation_fraction": None,
        "overlap_ratio": 0.0,
        "aligned_correlation": 0.0,
        "residual_rmse": 1.0,
        "exposure_slope": None,
        "exposure_intercept": None,
        "classification": "unrelated_false_positive",
    }
    if first.descriptors is None or second.descriptors is None:
        return empty_result

    # ORB uses binary descriptors, so Hamming distance is the appropriate matcher.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(second.descriptors, first.descriptors, k=2)
    good_matches = [
        matches[0]
        for matches in raw_matches
        if len(matches) == 2 and matches[0].distance < 0.75 * matches[1].distance
    ]
    empty_result["good_matches"] = len(good_matches)
    if len(good_matches) < 4:
        return empty_result

    source_points = np.float32([second.keypoints[match.queryIdx].pt for match in good_matches])
    target_points = np.float32([first.keypoints[match.trainIdx].pt for match in good_matches])
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.995,
    )
    if matrix is None or inlier_mask is None:
        return empty_result

    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(good_matches)
    scale = math.sqrt(float(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    rotation = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    diagonal = math.hypot(first.grayscale.shape[1], first.grayscale.shape[0])
    translation = math.hypot(float(matrix[0, 2]), float(matrix[1, 2])) / diagonal
    overlap, correlation, residual_rmse, exposure_slope, exposure_intercept = _aligned_metrics(
        first.grayscale, second.grayscale, matrix
    )

    # Conservative thresholds keep the strongest derivative evidence separate from borderline pairs.
    if (
        len(good_matches) >= 12
        and inliers >= 10
        and inlier_ratio >= 0.55
        and 0.8 <= scale <= 1.25
        and overlap >= 0.5
        and correlation >= 0.88
        and residual_rmse <= 0.12
    ):
        classification = "high_confidence_derivative"
    elif (
        len(good_matches) >= 8
        and inliers >= 6
        and inlier_ratio >= 0.35
        and 0.7 <= scale <= 1.35
        and overlap >= 0.35
        and correlation >= 0.7
        and residual_rmse <= 0.2
    ):
        classification = "probable_derivative"
    elif len(good_matches) < 6 or inlier_ratio < 0.2 or correlation < 0.4:
        classification = "unrelated_false_positive"
    else:
        classification = "inconclusive"

    return {
        "keypoints_first": len(first.keypoints),
        "keypoints_second": len(second.keypoints),
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "rotation_degrees": rotation,
        "scale": scale,
        "translation_fraction": translation,
        "overlap_ratio": overlap,
        "aligned_correlation": correlation,
        "residual_rmse": residual_rmse,
        "exposure_slope": exposure_slope,
        "exposure_intercept": exposure_intercept,
        "classification": classification,
    }


# Reuses the Phase 1 inventory instead of scanning and hashing the raw files again.
def _load_records(audit_dir: Path) -> tuple[Path, list[ImageRecord]]:
    summary = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    primary_id = summary["reporting_reference_candidate_id"]
    raw_root = Path(summary["raw_root"])
    records: list[ImageRecord] = []
    with (audit_dir / "image_inventory.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["candidate_id"] != primary_id:
                continue
            records.append(
                ImageRecord(
                    row["image_path"],
                    row["split"],
                    row["source_key"],
                    row["sha256"],
                    row["dhash"],
                    int(row["annotation_count"]),
                    row["is_empty_label"].casefold() == "true",
                )
            )
    return raw_root, records


# Unifies filename groups and Phase 1 hash candidates without treating either source as proof.
def build_candidate_pairs(audit_dir: Path, records: list[ImageRecord]) -> dict[tuple[str, str], set[str]]:
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        source_groups[record.source_key].append(record)
    for members in source_groups.values():
        if len(members) > 1:
            for first, second in combinations(members, 2):
                pair = tuple(sorted((first.relative_path, second.relative_path)))
                pairs[pair].add("shared_source_name")

    with (audit_dir / "near_duplicate_candidates.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            pair = tuple(sorted((row["first_image"], row["second_image"])))
            pairs[pair].add("phase1_dhash")
    return pairs


# Writes small evidence tables with stable column order.
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        if not rows:
            return
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Places two source images side by side without changing either file.
def _draw_pair(first_path: Path, second_path: Path, title: str, output_path: Path) -> None:
    first = cv2.imread(str(first_path))
    second = cv2.imread(str(second_path))
    panels = []
    for label, image in (("A", first), ("B", second)):
        scale = min(520 / image.shape[1], 520 / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
        panel = np.zeros((560, 540, 3), dtype=np.uint8)
        x = (540 - resized.shape[1]) // 2
        y = 30 + (520 - resized.shape[0]) // 2
        panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(panel, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    canvas = np.hstack(panels)
    cv2.putText(canvas, title[:145], (120, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


# Builds high-confidence components so repeated augmentation outputs can be counted without filename collisions.
def _derivative_groups(records: list[ImageRecord], pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records_by_path = {record.relative_path: record for record in records}
    union_find = _UnionFind(list(records_by_path))
    for row in pair_rows:
        if row["classification"] == "high_confidence_derivative":
            union_find.union(row["first_image"], row["second_image"])

    members_by_root: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        members_by_root[union_find.find(record.relative_path)].append(record)
    rows: list[dict[str, Any]] = []
    group_number = 0
    for members in sorted(members_by_root.values(), key=lambda group: group[0].relative_path):
        if len(members) < 2:
            continue
        group_number += 1
        splits = sorted({member.split for member in members})
        source_keys = sorted({member.source_key for member in members})
        for member in members:
            rows.append(
                {
                    "group_id": group_number,
                    "member_count": len(members),
                    "cross_split": len(splits) > 1,
                    "splits": " | ".join(splits),
                    "source_key_count": len(source_keys),
                    "source_keys": " | ".join(source_keys),
                    "split": member.split,
                    "image_path": member.relative_path,
                    "annotation_count": member.annotation_count,
                    "is_empty_label": member.is_empty_label,
                }
            )
    return rows


# Creates one direct empty-versus-annotated comparison for each inconsistent derivative group.
def _write_mixed_label_review(raw_root: Path, output_dir: Path, group_rows: list[dict[str, Any]]) -> None:
    rows_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        rows_by_group[row["group_id"]].append(row)

    review_dir = output_dir / "mixed_label_review"
    review_dir.mkdir()
    review_rows: list[dict[str, Any]] = []
    for group_id, members in sorted(rows_by_group.items()):
        empty_members = [member for member in members if member["is_empty_label"]]
        annotated_members = [member for member in members if not member["is_empty_label"]]
        if not empty_members or not annotated_members:
            continue
        empty_member = empty_members[0]
        annotated_member = annotated_members[0]
        filename = f"mixed_group_{group_id}.jpg"
        title = f"group {group_id} | A: empty label | B: {annotated_member['annotation_count']} annotation"
        _draw_pair(
            raw_root / Path(empty_member["image_path"]),
            raw_root / Path(annotated_member["image_path"]),
            title,
            review_dir / filename,
        )
        review_rows.append(
            {
                "review_image": filename,
                "group_id": group_id,
                "member_count": len(members),
                "empty_member_count": len(empty_members),
                "annotated_member_count": len(annotated_members),
                "empty_image": empty_member["image_path"],
                "annotated_image": annotated_member["image_path"],
            }
        )
    _write_csv(review_dir / "index.csv", review_rows)


def _summary(
    records: list[ImageRecord],
    pair_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    humerus_paths: set[str],
) -> dict[str, Any]:
    classification_counts = Counter(row["classification"] for row in pair_rows)
    cross_split_rows = [row for row in pair_rows if row["cross_split"]]
    cross_split_counts = Counter(row["classification"] for row in cross_split_rows)
    train_source_groups = defaultdict(list)
    for record in records:
        if record.split == "train":
            train_source_groups[record.source_key].append(record)
    train_group_sizes = Counter(len(group) for group in train_source_groups.values())
    mixed_train_group_sizes = Counter(
        len(group)
        for group in train_source_groups.values()
        if len({member.is_empty_label for member in group}) > 1
    )
    derivative_group_ids = {row["group_id"] for row in group_rows}
    cross_split_group_ids = {row["group_id"] for row in group_rows if row["cross_split"]}
    humerus_groups = {
        row["group_id"]
        for row in group_rows
        if row["image_path"] in humerus_paths
    }
    group_empty_statuses: dict[int, set[bool]] = defaultdict(set)
    for row in group_rows:
        group_empty_statuses[row["group_id"]].add(row["is_empty_label"])
    high_rows = [row for row in pair_rows if row["classification"] == "high_confidence_derivative"]
    return {
        "software_versions": {"opencv": cv2.__version__, "numpy": np.__version__},
        "matching_settings": {
            "maximum_image_dimension": 640,
            "orb_features": 1800,
            "orb_fast_threshold": 8,
            "descriptor_ratio_threshold": 0.75,
            "ransac_reprojection_threshold": 3.0,
        },
        "images_considered": len(records),
        "candidate_pairs": len(pair_rows),
        "pair_classifications": dict(sorted(classification_counts.items())),
        "cross_split_pair_classifications": dict(sorted(cross_split_counts.items())),
        "high_confidence_groups": len(derivative_group_ids),
        "cross_split_high_confidence_groups": len(cross_split_group_ids),
        "train_source_name_group_sizes": {str(size): count for size, count in sorted(train_group_sizes.items())},
        "train_source_name_groups_with_mixed_empty_status": sum(mixed_train_group_sizes.values()),
        "train_source_name_groups_with_mixed_empty_status_by_size": {
            str(size): count for size, count in sorted(mixed_train_group_sizes.items())
        },
        "high_confidence_groups_with_mixed_empty_status": sum(
            len(statuses) > 1 for statuses in group_empty_statuses.values()
        ),
        "high_confidence_rotation_pairs_over_1_degree": sum(abs(row["rotation_degrees"]) >= 1.0 for row in high_rows),
        "high_confidence_exposure_pairs_slope_outside_0_9_1_1": sum(
            row["exposure_slope"] < 0.9 or row["exposure_slope"] > 1.1 for row in high_rows
        ),
        "humerus_fracture_images": len(humerus_paths),
        "humerus_fracture_high_confidence_groups": len(humerus_groups),
    }


# Checks class overlap directly from the Phase 1 box table.
def _semantic_summary(audit_dir: Path, records_by_path: dict[str, ImageRecord]) -> dict[str, Any]:
    classes_by_image: dict[str, set[str]] = defaultdict(set)
    areas_by_class: dict[str, list[float]] = defaultdict(list)
    areas_by_image_class: dict[tuple[str, str], list[float]] = defaultdict(list)
    with (audit_dir / "bounding_boxes.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["image_path"] not in records_by_path:
                continue
            classes_by_image[row["image_path"]].add(row["class_name"])
            area = float(row["relative_area"])
            areas_by_class[row["class_name"]].append(area)
            areas_by_image_class[(row["image_path"], row["class_name"])].append(area)

    image_counts = Counter(class_name for names in classes_by_image.values() for class_name in names)
    cooccurrences = Counter(
        tuple(sorted(pair))
        for names in classes_by_image.values()
        for pair in combinations(names, 2)
    )
    humerus_fracture_images = {
        image_path for image_path, names in classes_by_image.items() if "humerus fracture" in names
    }
    humerus_fracture_areas = [
        area
        for image_path in humerus_fracture_images
        for area in areas_by_image_class[(image_path, "humerus fracture")]
    ]
    humerus_areas_on_same_images = [
        area
        for image_path in humerus_fracture_images
        for area in areas_by_image_class[(image_path, "humerus")]
    ]
    return {
        "images_per_class": dict(sorted(image_counts.items())),
        "class_cooccurrences": {
            " | ".join(pair): count for pair, count in sorted(cooccurrences.items())
        },
        "humerus_fracture_images": len(humerus_fracture_images),
        "humerus_fracture_images_also_labeled_humerus": sum(
            "humerus" in classes_by_image[image_path] for image_path in humerus_fracture_images
        ),
        "humerus_fracture_median_relative_area": median(humerus_fracture_areas) if humerus_fracture_areas else None,
        "humerus_median_relative_area_on_same_images": (
            median(humerus_areas_on_same_images) if humerus_areas_on_same_images else None
        ),
        "all_annotation_median_relative_area_by_class": {
            class_name: median(areas) for class_name, areas in sorted(areas_by_class.items())
        },
    }


# Runs the expensive image matching once and saves every classification for inspection.
def analyze_provenance(audit_dir: Path, output_dir: Path) -> dict[str, Any]:
    audit_dir = audit_dir.resolve()
    output_dir = output_dir.resolve()
    raw_root, records = _load_records(audit_dir)
    if output_dir.is_relative_to(raw_root):
        raise ValueError("Provenance outputs must be outside the immutable raw dataset.")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    records_by_path = {record.relative_path: record for record in records}
    candidate_pairs = build_candidate_pairs(audit_dir, records)
    feature_cache: dict[str, ImageFeatures] = {}

    def features(relative_path: str) -> ImageFeatures:
        # Most images appear in several pairs, so caching avoids repeated ORB work.
        if relative_path not in feature_cache:
            feature_cache[relative_path] = extract_features(raw_root / Path(relative_path))
        return feature_cache[relative_path]

    pair_rows: list[dict[str, Any]] = []
    for first_path, second_path in sorted(candidate_pairs):
        first_record = records_by_path[first_path]
        second_record = records_by_path[second_path]
        metrics = compare_image_pair(features(first_path), features(second_path))
        pair_rows.append(
            {
                "candidate_reasons": " | ".join(sorted(candidate_pairs[(first_path, second_path)])),
                "cross_split": first_record.split != second_record.split,
                "same_source_key": first_record.source_key == second_record.source_key,
                "first_split": first_record.split,
                "first_source_key": first_record.source_key,
                "first_image": first_path,
                "second_split": second_record.split,
                "second_source_key": second_record.source_key,
                "second_image": second_path,
                **metrics,
            }
        )

    group_rows = _derivative_groups(records, pair_rows)
    # Keep the humerus-fracture evidence separate so version-specific scarcity can be inspected.
    humerus_paths: set[str] = set()
    with (audit_dir / "bounding_boxes.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["class_name"] == "humerus fracture" and row["image_path"] in records_by_path:
                humerus_paths.add(row["image_path"])
    summary = _summary(records, pair_rows, group_rows, humerus_paths)
    summary["semantic_checks"] = _semantic_summary(audit_dir, records_by_path)

    _write_csv(output_dir / "pair_analysis.csv", pair_rows)
    _write_csv(output_dir / "derivative_groups.csv", group_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_mixed_label_review(raw_root, output_dir, group_rows)

    # Put the most concerning cross-split candidates first for quick visual review.
    review_dir = output_dir / "cross_split_review"
    review_dir.mkdir()
    priority = {"high_confidence_derivative": 0, "probable_derivative": 1, "inconclusive": 2, "unrelated_false_positive": 3}
    ranked_cross_split = sorted(
        (row for row in pair_rows if row["cross_split"]),
        key=lambda row: (priority[row["classification"]], row["residual_rmse"], -row["inliers"]),
    )
    review_rows: list[dict[str, Any]] = []
    # Thirty rows include every current inconclusive case plus a few rejected controls.
    for index, row in enumerate(ranked_cross_split[:30], start=1):
        filename = f"pair_{index:02d}.jpg"
        title = (
            f"{row['classification']} | r={row['aligned_correlation']:.3f} | "
            f"angle={row['rotation_degrees'] if row['rotation_degrees'] is not None else 0:.1f}"
        )
        _draw_pair(raw_root / Path(row["first_image"]), raw_root / Path(row["second_image"]), title, review_dir / filename)
        review_rows.append({"review_image": filename, **row})
    _write_csv(review_dir / "index.csv", review_rows)

    # Preserve every borderline pair as a compact visual-review artifact.
    inconclusive_dir = output_dir / "inconclusive_review"
    inconclusive_dir.mkdir()
    inconclusive_rows: list[dict[str, Any]] = []
    for index, row in enumerate((row for row in pair_rows if row["classification"] == "inconclusive"), start=1):
        filename = f"pair_{index:02d}.jpg"
        title = (
            f"inconclusive | r={row['aligned_correlation']:.3f} | "
            f"angle={row['rotation_degrees'] if row['rotation_degrees'] is not None else 0:.1f}"
        )
        _draw_pair(
            raw_root / Path(row["first_image"]),
            raw_root / Path(row["second_image"]),
            title,
            inconclusive_dir / filename,
        )
        inconclusive_rows.append({"review_image": filename, **row})
    _write_csv(inconclusive_dir / "index.csv", inconclusive_rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze possible transformed derivatives in the audited dataset.")
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    summary = analyze_provenance(arguments.audit_dir, arguments.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
