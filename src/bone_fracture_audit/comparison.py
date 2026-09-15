from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .audit import sha256_file
from .provenance import compare_image_pair, extract_features


DEFAULT_V3_ROOT = Path("data/raw/bone-fracture-detection/bone-fracture-detection-v3-yolov8")
DEFAULT_V4_ROOT = Path("data/raw/bone-fracture-detection/bone fracture detection.v4-v4.yolov8")
DEFAULT_V3_AUDIT = Path("outputs/dataset_versions/v3/audit")
DEFAULT_V4_AUDIT = Path("outputs/dataset_audit")
DEFAULT_OUTPUT = Path("outputs/dataset_comparison/v3_vs_v4")


@dataclass(frozen=True)
class VersionRecord:
    relative_path: str
    split: str
    filename: str
    source_key: str
    sha256: str
    annotation_count: int
    is_empty_label: bool


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Removes any parent export folders so records from both audits use split-relative paths.
def _split_relative_path(value: str) -> str:
    parts = PurePosixPath(value).parts
    for index, part in enumerate(parts):
        if part in {"train", "valid", "test"}:
            return PurePosixPath(*parts[index:]).as_posix()
    raise ValueError(f"Could not find a dataset split in {value}")


def _load_audit(audit_dir: Path) -> tuple[list[str], list[VersionRecord], dict[str, Any]]:
    summary = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    primary_id = summary["reporting_reference_candidate_id"]
    candidate = next(item for item in summary["candidates"] if item["candidate_id"] == primary_id)
    class_names = candidate["class_names"].split(" | ")
    records: list[VersionRecord] = []
    with (audit_dir / "image_inventory.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["candidate_id"] != primary_id:
                continue
            relative_path = _split_relative_path(row["image_path"])
            records.append(
                VersionRecord(
                    relative_path=relative_path,
                    split=row["split"],
                    filename=PurePosixPath(relative_path).name,
                    source_key=row["source_key"],
                    sha256=row["sha256"],
                    annotation_count=int(row["annotation_count"]),
                    is_empty_label=row["is_empty_label"].casefold() == "true",
                )
            )
    return class_names, records, summary


def _label_path(dataset_root: Path, record: VersionRecord) -> Path:
    return dataset_root / record.split / "labels" / f"{Path(record.filename).stem}.txt"


def _image_path(dataset_root: Path, record: VersionRecord) -> Path:
    return dataset_root / Path(record.relative_path)


def _read_label(path: Path, class_names: list[str]) -> list[tuple[str, tuple[float, ...]]]:
    rows: list[tuple[str, tuple[float, ...]]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if not values:
            continue
        class_id = int(values[0])
        rows.append((class_names[class_id], tuple(float(value) for value in values[1:])))
    return rows


def _geometry_matches(first: tuple[float, ...], second: tuple[float, ...], tolerance: float = 0.001) -> bool:
    if len(first) % 2 or len(second) % 2:
        return False
    first_points = list(zip(first[::2], first[1::2]))
    second_points = list(zip(second[::2], second[1::2]))
    if len(first_points) > 1 and max(abs(a - b) for a, b in zip(first_points[0], first_points[-1])) <= tolerance:
        first_points.pop()
    if len(second_points) > 1 and max(abs(a - b) for a, b in zip(second_points[0], second_points[-1])) <= tolerance:
        second_points.pop()
    if len(first_points) != len(second_points):
        return False
    return all(
        max(abs(first_value - second_value) for first_value, second_value in zip(first_point, second_point)) <= tolerance
        for first_point, second_point in zip(sorted(first_points), sorted(second_points))
    )


# Compares bytes, coordinates, and class names separately so an ID remap is visible.
def compare_label_pair(
    v3_path: Path,
    v4_path: Path,
    v3_classes: list[str],
    v4_classes: list[str],
) -> dict[str, Any]:
    v3_rows = _read_label(v3_path, v3_classes)
    v4_rows = _read_label(v4_path, v4_classes)
    transitions: Counter[tuple[str, str]] = Counter()
    unmatched_v4 = list(v4_rows)
    matched_rows: list[tuple[str, str]] = []
    for v3_name, v3_coordinates in v3_rows:
        match_index = next(
            (index for index, (_, coordinates) in enumerate(unmatched_v4) if _geometry_matches(v3_coordinates, coordinates)),
            None,
        )
        if match_index is not None:
            v4_name, _ = unmatched_v4.pop(match_index)
            transitions[(v3_name, v4_name)] += 1
            matched_rows.append((v3_name, v4_name))
    coordinates_equal = len(matched_rows) == len(v3_rows) == len(v4_rows)
    return {
        "byte_equal": v3_path.read_bytes() == v4_path.read_bytes(),
        "semantic_equal": coordinates_equal and all(first == second for first, second in matched_rows),
        "coordinates_equal": coordinates_equal,
        "v3_annotation_count": len(v3_rows),
        "v4_annotation_count": len(v4_rows),
        "v3_classes": " | ".join(sorted({name for name, _ in v3_rows})),
        "v4_classes": " | ".join(sorted({name for name, _ in v4_rows})),
        "transitions": transitions,
    }


def _compare_held_out_splits(
    v3_root: Path,
    v4_root: Path,
    v3_records: list[VersionRecord],
    v4_records: list[VersionRecord],
    v3_classes: list[str],
    v4_classes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], Counter[tuple[str, str]]]:
    rows: list[dict[str, Any]] = []
    transitions: Counter[tuple[str, str]] = Counter()
    v3_held_out = [record for record in v3_records if record.split in {"valid", "test"}]
    v4_held_out = [record for record in v4_records if record.split in {"valid", "test"}]
    v4_unused = {(record.split, record.filename): record for record in v4_held_out}
    paired: list[tuple[VersionRecord | None, VersionRecord | None]] = []
    for v3_record in v3_held_out:
        v4_record = v4_unused.pop((v3_record.split, v3_record.filename), None)
        if v4_record is None:
            same_source = [
                record
                for record in v4_unused.values()
                if record.split == v3_record.split and record.source_key == v3_record.source_key
            ]
            if len(same_source) == 1:
                v4_record = same_source[0]
                v4_unused.pop((v4_record.split, v4_record.filename))
        paired.append((v3_record, v4_record))
    paired.extend((None, record) for record in v4_unused.values())

    for v3_record, v4_record in sorted(
        paired,
        key=lambda pair: (
            (pair[0] or pair[1]).split,
            (pair[0] or pair[1]).source_key,
            (pair[0] or pair[1]).filename,
        ),
    ):
        split = (v3_record or v4_record).split
        row: dict[str, Any] = {
            "split": split,
            "source_key": (v3_record or v4_record).source_key,
            "v3_filename": v3_record.filename if v3_record else "",
            "v4_filename": v4_record.filename if v4_record else "",
            "in_v3": v3_record is not None,
            "in_v4": v4_record is not None,
            "image_byte_equal": False,
            "image_relationship": "unmatched",
            "image_aligned_correlation": "",
            "image_residual_rmse": "",
            "label_byte_equal": False,
            "label_semantic_equal": False,
            "label_coordinates_equal": False,
            "v3_annotation_count": "",
            "v4_annotation_count": "",
            "v3_classes": "",
            "v4_classes": "",
        }
        if v3_record and v4_record:
            row["image_byte_equal"] = v3_record.sha256 == v4_record.sha256
            if row["image_byte_equal"]:
                row["image_relationship"] = "byte_identical"
                row["image_aligned_correlation"] = 1.0
                row["image_residual_rmse"] = 0.0
            else:
                image_result = compare_image_pair(
                    extract_features(_image_path(v3_root, v3_record)),
                    extract_features(_image_path(v4_root, v4_record)),
                )
                row["image_relationship"] = image_result["classification"]
                row["image_aligned_correlation"] = image_result["aligned_correlation"]
                row["image_residual_rmse"] = image_result["residual_rmse"]
            label_result = compare_label_pair(
                _label_path(v3_root, v3_record),
                _label_path(v4_root, v4_record),
                v3_classes,
                v4_classes,
            )
            transitions.update(label_result.pop("transitions"))
            row.update(
                {
                    "label_byte_equal": label_result["byte_equal"],
                    "label_semantic_equal": label_result["semantic_equal"],
                    "label_coordinates_equal": label_result["coordinates_equal"],
                    "v3_annotation_count": label_result["v3_annotation_count"],
                    "v4_annotation_count": label_result["v4_annotation_count"],
                    "v3_classes": label_result["v3_classes"],
                    "v4_classes": label_result["v4_classes"],
                }
            )
        rows.append(row)

    by_split: dict[str, Any] = {}
    for split in ("valid", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        matched = [row for row in split_rows if row["in_v3"] and row["in_v4"]]
        by_split[split] = {
            "v3_files": sum(row["in_v3"] for row in split_rows),
            "v4_files": sum(row["in_v4"] for row in split_rows),
            "matched_records": len(matched),
            "filename_equal": sum(row["v3_filename"] == row["v4_filename"] for row in matched),
            "image_byte_equal": sum(row["image_byte_equal"] for row in matched),
            "image_relationships": dict(sorted(Counter(row["image_relationship"] for row in matched).items())),
            "label_byte_equal": sum(row["label_byte_equal"] for row in matched),
            "label_semantic_equal": sum(row["label_semantic_equal"] for row in matched),
            "label_coordinates_equal": sum(row["label_coordinates_equal"] for row in matched),
        }
    return rows, by_split, transitions


def _perfect_match() -> dict[str, Any]:
    return {
        "keypoints_first": "",
        "keypoints_second": "",
        "good_matches": "",
        "inliers": "",
        "inlier_ratio": 1.0,
        "rotation_degrees": 0.0,
        "scale": 1.0,
        "translation_fraction": 0.0,
        "overlap_ratio": 1.0,
        "aligned_correlation": 1.0,
        "residual_rmse": 0.0,
        "exposure_slope": 1.0,
        "exposure_intercept": 0.0,
        "classification": "exact_copy",
    }


def _match_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    rank = {
        "exact_copy": 4,
        "high_confidence_derivative": 3,
        "probable_derivative": 2,
        "inconclusive": 1,
        "unrelated_false_positive": 0,
    }[row["classification"]]
    return rank, float(row["aligned_correlation"]), float(row["inlier_ratio"]), -float(row["residual_rmse"])


# Maps every v4 training image to the strongest same-source v3 candidate.
def _compare_training(
    v3_root: Path,
    v4_root: Path,
    v3_records: list[VersionRecord],
    v4_records: list[VersionRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    v3_groups: dict[str, list[VersionRecord]] = defaultdict(list)
    v4_groups: dict[str, list[VersionRecord]] = defaultdict(list)
    for record in v3_records:
        if record.split == "train":
            v3_groups[record.source_key].append(record)
    for record in v4_records:
        if record.split == "train":
            v4_groups[record.source_key].append(record)

    pair_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    mapped_counts: Counter[str] = Counter()
    coverage: Counter[str] = Counter()
    for source_key in sorted(v4_groups):
        v3_group = v3_groups.get(source_key, [])
        v3_features = {
            record.relative_path: extract_features(_image_path(v3_root, record))
            for record in v3_group
        }
        v4_features: dict[str, Any] = {}
        for v4_record in v4_groups[source_key]:
            candidates: list[dict[str, Any]] = []
            for v3_record in v3_group:
                if v3_record.sha256 == v4_record.sha256:
                    metrics = _perfect_match()
                else:
                    if v4_record.relative_path not in v4_features:
                        v4_features[v4_record.relative_path] = extract_features(_image_path(v4_root, v4_record))
                    metrics = compare_image_pair(v3_features[v3_record.relative_path], v4_features[v4_record.relative_path])
                pair_row = {
                    "source_key": source_key,
                    "v3_image": v3_record.relative_path,
                    "v4_image": v4_record.relative_path,
                    **metrics,
                }
                pair_rows.append(pair_row)
                candidates.append(pair_row)

            best = max(candidates, key=_match_score) if candidates else None
            classification = best["classification"] if best else "no_same_source_candidate"
            mapped_counts[classification] += 1
            v3_record = next(
                (record for record in v3_group if best and record.relative_path == best["v3_image"]),
                None,
            )
            if v3_record and classification in {"exact_copy", "high_confidence_derivative", "probable_derivative"}:
                coverage[v3_record.relative_path] += 1
            mapping_rows.append(
                {
                    "source_key": source_key,
                    "v4_image": v4_record.relative_path,
                    "v4_annotation_count": v4_record.annotation_count,
                    "v4_is_empty_label": v4_record.is_empty_label,
                    "v3_candidate_count": len(v3_group),
                    "best_v3_image": best["v3_image"] if best else "",
                    "v3_annotation_count": v3_record.annotation_count if v3_record else "",
                    "v3_is_empty_label": v3_record.is_empty_label if v3_record else "",
                    "classification": classification,
                    "aligned_correlation": best["aligned_correlation"] if best else "",
                    "inlier_ratio": best["inlier_ratio"] if best else "",
                    "residual_rmse": best["residual_rmse"] if best else "",
                    "empty_status_equal": v3_record.is_empty_label == v4_record.is_empty_label if v3_record else "",
                }
            )

    coverage_rows = [
        {
            "v3_image": record.relative_path,
            "source_key": record.source_key,
            "v3_annotation_count": record.annotation_count,
            "v3_is_empty_label": record.is_empty_label,
            "mapped_v4_images": coverage[record.relative_path],
        }
        for record in v3_records
        if record.split == "train"
    ]
    coverage_distribution = Counter(row["mapped_v4_images"] for row in coverage_rows)
    source_key_size_relationships = Counter(
        f"{len(v3_groups.get(source_key, []))}->{len(v4_groups.get(source_key, []))}"
        for source_key in set(v3_groups) | set(v4_groups)
    )
    summary = {
        "v3_train_images": sum(record.split == "train" for record in v3_records),
        "v4_train_images": sum(record.split == "train" for record in v4_records),
        "shared_source_keys": len(set(v3_groups) & set(v4_groups)),
        "v3_source_keys": len(v3_groups),
        "v4_source_keys": len(v4_groups),
        "candidate_pairs": len(pair_rows),
        "expected_v4_images_at_three_outputs_per_v3_image": 3 * sum(record.split == "train" for record in v3_records),
        "shortfall_from_three_outputs_per_v3_image": 3 * sum(record.split == "train" for record in v3_records)
        - sum(record.split == "train" for record in v4_records),
        "source_key_size_relationships_v3_to_v4": dict(sorted(source_key_size_relationships.items())),
        "mapping_classifications": dict(sorted(mapped_counts.items())),
        "v3_source_image_coverage": {str(count): images for count, images in sorted(coverage_distribution.items())},
        "mapped_rows_with_different_empty_status": sum(
            row["empty_status_equal"] is False
            and row["classification"] in {"exact_copy", "high_confidence_derivative", "probable_derivative"}
            for row in mapping_rows
        ),
    }
    return pair_rows, mapping_rows, coverage_rows, summary


def _compare_exact_training_annotations(
    v3_root: Path,
    v4_root: Path,
    v3_records: list[VersionRecord],
    v4_records: list[VersionRecord],
    mapping_rows: list[dict[str, Any]],
    v3_classes: list[str],
    v4_classes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    v3_by_path = {record.relative_path: record for record in v3_records}
    v4_by_path = {record.relative_path: record for record in v4_records}
    rows: list[dict[str, Any]] = []
    transitions: Counter[tuple[str, str]] = Counter()
    for mapping in mapping_rows:
        if mapping["classification"] != "exact_copy":
            continue
        v3_record = v3_by_path[mapping["best_v3_image"]]
        v4_record = v4_by_path[mapping["v4_image"]]
        result = compare_label_pair(
            _label_path(v3_root, v3_record),
            _label_path(v4_root, v4_record),
            v3_classes,
            v4_classes,
        )
        transitions.update(result.pop("transitions"))
        rows.append(
            {
                "source_key": mapping["source_key"],
                "v3_image": v3_record.relative_path,
                "v4_image": v4_record.relative_path,
                **result,
            }
        )
    transition_rows = [
        {"v3_class": first, "v4_class": second, "matched_coordinate_rows": count}
        for (first, second), count in sorted(transitions.items())
    ]
    summary = {
        "exact_image_pairs": len(rows),
        "label_byte_equal": sum(row["byte_equal"] for row in rows),
        "label_semantic_equal": sum(row["semantic_equal"] for row in rows),
        "label_coordinates_equal": sum(row["coordinates_equal"] for row in rows),
    }
    return rows, summary, transition_rows


def _class_comparison(v3_audit: Path, v4_audit: Path) -> list[dict[str, Any]]:
    def load(path: Path) -> dict[tuple[str, str], int]:
        values: dict[tuple[str, str], int] = {}
        with path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                values[(row["split"], row["class_name"])] = int(row["bounding_boxes"])
        return values

    v3 = load(v3_audit / "class_distribution.csv")
    v4 = load(v4_audit / "class_distribution.csv")
    rows: list[dict[str, Any]] = []
    for split in ("train", "valid", "test", "total"):
        names = sorted({name for row_split, name in set(v3) | set(v4) if row_split == split})
        for name in names:
            rows.append(
                {
                    "split": split,
                    "class_name": name,
                    "v3_annotations": v3.get((split, name), 0),
                    "v4_annotations": v4.get((split, name), 0),
                    "difference_v4_minus_v3": v4.get((split, name), 0) - v3.get((split, name), 0),
                }
            )
    return rows


def run_comparison(
    v3_root: Path,
    v4_root: Path,
    v3_audit: Path,
    v4_audit: Path,
    output_dir: Path,
) -> dict[str, Any]:
    v3_root = v3_root.resolve()
    v4_root = v4_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(v3_root) or output_dir.is_relative_to(v4_root):
        raise ValueError("Comparison artifacts must be written outside immutable raw datasets.")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    v3_classes, v3_records, v3_summary = _load_audit(v3_audit)
    v4_classes, v4_records, v4_summary = _load_audit(v4_audit)
    held_out_rows, held_out_summary, transitions = _compare_held_out_splits(
        v3_root, v4_root, v3_records, v4_records, v3_classes, v4_classes
    )
    pair_rows, mapping_rows, coverage_rows, training_summary = _compare_training(
        v3_root, v4_root, v3_records, v4_records
    )
    exact_annotation_rows, exact_annotation_summary, training_transitions = _compare_exact_training_annotations(
        v3_root,
        v4_root,
        v3_records,
        v4_records,
        mapping_rows,
        v3_classes,
        v4_classes,
    )
    training_summary["exact_copy_annotation_comparison"] = exact_annotation_summary
    class_rows = _class_comparison(v3_audit, v4_audit)
    transition_rows = [
        {"v3_class": first, "v4_class": second, "matched_coordinate_rows": count}
        for (first, second), count in sorted(transitions.items())
    ]

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "v3_root": v3_root.as_posix(),
        "v4_root": v4_root.as_posix(),
        "v3_classes": v3_classes,
        "v4_classes": v4_classes,
        "v3_raw_manifest": v3_summary["raw_manifest"],
        "v4_raw_manifest": v4_summary["raw_manifest"],
        "v4_candidate_comparisons": v4_summary["candidate_comparisons"],
        "held_out": held_out_summary,
        "training": training_summary,
        "held_out_class_transitions": transition_rows,
        "training_exact_copy_class_transitions": training_transitions,
    }
    _write_csv(output_dir / "held_out_file_comparison.csv", held_out_rows)
    _write_csv(output_dir / "held_out_class_transitions.csv", transition_rows, ["v3_class", "v4_class", "matched_coordinate_rows"])
    _write_csv(output_dir / "training_candidate_pairs.csv", pair_rows)
    _write_csv(output_dir / "training_v3_to_v4_mapping.csv", mapping_rows)
    _write_csv(output_dir / "training_v3_source_coverage.csv", coverage_rows)
    _write_csv(output_dir / "training_exact_annotation_comparison.csv", exact_annotation_rows)
    _write_csv(
        output_dir / "training_exact_class_transitions.csv",
        training_transitions,
        ["v3_class", "v4_class", "matched_coordinate_rows"],
    )
    _write_csv(output_dir / "class_comparison.csv", class_rows)
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two audited YOLO dataset versions without changing raw files.")
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--v3-audit", type=Path, default=DEFAULT_V3_AUDIT)
    parser.add_argument("--v4-audit", type=Path, default=DEFAULT_V4_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    summary = run_comparison(
        arguments.v3_root,
        arguments.v4_root,
        arguments.v3_audit,
        arguments.v4_audit,
        arguments.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
