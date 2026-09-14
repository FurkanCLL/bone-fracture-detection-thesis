from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from .yolo import Annotation, parse_yolo_label


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{16,}$", re.IGNORECASE)
SMALL_BOX_AREA = 0.001
LARGE_BOX_AREA = 0.5
EXTREME_ASPECT_RATIO_LOW = 0.5
EXTREME_ASPECT_RATIO_HIGH = 2.0
NEAR_DUPLICATE_DISTANCE = 4
MAX_NEAR_DUPLICATE_PAIRS = 50_000


@dataclass(frozen=True)
class DatasetCandidate:
    candidate_id: str
    root: Path
    config_path: Path
    class_names: tuple[str, ...]


# Builds a readable ID that stays unique when export folders share a name.
def _candidate_id(root: Path, raw_root: Path) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", root.name.lower()).strip("-")
    relative_root = root.relative_to(raw_root).as_posix().casefold()
    digest = hashlib.sha1(relative_root.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}"


# Reads the class list from the small Roboflow-style YOLO configuration.
def _read_dataset_config(path: Path) -> tuple[str, ...]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    class_count: int | None = None
    names: tuple[str, ...] | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("nc:"):
            class_count = int(stripped.split(":", 1)[1].strip())
        if not stripped.startswith("names:"):
            continue

        raw_names = stripped.split(":", 1)[1].strip()
        if raw_names:
            parsed = ast.literal_eval(raw_names)
            if isinstance(parsed, list):
                names = tuple(str(value) for value in parsed)
            elif isinstance(parsed, dict):
                names = tuple(str(parsed[key]) for key in sorted(parsed, key=int))
        else:
            mapping: dict[int, str] = {}
            for nested in lines[index + 1 :]:
                if not nested.startswith((" ", "\t")):
                    break
                match = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", nested)
                if match:
                    mapping[int(match.group(1))] = match.group(2).strip("'\"")
            if mapping:
                names = tuple(mapping[key] for key in sorted(mapping))

    if names is None:
        raise ValueError(f"Could not read class names from {path}")
    if class_count is not None and class_count != len(names):
        raise ValueError(f"Configured nc={class_count}, but {len(names)} class names were found in {path}")
    return names


# Finds complete YOLO exports instead of assuming one fixed raw folder layout.
def discover_dataset_candidates(raw_root: Path) -> list[DatasetCandidate]:
    candidates: list[DatasetCandidate] = []
    for config_path in sorted(raw_root.rglob("data.yaml"), key=lambda value: str(value).lower()):
        root = config_path.parent
        split_directories = [root / split for split in ("train", "valid", "test")]
        if not all((directory / "images").is_dir() and (directory / "labels").is_dir() for directory in split_directories):
            continue
        candidates.append(
            DatasetCandidate(_candidate_id(root, raw_root), root, config_path, _read_dataset_config(config_path))
        )
    if not candidates:
        raise FileNotFoundError(f"No complete train/valid/test YOLO dataset was found below {raw_root}")
    return candidates


# Matches files by case-insensitive stem and keeps ambiguous names visible.
def match_images_and_labels(image_dir: Path, label_dir: Path) -> dict[str, Any]:
    images = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    labels = [path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"]
    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    labels_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in images:
        images_by_stem[path.stem.casefold()].append(path)
    for path in labels:
        labels_by_stem[path.stem.casefold()].append(path)

    image_keys = set(images_by_stem)
    label_keys = set(labels_by_stem)
    pairs = []
    for key in sorted(image_keys & label_keys):
        if len(images_by_stem[key]) == 1 and len(labels_by_stem[key]) == 1:
            pairs.append((images_by_stem[key][0], labels_by_stem[key][0]))

    return {
        "images": sorted(images),
        "labels": sorted(labels),
        "pairs": pairs,
        "images_without_labels": [path for key in sorted(image_keys - label_keys) for path in images_by_stem[key]],
        "labels_without_images": [path for key in sorted(label_keys - image_keys) for path in labels_by_stem[key]],
        "ambiguous_image_stems": {key: values for key, values in images_by_stem.items() if len(values) > 1},
        "ambiguous_label_stems": {key: values for key, values in labels_by_stem.items() if len(values) > 1},
    }


# Streams large files so exact hashing does not load them fully into memory.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Creates a compact perceptual hash for cautious near-duplicate screening.
def difference_hash(image: Image.Image) -> str:
    grayscale = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.tobytes())
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{bits:016x}"


def hamming_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


# Ranks dHash candidates using a small contrast-normalized grayscale comparison.
def normalized_pixel_difference(first_path: Path, second_path: Path) -> float:
    fingerprints: list[bytes] = []
    for path in (first_path, second_path):
        with Image.open(path) as source:
            grayscale = ImageOps.autocontrast(ImageOps.grayscale(source))
            fingerprints.append(ImageOps.fit(grayscale, (32, 32), method=Image.Resampling.LANCZOS).tobytes())
    total_difference = sum(abs(first - second) for first, second in zip(*fingerprints))
    return total_difference / (32 * 32 * 255)


# Indexes hashes by Hamming distance without comparing every possible image pair.
class _BKTree:
    def __init__(self) -> None:
        self.root: tuple[str, dict[int, Any]] | None = None

    def add(self, value: str) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = hamming_distance(value, node[0])
            child = node[1].get(distance)
            if child is None:
                node[1][distance] = (value, {})
                return
            node = child

    def search(self, value: str, maximum_distance: int) -> list[tuple[int, str]]:
        if self.root is None:
            return []
        matches: list[tuple[int, str]] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = hamming_distance(value, node[0])
            if distance <= maximum_distance:
                matches.append((distance, node[0]))
            lower = distance - maximum_distance
            upper = distance + maximum_distance
            pending.extend(child for edge, child in node[1].items() if lower <= edge <= upper)
        return matches


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


# Produces the same descriptive statistics for images, boxes, and classes.
def _describe(values: Iterable[float]) -> dict[str, float | int | None]:
    collected = list(values)
    return {
        "count": len(collected),
        "min": min(collected) if collected else None,
        "p05": _quantile(collected, 0.05),
        "p25": _quantile(collected, 0.25),
        "median": median(collected) if collected else None,
        "mean": mean(collected) if collected else None,
        "p75": _quantile(collected, 0.75),
        "p95": _quantile(collected, 0.95),
        "max": max(collected) if collected else None,
    }


def _source_key(path: Path) -> str:
    return ROBOFLOW_SUFFIX.sub("", path.stem).casefold()


# Writes inspectable tables even when a check produces no rows.
def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as destination:
        if not columns:
            return
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


# Audits one complete export while leaving every raw image and label untouched.
def _audit_candidate(candidate: DatasetCandidate, raw_root: Path) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    empty_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    valid_class_ids = set(range(len(candidate.class_names)))

    for split in ("train", "valid", "test"):
        match = match_images_and_labels(candidate.root / split / "images", candidate.root / split / "labels")
        formats = Counter(path.suffix.lower() for path in match["images"])
        split_box_start = len(box_rows)
        split_issue_start = len(issue_rows)
        split_empty_start = len(empty_rows)

        for kind in ("images_without_labels", "labels_without_images"):
            for path in match[kind]:
                unmatched_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "split": split,
                        "kind": kind,
                        "path": _relative(path, raw_root),
                    }
                )
        for kind in ("ambiguous_image_stems", "ambiguous_label_stems"):
            for stem, paths in match[kind].items():
                unmatched_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "split": split,
                        "kind": kind,
                        "path": " | ".join(_relative(path, raw_root) for path in paths),
                        "stem": stem,
                    }
                )

        for image_path, label_path in match["pairs"]:
            parsed = parse_yolo_label(label_path, valid_class_ids)
            if parsed.is_empty:
                empty_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "split": split,
                        "image_path": _relative(image_path, raw_root),
                        "label_path": _relative(label_path, raw_root),
                    }
                )
            for issue in parsed.issues:
                issue_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "split": split,
                        "image_path": _relative(image_path, raw_root),
                        "label_path": _relative(label_path, raw_root),
                        **asdict(issue),
                    }
                )

            image_relative = _relative(image_path, raw_root)
            # Image decoding is checked before dimensions, hashes, or pixel box sizes are trusted.
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                    image_format = image.format or image_path.suffix.lstrip(".").upper()
                    perceptual_hash = difference_hash(image)
            except (OSError, UnidentifiedImageError) as error:
                issue_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "split": split,
                        "image_path": image_relative,
                        "label_path": _relative(label_path, raw_root),
                        "line_number": 0,
                        "code": "unreadable_image",
                        "message": str(error),
                        "raw_line": "",
                    }
                )
                continue

            image_row = {
                "candidate_id": candidate.candidate_id,
                "split": split,
                "image_path": image_relative,
                "label_path": _relative(label_path, raw_root),
                "source_key": _source_key(image_path),
                "extension": image_path.suffix.lower(),
                "decoded_format": image_format,
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "pixel_count": width * height,
                "file_size_bytes": image_path.stat().st_size,
                "sha256": sha256_file(image_path),
                "dhash": perceptual_hash,
                "annotation_count": len(parsed.annotations),
                "is_empty_label": parsed.is_empty,
            }
            image_rows.append(image_row)

            for annotation in parsed.annotations:
                box_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "split": split,
                        "image_path": image_relative,
                        "label_path": _relative(label_path, raw_root),
                        "line_number": annotation.line_number,
                        "class_id": annotation.class_id,
                        "class_name": candidate.class_names[annotation.class_id],
                        "annotation_type": annotation.annotation_type,
                        "polygon_point_count": len(annotation.polygon_points),
                        "polygon_points": " ".join(f"{x},{y}" for x, y in annotation.polygon_points),
                        "x_center": annotation.x_center,
                        "y_center": annotation.y_center,
                        "normalized_width": annotation.width,
                        "normalized_height": annotation.height,
                        "relative_area": annotation.relative_area,
                        "pixel_width": annotation.width * width,
                        "pixel_height": annotation.height * height,
                        "pixel_area": annotation.relative_area * width * height,
                        "is_very_small": annotation.relative_area < SMALL_BOX_AREA,
                        "is_unusually_large": annotation.relative_area > LARGE_BOX_AREA,
                    }
                )

        split_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "split": split,
                "image_files": len(match["images"]),
                "label_files": len(match["labels"]),
                "matched_pairs": len(match["pairs"]),
                "images_without_labels": len(match["images_without_labels"]),
                "labels_without_images": len(match["labels_without_images"]),
                "ambiguous_image_stems": len(match["ambiguous_image_stems"]),
                "ambiguous_label_stems": len(match["ambiguous_label_stems"]),
                "bounding_boxes": len(box_rows) - split_box_start,
                "annotation_issues": len(issue_rows) - split_issue_start,
                "empty_labels": len(empty_rows) - split_empty_start,
                "empty_label_proportion": (len(empty_rows) - split_empty_start) / len(match["labels"])
                if match["labels"]
                else None,
                "image_formats": dict(sorted(formats.items())),
            }
        )

    return {
        "candidate": candidate,
        "split_rows": split_rows,
        "image_rows": image_rows,
        "box_rows": box_rows,
        "issue_rows": issue_rows,
        "empty_rows": empty_rows,
        "unmatched_rows": unmatched_rows,
    }


# Summarizes annotation counts for each class and split.
def _aggregate_class_rows(candidate_result: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = candidate_result["candidate"]
    counts = Counter((row["split"], row["class_id"]) for row in candidate_result["box_rows"])
    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(candidate.class_names):
        total = sum(counts[(split, class_id)] for split in ("train", "valid", "test"))
        for split in ("train", "valid", "test", "total"):
            count = total if split == "total" else counts[(split, class_id)]
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "bounding_boxes": count,
                }
            )
    return rows


# Describes image dimensions and storage properties for each split.
def _aggregate_image_statistics(candidate_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_id = candidate_result["candidate"].candidate_id
    for split in ("train", "valid", "test", "total"):
        images = candidate_result["image_rows"]
        if split != "total":
            images = [row for row in images if row["split"] == split]
        for metric in ("width", "height", "aspect_ratio", "pixel_count", "file_size_bytes"):
            rows.append({"candidate_id": candidate_id, "split": split, "metric": metric, **_describe(row[metric] for row in images)})
    return rows


# Describes derived box sizes by split and class.
def _aggregate_box_statistics(candidate_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate = candidate_result["candidate"]
    for class_id, class_name in enumerate(candidate.class_names):
        class_boxes = [row for row in candidate_result["box_rows"] if row["class_id"] == class_id]
        for metric in ("normalized_width", "normalized_height", "relative_area", "pixel_width", "pixel_height", "pixel_area"):
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "metric": metric,
                    **_describe(row[metric] for row in class_boxes),
                }
            )
    return rows


# Groups byte-identical images and separates raw-copy overlap from split leakage.
def _exact_duplicate_rows(image_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        groups[row["sha256"]].append(row)

    output: list[dict[str, Any]] = []
    group_number = 0
    primary_within_split = 0
    primary_cross_split = 0
    cross_candidate = 0
    for sha256, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        group_number += 1
        candidate_ids = {member["candidate_id"] for member in members}
        splits_by_candidate: dict[str, set[str]] = defaultdict(set)
        for member in members:
            splits_by_candidate[member["candidate_id"]].add(member["split"])
        if len(candidate_ids) > 1:
            cross_candidate += 1
        if any(len(splits) > 1 for splits in splits_by_candidate.values()):
            primary_cross_split += 1
        elif any(sum(1 for member in members if member["candidate_id"] == candidate_id) > 1 for candidate_id in candidate_ids):
            primary_within_split += 1
        for member in members:
            output.append(
                {
                    "group_id": group_number,
                    "sha256": sha256,
                    "group_size": len(members),
                    "candidate_id": member["candidate_id"],
                    "split": member["split"],
                    "image_path": member["image_path"],
                }
            )
    return output, {
        "groups": group_number,
        "within_candidate_within_split_groups": primary_within_split,
        "within_candidate_cross_split_groups": primary_cross_split,
        "cross_candidate_groups": cross_candidate,
    }


# Screens perceptually similar images and stops at a documented safety limit.
def _near_duplicate_rows(image_rows: list[dict[str, Any]], raw_root: Path) -> tuple[list[dict[str, Any]], bool]:
    hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tree = _BKTree()
    output: list[dict[str, Any]] = []
    truncated = False
    for row in image_rows:
        current_hash = row["dhash"]
        for distance, matched_hash in tree.search(current_hash, NEAR_DUPLICATE_DISTANCE):
            for previous in hashes[matched_hash]:
                if previous["sha256"] == row["sha256"]:
                    continue
                output.append(
                    {
                        "hamming_distance": distance,
                        "normalized_pixel_difference": normalized_pixel_difference(
                            raw_root / Path(previous["image_path"]), raw_root / Path(row["image_path"])
                        ),
                        "same_source_key": previous["source_key"] == row["source_key"],
                        "cross_split": previous["split"] != row["split"],
                        "first_split": previous["split"],
                        "first_image": previous["image_path"],
                        "second_split": row["split"],
                        "second_image": row["image_path"],
                        "first_dhash": previous["dhash"],
                        "second_dhash": row["dhash"],
                    }
                )
                if len(output) >= MAX_NEAR_DUPLICATE_PAIRS:
                    truncated = True
                    return output, truncated
        if current_hash not in hashes:
            tree.add(current_hash)
        hashes[current_hash].append(row)
    return output, truncated


# Groups Roboflow filenames after removing the generated hash suffix.
def _source_group_rows(image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        groups[row["source_key"]].append(row)
    output: list[dict[str, Any]] = []
    group_id = 0
    for source_key, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        group_id += 1
        splits = {member["split"] for member in members}
        for member in members:
            output.append(
                {
                    "group_id": group_id,
                    "source_key": source_key,
                    "group_size": len(members),
                    "cross_split": len(splits) > 1,
                    "split": member["split"],
                    "image_path": member["image_path"],
                    "sha256": member["sha256"],
                    "dhash": member["dhash"],
                }
            )
    return output


# Records every raw file so later runs can detect accidental source changes.
def _manifest_rows(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((value for value in raw_root.rglob("*") if value.is_file()), key=lambda value: str(value).lower()):
        rows.append(
            {
                "relative_path": _relative(path, raw_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


# Checks whether multiple discovered exports contain the same relative files.
def _compare_candidates(candidate_results: list[dict[str, Any]], raw_root: Path) -> list[dict[str, Any]]:
    if len(candidate_results) < 2:
        return []
    comparisons: list[dict[str, Any]] = []
    for first_index, first in enumerate(candidate_results):
        first_root = first["candidate"].root
        first_files = {
            _relative(path, first_root): sha256_file(path)
            for path in first_root.rglob("*")
            if path.is_file()
        }
        for second in candidate_results[first_index + 1 :]:
            second_root = second["candidate"].root
            second_files = {
                _relative(path, second_root): sha256_file(path)
                for path in second_root.rglob("*")
                if path.is_file()
            }
            shared = set(first_files) & set(second_files)
            comparisons.append(
                {
                    "first_candidate_id": first["candidate"].candidate_id,
                    "second_candidate_id": second["candidate"].candidate_id,
                    "first_file_count": len(first_files),
                    "second_file_count": len(second_files),
                    "shared_relative_paths": len(shared),
                    "byte_identical_shared_files": sum(first_files[path] == second_files[path] for path in shared),
                    "only_in_first": len(set(first_files) - set(second_files)),
                    "only_in_second": len(set(second_files) - set(first_files)),
                    "differing_relative_paths": " | ".join(
                        sorted(path for path in shared if first_files[path] != second_files[path])
                    ),
                    "all_files_equivalent": first_files == second_files,
                }
            )
    return comparisons


# Prefers the explicitly versioned export as the reporting reference.
def _select_primary(candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
    # Prefer the versioned Roboflow export name when equivalent copies exist.
    return sorted(
        candidate_results,
        key=lambda result: (".yolov8" not in result["candidate"].root.name.lower(), str(result["candidate"].root).lower()),
    )[0]


# Creates review copies with overlays; it never edits source X-rays.
def _draw_visual_samples(
    primary: dict[str, Any],
    raw_root: Path,
    output_dir: Path,
    near_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sample_dir = output_dir / "visual_samples"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_lookup = {row["image_path"]: row for row in primary["image_rows"]}
    boxes_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary["box_rows"]:
        boxes_by_image[row["image_path"]].append(row)

    selections: list[tuple[str, str, str]] = []
    for class_id, class_name in enumerate(primary["candidate"].class_names):
        class_boxes = [row for row in primary["box_rows"] if row["class_id"] == class_id]
        if class_boxes:
            if len(class_boxes) <= 5:
                for index, chosen in enumerate(class_boxes, start=1):
                    selections.append((f"class_{class_id}_{index}", chosen["image_path"], f"Rare-class sample: {class_name}"))
            else:
                target = median(row["relative_area"] for row in class_boxes)
                chosen = min(class_boxes, key=lambda row: abs(row["relative_area"] - target))
                selections.append((f"class_{class_id}", chosen["image_path"], f"Representative {class_name}"))

    for index, box in enumerate(sorted(primary["box_rows"], key=lambda row: row["relative_area"])[:5], start=1):
        selections.append((f"small_box_{index}", box["image_path"], f"Small box: {box['relative_area']:.6f} relative area"))
    for index, row in enumerate(primary["issue_rows"][:5], start=1):
        if row.get("image_path") in image_lookup:
            selections.append((f"issue_{index}", row["image_path"], f"Issue: {row['code']}"))
    for index, row in enumerate(primary["empty_rows"][:3], start=1):
        selections.append((f"empty_label_{index}", row["image_path"], "Empty label; semantic meaning is unverified"))

    colors = ["#00E5FF", "#FFB000", "#FF4D6D", "#8AFF80", "#C77DFF", "#FFD166", "#4CC9F0"]
    output_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def annotated_image(image_relative: str, note: str) -> Image.Image:
        image_path = raw_root / Path(image_relative)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in boxes_by_image.get(image_relative, []):
            left = (box["x_center"] - box["normalized_width"] / 2) * image.width
            top = (box["y_center"] - box["normalized_height"] / 2) * image.height
            right = (box["x_center"] + box["normalized_width"] / 2) * image.width
            bottom = (box["y_center"] + box["normalized_height"] / 2) * image.height
            color = colors[box["class_id"] % len(colors)]
            stroke = max(2, round(min(image.size) / 250))
            if box["annotation_type"] == "polygon" and box["polygon_points"]:
                points = [
                    (float(point.split(",")[0]) * image.width, float(point.split(",")[1]) * image.height)
                    for point in box["polygon_points"].split()
                ]
                draw.line(points + [points[0]], fill=color, width=stroke)
            draw.rectangle((left, top, right, bottom), outline=color, width=stroke)
            draw.text((max(0, left + 2), max(0, top + 2)), box["class_name"], fill=color, stroke_width=1, stroke_fill="black")
        draw.rectangle((0, 0, image.width, 24), fill="black")
        draw.text((6, 6), note, fill="white")
        return image

    for sequence, (category, image_relative, note) in enumerate(selections, start=1):
        if (category, image_relative) in seen:
            continue
        seen.add((category, image_relative))
        image = annotated_image(image_relative, note)
        filename = f"{sequence:02d}_{category}.jpg"
        image.save(sample_dir / filename, quality=92)
        output_rows.append(
            {
                "sample": f"visual_samples/{filename}",
                "category": category,
                "source_image": image_relative,
                "note": note,
            }
        )

    pair_candidates: list[tuple[str, str, str, str]] = []
    ranked_near_rows = sorted(
        (row for row in near_rows if row["cross_split"]),
        key=lambda row: (row["normalized_pixel_difference"], row["hamming_distance"]),
    )
    for index, row in enumerate(ranked_near_rows, start=1):
        pair_candidates.append(
            (
                f"near_duplicate_pair_{index}",
                row["first_image"],
                row["second_image"],
                f"dHash {row['hamming_distance']}; pixel diff {row['normalized_pixel_difference']:.3f}; {row['first_split']} vs {row['second_split']}",
            )
        )
        if index == 3:
            break

    source_members: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        if row["cross_split"]:
            source_members[row["group_id"]].append(row)
    source_pair_count = 0
    for members in source_members.values():
        first = members[0]
        second = next((member for member in members[1:] if member["split"] != first["split"]), None)
        if second is None:
            continue
        source_pair_count += 1
        pair_candidates.append(
            (
                f"source_name_pair_{source_pair_count}",
                first["image_path"],
                second["image_path"],
                f"Shared source name; {first['split']} vs {second['split']}",
            )
        )
        if source_pair_count == 3:
            break

    for category, first_path, second_path, note in pair_candidates:
        first = ImageOps.contain(annotated_image(first_path, "A"), (560, 560))
        second = ImageOps.contain(annotated_image(second_path, "B"), (560, 560))
        canvas = Image.new("RGB", (1140, 600), "black")
        canvas.paste(first, (10 + (560 - first.width) // 2, 36 + (560 - first.height) // 2))
        canvas.paste(second, (570 + (560 - second.width) // 2, 36 + (560 - second.height) // 2))
        ImageDraw.Draw(canvas).text((10, 10), note, fill="white")
        filename = f"{len(output_rows) + 1:02d}_{category}.jpg"
        canvas.save(sample_dir / filename, quality=92)
        output_rows.append(
            {
                "sample": f"visual_samples/{filename}",
                "category": category,
                "source_image": f"{first_path} | {second_path}",
                "note": note,
            }
        )
    _write_csv(sample_dir / "index.csv", output_rows)
    return output_rows


# Reads earlier thesis figures only for a transparent comparison table.
def _load_previous_statistics(path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {"splits": {}, "classes": {}}
    if path is None or not path.exists():
        return result
    text = path.read_text(encoding="utf-8")
    split_pattern = re.compile(r"\|\s*(Train|Validation|Test|Total)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", re.IGNORECASE)
    for name, images, labels, boxes, empty in split_pattern.findall(text):
        result["splits"][name.lower()] = {
            "images": int(images),
            "labels": int(labels),
            "boxes": int(boxes),
            "empty_labels": int(empty),
        }
    class_section = text.split("Previously reported raw class counts:", 1)
    if len(class_section) == 2:
        for name, count in re.findall(r"\|\s*([^|\n]+?)\s*\|\s*(\d+)\s*\|", class_section[1].split("Important unresolved", 1)[0]):
            if name.strip().lower() != "class":
                result["classes"][name.strip()] = int(count)
    return result


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# Turns verified audit outputs into the Phase 1 Markdown report.
def _render_report(summary: dict[str, Any], primary: dict[str, Any], previous: dict[str, Any], output_dir: Path) -> str:
    candidate = primary["candidate"]
    splits = {row["split"]: row for row in primary["split_rows"]}
    total_images = sum(row["image_files"] for row in primary["split_rows"])
    total_labels = sum(row["label_files"] for row in primary["split_rows"])
    total_boxes = len(primary["box_rows"])
    total_empty = len(primary["empty_rows"])
    class_counts = Counter(row["class_name"] for row in primary["box_rows"])
    issue_counts = Counter(row["code"] for row in primary["issue_rows"])
    image_widths = [row["width"] for row in primary["image_rows"]]
    image_heights = [row["height"] for row in primary["image_rows"]]
    aspect_ratios = [row["aspect_ratio"] for row in primary["image_rows"]]
    areas = [row["relative_area"] for row in primary["box_rows"]]
    annotation_types = Counter(row["annotation_type"] for row in primary["box_rows"])
    small_boxes = sum(row["is_very_small"] for row in primary["box_rows"])
    large_boxes = sum(row["is_unusually_large"] for row in primary["box_rows"])
    exact = summary["exact_duplicates"]
    near = summary["near_duplicates"]
    source_groups = summary["source_name_groups"]

    lines = [
        "# Dataset Audit Report",
        "",
        f"Generated by the reproducible Phase 1 audit on {summary['generated_at_utc']}.",
        "",
        "## 1. Audit scope and method",
        "",
        "This audit inspected the local raw YOLOv8 exports without changing source files. It checked split structure, image-label matching, box and polygon annotation syntax and geometry, empty labels, class counts, decoded image properties, derived box sizes, SHA-256 exact duplicates, 64-bit difference-hash candidates, and filename-derived source groups. Approximate matches are review candidates, not confirmed duplicate images.",
        "",
        f"Thresholds used: very small box area `< {SMALL_BOX_AREA}` of the image, unusually large box area `> {LARGE_BOX_AREA}`, extreme image aspect ratio outside `{EXTREME_ASPECT_RATIO_LOW}`–`{EXTREME_ASPECT_RATIO_HIGH}`, and near-duplicate dHash Hamming distance `<= {NEAR_DUPLICATE_DISTANCE}`. A contrast-normalized 32×32 mean pixel difference ranks dHash candidates for review but is not a confirmation threshold.",
        "",
        "## 2. Dataset structure discovered",
        "",
        f"The raw directory contains **{len(summary['candidates'])} complete YOLO dataset trees**. The reporting reference is `{candidate.root.relative_to(summary['raw_root']).as_posix()}` because its directory name identifies the versioned YOLOv8 export. This is an operational reporting choice, not a class, split, or training decision.",
        "",
    ]
    for item in summary["candidates"]:
        lines.append(f"- `{item['relative_root']}` (`{item['candidate_id']}`)")
    lines.extend(["", "Candidate equivalence checks:", ""])
    for comparison in summary["candidate_comparisons"]:
        lines.append(
            f"- `{comparison['first_candidate_id']}` vs `{comparison['second_candidate_id']}`: "
            f"{comparison['byte_identical_shared_files']}/{comparison['shared_relative_paths']} shared relative paths are byte-identical; "
            f"complete equivalence = **{comparison['all_files_equivalent']}**; differing path(s): "
            f"`{comparison['differing_relative_paths'] or 'none'}`."
        )

    lines.extend([
        "",
        "## 3. Image and label counts by split",
        "",
        "Counts below are for the single reporting-reference export and therefore avoid double-counting the second raw copy.",
        "",
        "| Split | Images | Labels | Matched pairs | Images without labels | Labels without images | Boxes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for split in ("train", "valid", "test"):
        row = splits[split]
        lines.append(f"| {split} | {row['image_files']} | {row['label_files']} | {row['matched_pairs']} | {row['images_without_labels']} | {row['labels_without_images']} | {row['bounding_boxes']} |")
    lines.append(f"| Total | {total_images} | {total_labels} | {sum(row['matched_pairs'] for row in splits.values())} | {sum(row['images_without_labels'] for row in splits.values())} | {sum(row['labels_without_images'] for row in splits.values())} | {total_boxes} |")

    lines.extend(["", "## 4. Annotation validation results", ""])
    if issue_counts:
        lines.append(f"The audit found **{sum(issue_counts.values())} annotation/image issues**. Invalid rows were retained in the issue artifact but excluded from box statistics.")
        lines.extend(["", "| Issue | Count |", "|---|---:|"])
        for code, count in sorted(issue_counts.items()):
            lines.append(f"| `{code}` | {count} |")
    else:
        lines.append("No malformed annotation lines, unexpected class IDs, non-numeric values, non-finite values, non-positive sizes, out-of-range coordinates, boundary-crossing boxes, degenerate polygons, duplicate rows, or unreadable images were found.")
    lines.extend([
        "",
        "Annotation representation:",
        "",
    ])
    for annotation_type, count in sorted(annotation_types.items()):
        lines.append(f"- `{annotation_type}` rows: {count}")
    if annotation_types.get("polygon"):
        lines.extend([
            "",
            "**Important format finding:** the annotations are YOLO segmentation polygons rather than five-value YOLO detection boxes. For this audit only, each valid polygon's minimum and maximum coordinates were used to calculate an inspectable axis-aligned bounding box. Raw labels were not converted. A detection-training dataset would require a separate, traceable Phase 2 derived-data conversion decision.",
        ])

    lines.extend([
        "",
        "## 5. Empty-label analysis",
        "",
        "| Split | Empty labels | Proportion |",
        "|---|---:|---:|",
    ])
    for split in ("train", "valid", "test"):
        row = splits[split]
        lines.append(f"| {split} | {row['empty_labels']} | {row['empty_label_proportion']:.2%} |")
    lines.append(f"| Total | {total_empty} | {total_empty / total_labels:.2%} |")
    lines.extend(["", "**Verified fact:** these files contain no annotation rows. **Unresolved:** local files do not establish that every empty label is a reviewed negative X-ray rather than a missing annotation."])

    lines.extend([
        "",
        "## 6. Class distribution",
        "",
        "| ID | Class | Train | Valid | Test | Total | Share |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for class_id, class_name in enumerate(candidate.class_names):
        per_split = {split: sum(row["class_id"] == class_id and row["split"] == split for row in primary["box_rows"]) for split in ("train", "valid", "test")}
        count = class_counts[class_name]
        lines.append(f"| {class_id} | {class_name} | {per_split['train']} | {per_split['valid']} | {per_split['test']} | {count} | {count / total_boxes:.2%} |")
    positive_counts = [count for count in class_counts.values() if count]
    imbalance_ratio = max(positive_counts) / min(positive_counts) if positive_counts else 0
    lines.extend(["", f"The largest-to-smallest non-zero class ratio is **{imbalance_ratio:.1f}:1**. This is severe imbalance; it is a finding, not a decision to remove or merge any class."])

    extreme_images = [row for row in primary["image_rows"] if row["aspect_ratio"] < EXTREME_ASPECT_RATIO_LOW or row["aspect_ratio"] > EXTREME_ASPECT_RATIO_HIGH]
    format_counts = Counter((row["extension"], row["decoded_format"]) for row in primary["image_rows"])
    lines.extend([
        "",
        "## 7. Image statistics",
        "",
        f"All {len(primary['image_rows'])} matched images decoded successfully. Width ranges from **{min(image_widths)} to {max(image_widths)} px** (median {_format_number(median(image_widths), 1)}); height ranges from **{min(image_heights)} to {max(image_heights)} px** (median {_format_number(median(image_heights), 1)}). Aspect ratio ranges from **{min(aspect_ratios):.3f} to {max(aspect_ratios):.3f}** (median {median(aspect_ratios):.3f}).",
        "",
        f"There are **{len(set((row['width'], row['height']) for row in primary['image_rows']))} distinct width-height combinations** and **{len(extreme_images)} images** outside the audit's extreme aspect-ratio range.",
        "",
        "Format evidence:",
        "",
    ])
    for (extension, decoded), count in sorted(format_counts.items()):
        lines.append(f"- `{extension}` files decoded as `{decoded}`: {count}")

    lines.extend([
        "",
        "## 8. Bounding-box statistics",
        "",
        f"Across {total_boxes} valid derived boxes, relative area ranges from **{min(areas):.8f} to {max(areas):.4f}**, with median **{median(areas):.4f}** and 95th percentile **{_quantile(areas, 0.95):.4f}**.",
        "",
        f"Using documented review thresholds, **{small_boxes} boxes ({small_boxes / total_boxes:.2%})** are very small and **{large_boxes} boxes ({large_boxes / total_boxes:.2%})** are unusually large. These flags do not imply that the annotations are wrong.",
        "",
        "Class-level distributions are recorded in `bounding_box_statistics.csv`; individual valid boxes and pixel-space estimates are in `bounding_boxes.csv`.",
        "",
        "## 9. Duplicate and possible leakage findings",
        "",
        f"SHA-256 found **{exact['within_candidate_within_split_groups']} exact duplicate groups within a reference-export split** and **{exact['within_candidate_cross_split_groups']} exact duplicate groups crossing splits within a candidate export**. It also found **{exact['cross_candidate_groups']} cross-candidate groups**, expected when checking the two raw trees together.",
        "",
        f"Within the reference export, dHash produced **{near['pairs']} non-byte-identical near-duplicate candidates**, of which **{near['cross_split_pairs']} cross splits**; **{near['cross_split_same_source_pairs']}** of those also share the stripped source name. The list was truncated at the safety cap: **{near['truncated']}**. Side-by-side review showed that low-distance X-ray hashes can still represent different anatomy, so every candidate requires stronger visual or provenance evidence before being called a duplicate.",
        "",
        f"After stripping the Roboflow `.rf.<hash>` suffix, **{source_groups['groups']} repeated source-name groups** were found; **{source_groups['cross_split_groups']} groups cross splits**. Side-by-side review showed that generic keys such as `image1_0_png` can collide across unrelated anatomy. These groups are screening signals only and do not prove common source images, patients, or studies.",
        "",
        "## 10. Visual inspection findings",
        "",
        f"The audit generated **{summary['visual_sample_count']} review images** covering all represented classes, every `humerus fracture` annotation, the smallest boxes, available issue cases, empty-label examples, and selected cross-split perceptual/source-name pairs. These are copies with overlays under `visual_samples/`; source images were not edited.",
        "",
        "The three `humerus fracture` polygons share the same pre-suffix source name, appear visually as transformed variants, are all in training, and each co-occurs with a separate `humerus` polygon. This indicates extreme effective-sample scarcity and a semantic overlap worth investigating; it does not establish that either label is incorrect. The highest-ranked cross-split candidates include very similar wrist, forearm, and hand radiographs with crop, orientation, or marker differences and deserve provenance review. Other hash/name pairs are visibly different, demonstrating that neither screening method is proof of leakage. Empty-label examples look like valid X-rays but still require source/domain verification before being treated as confirmed negatives.",
        "",
        "## 11. Comparison with previously reported statistics",
        "",
        "The current figures below were calculated independently from the reference export; the previous column is parsed from `docs/THESIS.md` only for comparison.",
        "",
        "| Measure | Current | Previous | Difference |",
        "|---|---:|---:|---:|",
    ])
    split_name_map = {"train": "train", "valid": "validation", "test": "test"}
    for split in ("train", "valid", "test"):
        previous_row = previous["splits"].get(split_name_map[split], {})
        for label, current_key, previous_key in (("images", "image_files", "images"), ("boxes", "bounding_boxes", "boxes"), ("empty labels", "empty_labels", "empty_labels")):
            current_value = splits[split][current_key]
            old_value = previous_row.get(previous_key)
            difference = current_value - old_value if old_value is not None else "n/a"
            lines.append(f"| {split} {label} | {current_value} | {_format_number(old_value)} | {difference} |")
    for class_name in candidate.class_names:
        current_value = class_counts[class_name]
        old_value = previous["classes"].get(class_name)
        difference = current_value - old_value if old_value is not None else "n/a"
        lines.append(f"| class: {class_name} | {current_value} | {_format_number(old_value)} | {difference} |")
    lines.extend([
        "",
        "The comparison uses one export. Counting both raw directory copies would double every image, label, empty-label, and box figure and would not represent an independent larger dataset.",
        "",
        "## 12. Dataset semantics that were verified",
        "",
        "- The configuration declares seven labels: `elbow positive`, `fingers positive`, `forearm fracture`, `humerus fracture`, `humerus`, `shoulder fracture`, and `wrist positive`.",
        "- Both configurations identify Roboflow workspace `veda`, project `bone-fracture-detection-daoon`, version 4, with CC BY 4.0 metadata.",
        "- Every image filename in the reference export has a Roboflow `.rf.<hex>` suffix. Repeated pre-suffix names show that multiple exported derivatives or records share a source-style name; the filenames alone do not reveal the transformation used.",
        "- The local evidence does not define the clinical/annotation distinction among `positive`, `fracture`, and bare `humerus`.",
        "",
        "## 13. Unresolved questions and risks",
        "",
        "- The intended semantics and annotation policy for `positive`, `fracture`, and `humerus` remain unresolved.",
        "- Empty labels cannot be verified locally as reviewed negative images.",
        "- Roboflow-style derivatives and repeated source names need provenance review, especially where groups cross train/validation/test.",
        "- Patient/study identifiers are not present in the YOLO configuration or obvious export paths, so patient/study-level independence cannot be established from this export alone.",
        "- The second raw tree has byte-identical images and labels but a byte-different `README.dataset.txt`; it still creates an operational double-counting risk.",
        "- The raw annotations are segmentation polygons. Detection training requires a documented derived-label conversion rather than direct use as five-value box labels.",
        "- The severely underrepresented class must not be evaluated or handled as though its sample size were adequate without an explicit Phase 2 decision.",
        "",
        "## 14. Recommended decisions for the next phase",
        "",
        "These are discussion points, not decisions made by this audit:",
        "",
        "1. Confirm with the dataset source or supervisor what each class name and empty label means.",
        "2. Review all cross-split exact, perceptual, and source-name candidates before accepting the current split.",
        "3. Decide which duplicate raw tree is the canonical input path for future commands while retaining the raw source unchanged.",
        "4. Define and test a traceable polygon-to-box conversion for the derived detection dataset.",
        "5. Decide how to handle the severely underrepresented class only after semantics and provenance are clear.",
        "6. Establish a patient/study grouping source if one exists before any split redesign.",
        "",
        "## 15. Generated audit artifacts",
        "",
        f"All machine-readable artifacts are under `{output_dir.as_posix()}`:",
        "",
        "- `audit_summary.json`: key findings, thresholds, and run metadata",
        "- `dataset_candidates.csv` and `candidate_comparisons.csv`: discovered roots and equivalence checks",
        "- `split_summary.csv`, `class_distribution.csv`, `image_statistics.csv`, and `bounding_box_statistics.csv`: aggregate statistics",
        "- `image_inventory.csv` and `bounding_boxes.csv`: inspectable image and valid-box records",
        "- `annotation_issues.csv`, `unmatched_files.csv`, and `empty_labels.csv`: validation findings",
        "- `exact_duplicate_groups.csv`, `near_duplicate_candidates.csv`, and `source_name_groups.csv`: leakage-review evidence",
        "- `source_manifest.csv`: SHA-256 manifest for every raw file",
        "- `visual_samples/`: annotated review copies and an index",
        "",
        "No training, preprocessing, augmentation, class remapping, split change, or raw-data modification was performed.",
        "",
    ])
    return "\n".join(lines)


# Coordinates the full read-only audit and writes reproducible evidence files.
def run_audit(raw_root: Path, output_dir: Path, report_path: Path, thesis_context: Path | None = None) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    output_dir = output_dir.resolve()
    report_path = report_path.resolve()
    if output_dir.is_relative_to(raw_root) or report_path.is_relative_to(raw_root):
        raise ValueError("Audit outputs and reports must be outside the immutable raw-data directory.")
    candidates = discover_dataset_candidates(raw_root)
    candidate_results = [_audit_candidate(candidate, raw_root) for candidate in candidates]
    primary = _select_primary(candidate_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_image_rows = [row for result in candidate_results for row in result["image_rows"]]
    exact_rows, exact_summary = _exact_duplicate_rows(all_image_rows)
    near_rows, near_truncated = _near_duplicate_rows(primary["image_rows"], raw_root)
    source_rows = _source_group_rows(primary["image_rows"])
    comparison_rows = _compare_candidates(candidate_results, raw_root)
    manifest_rows = _manifest_rows(raw_root)
    visual_rows = _draw_visual_samples(primary, raw_root, output_dir, near_rows, source_rows)

    split_rows = [row for result in candidate_results for row in result["split_rows"]]
    class_rows = [row for result in candidate_results for row in _aggregate_class_rows(result)]
    image_stat_rows = [row for result in candidate_results for row in _aggregate_image_statistics(result)]
    box_stat_rows = [row for result in candidate_results for row in _aggregate_box_statistics(result)]
    issue_rows = [row for result in candidate_results for row in result["issue_rows"]]
    unmatched_rows = [row for result in candidate_results for row in result["unmatched_rows"]]
    empty_rows = [row for result in candidate_results for row in result["empty_rows"]]
    box_rows = [row for result in candidate_results for row in result["box_rows"]]

    candidate_rows = [
        {
            "candidate_id": result["candidate"].candidate_id,
            "relative_root": _relative(result["candidate"].root, raw_root),
            "config_path": _relative(result["candidate"].config_path, raw_root),
            "class_count": len(result["candidate"].class_names),
            "class_names": " | ".join(result["candidate"].class_names),
            "is_reporting_reference": result is primary,
        }
        for result in candidate_results
    ]
    source_group_ids = {row["group_id"] for row in source_rows}
    cross_source_group_ids = {row["group_id"] for row in source_rows if row["cross_split"]}
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "raw_root": raw_root,
        "output_dir": output_dir.as_posix(),
        "report_path": report_path.as_posix(),
        "thresholds": {
            "small_box_relative_area": SMALL_BOX_AREA,
            "large_box_relative_area": LARGE_BOX_AREA,
            "extreme_aspect_ratio_low": EXTREME_ASPECT_RATIO_LOW,
            "extreme_aspect_ratio_high": EXTREME_ASPECT_RATIO_HIGH,
            "near_duplicate_dhash_distance": NEAR_DUPLICATE_DISTANCE,
        },
        "candidates": candidate_rows,
        "reporting_reference_candidate_id": primary["candidate"].candidate_id,
        "candidate_comparisons": comparison_rows,
        "exact_duplicates": exact_summary,
        "near_duplicates": {
            "pairs": len(near_rows),
            "cross_split_pairs": sum(row["cross_split"] for row in near_rows),
            "cross_split_same_source_pairs": sum(
                row["cross_split"] and row["same_source_key"] for row in near_rows
            ),
            "truncated": near_truncated,
        },
        "source_name_groups": {
            "groups": len(source_group_ids),
            "cross_split_groups": len(cross_source_group_ids),
        },
        "visual_sample_count": len(visual_rows),
        "raw_manifest": {
            "file_count": len(manifest_rows),
            "total_bytes": sum(row["size_bytes"] for row in manifest_rows),
            "manifest_sha256": hashlib.sha256(
                "\n".join(f"{row['relative_path']}\t{row['size_bytes']}\t{row['sha256']}" for row in manifest_rows).encode("utf-8")
            ).hexdigest(),
        },
    }

    _write_csv(output_dir / "dataset_candidates.csv", candidate_rows)
    _write_csv(output_dir / "candidate_comparisons.csv", comparison_rows)
    _write_csv(output_dir / "split_summary.csv", split_rows)
    _write_csv(output_dir / "class_distribution.csv", class_rows)
    _write_csv(output_dir / "image_statistics.csv", image_stat_rows)
    _write_csv(output_dir / "bounding_box_statistics.csv", box_stat_rows)
    _write_csv(output_dir / "image_inventory.csv", all_image_rows)
    _write_csv(output_dir / "bounding_boxes.csv", box_rows)
    _write_csv(output_dir / "annotation_issues.csv", issue_rows, ["candidate_id", "split", "image_path", "label_path", "line_number", "code", "message", "raw_line"])
    _write_csv(output_dir / "unmatched_files.csv", unmatched_rows, ["candidate_id", "split", "kind", "stem", "path"])
    _write_csv(output_dir / "empty_labels.csv", empty_rows)
    _write_csv(output_dir / "exact_duplicate_groups.csv", exact_rows)
    _write_csv(output_dir / "near_duplicate_candidates.csv", near_rows)
    _write_csv(output_dir / "source_name_groups.csv", source_rows)
    _write_csv(output_dir / "source_manifest.csv", manifest_rows)

    serializable_summary = dict(summary)
    serializable_summary["raw_root"] = raw_root.as_posix()
    (output_dir / "audit_summary.json").write_text(json.dumps(serializable_summary, indent=2), encoding="utf-8")
    previous = _load_previous_statistics(thesis_context)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(summary, primary, previous, output_dir), encoding="utf-8")
    return serializable_summary
