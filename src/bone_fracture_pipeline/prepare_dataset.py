from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from bone_fracture_audit.audit import (
    IMAGE_EXTENSIONS,
    _read_dataset_config,
    match_images_and_labels,
    sha256_file,
)
from bone_fracture_audit.yolo import Annotation, ParsedLabel, parse_yolo_label


TOOL_VERSION = "1.0.0"
SPLITS = ("train", "valid", "test")
CLASS_NAMES = (
    "elbow positive",
    "fingers positive",
    "forearm fracture",
    "humerus fracture",
    "shoulder fracture",
    "wrist positive",
)


class PreparationError(RuntimeError):
    """Raised when source drift or an unsafe preparation state is detected."""


@dataclass(frozen=True)
class SplitExpectation:
    images: int
    annotations: int
    empty_labels: int
    class_counts: tuple[int, ...]


@dataclass(frozen=True)
class DatasetExpectations:
    class_names: tuple[str, ...]
    splits: Mapping[str, SplitExpectation]

    @property
    def images(self) -> int:
        return sum(split.images for split in self.splits.values())

    @property
    def annotations(self) -> int:
        return sum(split.annotations for split in self.splits.values())

    @property
    def empty_labels(self) -> int:
        return sum(split.empty_labels for split in self.splits.values())

    @property
    def class_counts(self) -> tuple[int, ...]:
        return tuple(
            sum(split.class_counts[class_id] for split in self.splits.values())
            for class_id in range(len(self.class_names))
        )


CANONICAL_EXPECTATIONS = DatasetExpectations(
    class_names=CLASS_NAMES,
    splits={
        "train": SplitExpectation(1211, 698, 607, (113, 178, 107, 104, 120, 76)),
        "valid": SplitExpectation(348, 204, 175, (29, 48, 43, 36, 20, 28)),
        "test": SplitExpectation(169, 96, 86, (17, 27, 14, 15, 17, 6)),
    },
)


@dataclass(frozen=True)
class SourcePair:
    split: str
    image_path: Path
    label_path: Path
    parsed_label: ParsedLabel


@dataclass(frozen=True)
class SourceInventory:
    pairs: tuple[SourcePair, ...]
    split_summary: Mapping[str, dict[str, object]]
    class_counts: tuple[int, ...]


@dataclass(frozen=True)
class Fingerprint:
    digest: str
    file_hashes: Mapping[str, str]
    file_count: int


def polygon_to_box(annotation: Annotation) -> tuple[float, float, float, float]:
    if annotation.annotation_type != "polygon":
        raise PreparationError(
            f"Expected a polygon at source line {annotation.line_number}, "
            f"but found {annotation.annotation_type!r}."
        )
    if len(set(annotation.polygon_points)) < 3:
        raise PreparationError(f"Polygon at source line {annotation.line_number} is degenerate.")

    x_values = [point[0] for point in annotation.polygon_points]
    y_values = [point[1] for point in annotation.polygon_points]
    left, right = min(x_values), max(x_values)
    top, bottom = min(y_values), max(y_values)
    box = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
    _validate_box(box, f"source line {annotation.line_number}")
    return box


def serialize_detection_row(class_id: int, box: Sequence[float]) -> str:
    _validate_box(box, f"class {class_id}")
    return f"{class_id} " + " ".join(f"{value:.10f}" for value in box)


# Fingerprints all raw files using path-hash pairs in a stable order.
def fingerprint_dataset(source: Path) -> Fingerprint:
    source = source.resolve()
    if not source.is_dir():
        raise PreparationError(f"Source dataset directory does not exist: {source}")

    files = sorted(
        (path for path in source.rglob("*") if path.is_file()),
        key=lambda path: (path.relative_to(source).as_posix().casefold(), path.relative_to(source).as_posix()),
    )
    if not files:
        raise PreparationError(f"Source dataset contains no files: {source}")

    canonical_digest = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for path in files:
        relative_path = path.relative_to(source).as_posix()
        file_digest = sha256_file(path)
        file_hashes[relative_path] = file_digest
        canonical_digest.update(f"{relative_path}\t{file_digest}\n".encode("utf-8"))
    return Fingerprint(canonical_digest.hexdigest(), file_hashes, len(files))


def prepare_dataset(
    source: Path,
    output: Path,
    artifacts: Path,
    *,
    overwrite: bool = False,
    expectations: DatasetExpectations = CANONICAL_EXPECTATIONS,
    project_root: Path | None = None,
) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    artifacts = artifacts.resolve()
    project_root = (project_root or Path.cwd()).resolve()

    _validate_locations(source, output, artifacts)
    _validate_expectation_shape(expectations)
    if output.exists() and not overwrite:
        raise PreparationError(f"Prepared dataset already exists: {output}. Use --overwrite to rebuild it.")
    if artifacts.exists() and not overwrite:
        raise PreparationError(f"Traceability directory already exists: {artifacts}. Use --overwrite to rebuild it.")

    source_config = source / "data.yaml"
    if not source_config.is_file():
        raise PreparationError(f"Source data.yaml is missing: {source_config}")
    try:
        class_names = _read_dataset_config(source_config)
    except (OSError, SyntaxError, TypeError, ValueError) as error:
        raise PreparationError(f"Could not validate source data.yaml: {error}") from error
    if class_names != expectations.class_names:
        raise PreparationError(
            f"Unexpected class mapping. Expected {expectations.class_names!r}, found {class_names!r}."
        )

    source_before = fingerprint_dataset(source)
    inventory = _inspect_source(source, expectations)
    prepared_build = output.parent / f".{output.name}_building"
    artifacts_build = artifacts.parent / f".{artifacts.name}_building"
    _prepare_build_locations((prepared_build, artifacts_build), (output, artifacts))

    try:
        prepared_build.mkdir(parents=True)
        artifacts_build.mkdir(parents=True)
        manifest_rows, conversion_rows = _build_prepared_dataset(
            source,
            prepared_build,
            output,
            inventory,
            source_before.file_hashes,
            project_root,
        )
        _write_data_yaml(prepared_build / "data.yaml", class_names)
        _validate_prepared_dataset(prepared_build, inventory, expectations, manifest_rows)

        source_after = fingerprint_dataset(source)
        if source_after.digest != source_before.digest:
            raise PreparationError("Raw source contents changed during preparation; the build was cancelled.")

        summary = _build_summary(
            source,
            output,
            artifacts,
            source_config,
            source_before,
            inventory,
            class_names,
            project_root,
        )
        _write_csv(artifacts_build / "file_manifest.csv", manifest_rows)
        _write_csv(artifacts_build / "annotation_conversion.csv", conversion_rows)
        (artifacts_build / "preparation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        _finalize_directories(
            ((prepared_build, output), (artifacts_build, artifacts)), overwrite=overwrite
        )
        return summary
    except Exception:
        _remove_build_directory(prepared_build)
        _remove_build_directory(artifacts_build)
        raise


def _validate_locations(source: Path, output: Path, artifacts: Path) -> None:
    if not source.is_dir():
        raise PreparationError(f"Source dataset directory does not exist: {source}")
    raw_root = next((parent for parent in (source, *source.parents) if parent.name.casefold() == "raw"), source)
    for name, path in (("prepared output", output), ("traceability output", artifacts)):
        if path == raw_root or path.is_relative_to(raw_root):
            raise PreparationError(f"The {name} must not be inside the raw-data directory: {path}")
    if output == artifacts or output.is_relative_to(artifacts) or artifacts.is_relative_to(output):
        raise PreparationError("Prepared and traceability output directories must be separate.")


def _validate_expectation_shape(expectations: DatasetExpectations) -> None:
    if tuple(expectations.splits) != SPLITS:
        raise PreparationError(f"Expected split order {SPLITS!r}, found {tuple(expectations.splits)!r}.")
    class_count = len(expectations.class_names)
    if class_count == 0 or any(len(split.class_counts) != class_count for split in expectations.splits.values()):
        raise PreparationError("Dataset expectations contain inconsistent class counts.")


def _inspect_source(source: Path, expectations: DatasetExpectations) -> SourceInventory:
    pairs: list[SourcePair] = []
    total_class_counts: Counter[int] = Counter()
    split_summary: dict[str, dict[str, object]] = {}
    valid_class_ids = set(range(len(expectations.class_names)))

    for split in SPLITS:
        image_dir = source / split / "images"
        label_dir = source / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise PreparationError(f"Missing image or label directory for split {split!r}.")
        _reject_unexpected_entries(image_dir, IMAGE_EXTENSIONS)
        _reject_unexpected_entries(label_dir, {".txt"})

        matching = match_images_and_labels(image_dir, label_dir)
        problems = {
            key: matching[key]
            for key in (
                "images_without_labels",
                "labels_without_images",
                "ambiguous_image_stems",
                "ambiguous_label_stems",
            )
            if matching[key]
        }
        if problems:
            raise PreparationError(f"Unmatched or ambiguous files in split {split!r}: {problems}")

        split_class_counts: Counter[int] = Counter()
        split_annotations = 0
        split_empty = 0
        for image_path, label_path in sorted(
            matching["pairs"], key=lambda pair: (pair[0].name.casefold(), pair[0].name)
        ):
            parsed = parse_yolo_label(label_path, valid_class_ids)
            if parsed.issues:
                details = "; ".join(
                    f"line {issue.line_number} [{issue.code}] {issue.message}" for issue in parsed.issues
                )
                raise PreparationError(f"Invalid source label {label_path}: {details}")
            for annotation in parsed.annotations:
                if annotation.annotation_type != "polygon":
                    raise PreparationError(
                        f"Dataset drift in {label_path} line {annotation.line_number}: "
                        "expected a polygon, not a five-value detection row."
                    )
                polygon_to_box(annotation)
                split_class_counts[annotation.class_id] += 1
            split_annotations += len(parsed.annotations)
            split_empty += int(parsed.is_empty)
            pairs.append(SourcePair(split, image_path, label_path, parsed))

        expected = expectations.splits[split]
        actual_class_counts = tuple(split_class_counts[index] for index in range(len(expectations.class_names)))
        if len(matching["images"]) != expected.images or len(matching["labels"]) != expected.images:
            raise PreparationError(
                f"Unexpected {split} file count: {len(matching['images'])} images and "
                f"{len(matching['labels'])} labels; expected {expected.images} of each."
            )
        if split_annotations != expected.annotations or split_empty != expected.empty_labels:
            raise PreparationError(
                f"Unexpected {split} annotation state: {split_annotations} annotations and "
                f"{split_empty} empty labels; expected {expected.annotations} and {expected.empty_labels}."
            )
        if actual_class_counts != expected.class_counts:
            raise PreparationError(
                f"Unexpected {split} class counts: {actual_class_counts!r}; expected {expected.class_counts!r}."
            )

        total_class_counts.update(split_class_counts)
        split_summary[split] = {
            "images": expected.images,
            "labels": expected.images,
            "annotations": split_annotations,
            "empty_labels": split_empty,
            "per_class_annotations": _class_count_mapping(actual_class_counts, expectations.class_names),
        }

    actual_total_counts = tuple(total_class_counts[index] for index in range(len(expectations.class_names)))
    if len(pairs) != expectations.images or actual_total_counts != expectations.class_counts:
        raise PreparationError("Source totals do not match the approved canonical v3 expectations.")
    return SourceInventory(tuple(pairs), split_summary, actual_total_counts)


def _reject_unexpected_entries(directory: Path, allowed_extensions: set[str]) -> None:
    unexpected = [
        path.name
        for path in directory.iterdir()
        if not path.is_file() or path.suffix.lower() not in allowed_extensions
    ]
    if unexpected:
        raise PreparationError(f"Unexpected entries in {directory}: {sorted(unexpected, key=str.casefold)!r}")


def _build_prepared_dataset(
    source: Path,
    prepared_build: Path,
    final_output: Path,
    inventory: SourceInventory,
    source_hashes: Mapping[str, str],
    project_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    manifest_rows: list[dict[str, object]] = []
    conversion_rows: list[dict[str, object]] = []

    for split in SPLITS:
        (prepared_build / split / "images").mkdir(parents=True)
        (prepared_build / split / "labels").mkdir(parents=True)

    for pair in inventory.pairs:
        prepared_image = prepared_build / pair.split / "images" / pair.image_path.name
        prepared_label = prepared_build / pair.split / "labels" / pair.label_path.name
        final_image = final_output / pair.split / "images" / pair.image_path.name
        final_label = final_output / pair.split / "labels" / pair.label_path.name

        # copyfile creates an independent file without re-encoding image bytes.
        shutil.copyfile(pair.image_path, prepared_image)
        if prepared_image.is_symlink():
            raise PreparationError(f"Prepared image unexpectedly became a link: {prepared_image}")
        relative_source_image = pair.image_path.relative_to(source).as_posix()
        source_image_hash = source_hashes[relative_source_image]
        prepared_image_hash = sha256_file(prepared_image)
        if source_image_hash != prepared_image_hash:
            raise PreparationError(f"Image-copy hash mismatch for {pair.image_path}")

        detection_rows: list[str] = []
        for annotation in pair.parsed_label.annotations:
            box = polygon_to_box(annotation)
            detection_rows.append(serialize_detection_row(annotation.class_id, box))
            x_values = [point[0] for point in annotation.polygon_points]
            y_values = [point[1] for point in annotation.polygon_points]
            conversion_rows.append(
                {
                    "split": pair.split,
                    "filename": pair.image_path.name,
                    "source_line_number": annotation.line_number,
                    "class_id": annotation.class_id,
                    "source_annotation_representation": annotation.annotation_type,
                    "polygon_xmin": _format_coordinate(min(x_values)),
                    "polygon_ymin": _format_coordinate(min(y_values)),
                    "polygon_xmax": _format_coordinate(max(x_values)),
                    "polygon_ymax": _format_coordinate(max(y_values)),
                    "generated_x_center": _format_coordinate(box[0]),
                    "generated_y_center": _format_coordinate(box[1]),
                    "generated_width": _format_coordinate(box[2]),
                    "generated_height": _format_coordinate(box[3]),
                }
            )

        if detection_rows:
            prepared_label.write_text("\n".join(detection_rows) + "\n", encoding="utf-8", newline="\n")
        else:
            prepared_label.write_bytes(b"")

        manifest_rows.append(
            {
                "split": pair.split,
                "filename": pair.image_path.name,
                "source_image_path": _display_path(pair.image_path, project_root),
                "source_label_path": _display_path(pair.label_path, project_root),
                "prepared_image_path": _display_path(final_image, project_root),
                "prepared_label_path": _display_path(final_label, project_root),
                "source_image_sha256": source_image_hash,
                "prepared_image_sha256": prepared_image_hash,
                "source_label_sha256": source_hashes[pair.label_path.relative_to(source).as_posix()],
                "prepared_label_sha256": sha256_file(prepared_label),
                "source_annotation_count": len(pair.parsed_label.annotations),
                "prepared_annotation_count": len(detection_rows),
                "is_empty": pair.parsed_label.is_empty,
            }
        )

    return manifest_rows, conversion_rows


def _write_data_yaml(path: Path, class_names: Sequence[str]) -> None:
    lines = ["path: .", "train: train/images", "val: valid/images", "test: test/images", "names:"]
    lines.extend(f"  {class_id}: {name}" for class_id, name in enumerate(class_names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _validate_prepared_dataset(
    prepared: Path,
    inventory: SourceInventory,
    expectations: DatasetExpectations,
    manifest_rows: Sequence[Mapping[str, object]],
) -> None:
    if len(manifest_rows) != expectations.images:
        raise PreparationError("Prepared manifest row count does not match the expected image count.")

    prepared_class_counts: Counter[int] = Counter()
    prepared_annotations = 0
    prepared_empty = 0
    for split in SPLITS:
        matching = match_images_and_labels(prepared / split / "images", prepared / split / "labels")
        if any(
            matching[key]
            for key in (
                "images_without_labels",
                "labels_without_images",
                "ambiguous_image_stems",
                "ambiguous_label_stems",
            )
        ):
            raise PreparationError(f"Prepared split {split!r} contains unmatched or ambiguous files.")
        expected = expectations.splits[split]
        if len(matching["pairs"]) != expected.images:
            raise PreparationError(f"Prepared split {split!r} has an unexpected pair count.")

        split_annotations = 0
        split_empty = 0
        split_class_counts: Counter[int] = Counter()
        for _, label_path in matching["pairs"]:
            parsed = parse_yolo_label(label_path, set(range(len(expectations.class_names))))
            if parsed.issues or any(annotation.annotation_type != "box" for annotation in parsed.annotations):
                raise PreparationError(f"Generated label failed validation: {label_path}")
            split_annotations += len(parsed.annotations)
            split_empty += int(parsed.is_empty)
            split_class_counts.update(annotation.class_id for annotation in parsed.annotations)

        actual_classes = tuple(split_class_counts[index] for index in range(len(expectations.class_names)))
        if (
            split_annotations != expected.annotations
            or split_empty != expected.empty_labels
            or actual_classes != expected.class_counts
        ):
            raise PreparationError(f"Prepared split {split!r} does not preserve source annotation totals.")
        prepared_annotations += split_annotations
        prepared_empty += split_empty
        prepared_class_counts.update(split_class_counts)

    if prepared_annotations != expectations.annotations or prepared_empty != expectations.empty_labels:
        raise PreparationError("Prepared dataset totals do not match the approved expectations.")
    if tuple(prepared_class_counts[index] for index in range(len(expectations.class_names))) != inventory.class_counts:
        raise PreparationError("Prepared per-class counts differ from the source dataset.")
    if any(row["source_annotation_count"] != row["prepared_annotation_count"] for row in manifest_rows):
        raise PreparationError("At least one source annotation was lost or duplicated during conversion.")
    if any(row["source_image_sha256"] != row["prepared_image_sha256"] for row in manifest_rows):
        raise PreparationError("At least one prepared image differs from its source image.")


def _build_summary(
    source: Path,
    output: Path,
    artifacts: Path,
    source_config: Path,
    fingerprint: Fingerprint,
    inventory: SourceInventory,
    class_names: Sequence[str],
    project_root: Path,
) -> dict[str, object]:
    return {
        "source_dataset_path": _display_path(source, project_root),
        "prepared_dataset_path": _display_path(output, project_root),
        "traceability_artifacts_path": _display_path(artifacts, project_root),
        "source_dataset_fingerprint": fingerprint.digest,
        "source_fingerprint_file_count": fingerprint.file_count,
        "source_fingerprint_algorithm": (
            "SHA-256 of UTF-8 '<relative POSIX path>\\t<file SHA-256>\\n' records, "
            "sorted by case-folded relative path with the original path as a tie-breaker"
        ),
        "source_data_yaml_sha256": sha256_file(source_config),
        "images": len(inventory.pairs),
        "labels": len(inventory.pairs),
        "annotations": sum(len(pair.parsed_label.annotations) for pair in inventory.pairs),
        "empty_labels": sum(pair.parsed_label.is_empty for pair in inventory.pairs),
        "splits": inventory.split_summary,
        "per_class_annotations": _class_count_mapping(inventory.class_counts, class_names),
        "class_mapping": {str(index): name for index, name in enumerate(class_names)},
        "conversion_method": "minimum enclosing axis-aligned box from polygon coordinate extrema",
        "output_numeric_precision_decimal_places": 10,
        "tool_version": TOOL_VERSION,
        "repository_commit_sha": _repository_commit(project_root),
        "preparation_timestamp_utc": datetime.now(UTC).isoformat(),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise PreparationError(f"Cannot write an empty required artifact: {path}")
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_box(box: Sequence[float], context: str) -> None:
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise PreparationError(f"Non-finite or incomplete box for {context}.")
    x_center, y_center, width, height = box
    if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
        raise PreparationError(f"Box center is outside normalized bounds for {context}.")
    if width <= 0.0 or height <= 0.0 or width > 1.0 or height > 1.0:
        raise PreparationError(f"Box size is invalid for {context}.")
    tolerance = 1e-12
    if (
        x_center - width / 2 < -tolerance
        or x_center + width / 2 > 1.0 + tolerance
        or y_center - height / 2 < -tolerance
        or y_center + height / 2 > 1.0 + tolerance
    ):
        raise PreparationError(f"Box extends outside normalized bounds for {context}.")


def _format_coordinate(value: float) -> str:
    return f"{value:.10f}"


def _class_count_mapping(counts: Sequence[int], names: Sequence[str]) -> dict[str, int]:
    return {f"{class_id}: {name}": counts[class_id] for class_id, name in enumerate(names)}


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


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
        raise PreparationError(f"Refusing to remove an unsafe build path: {path}")
    shutil.rmtree(path)


# Swaps both the prepared data and its evidence as one rollback-capable transaction.
def _finalize_directories(staged_and_final: Sequence[tuple[Path, Path]], *, overwrite: bool) -> None:
    backups: list[tuple[Path, Path]] = []
    finalized: list[Path] = []
    try:
        for _, final in staged_and_final:
            if not final.exists():
                continue
            if not overwrite:
                raise PreparationError(f"Output appeared during preparation and was not replaced: {final}")
            backup = final.parent / f".{final.name}_backup"
            if backup.exists():
                raise PreparationError(f"Stale backup blocks safe replacement: {backup}")
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the canonical v3 polygon export as a deterministic YOLO detection dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/bone-fracture-detection/bone-fracture-detection-v3-yolov8"),
        help="Canonical raw v3 dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prepared/v3_detection"),
        help="Prepared detection dataset directory.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("outputs/phase2a/v3_detection"),
        help="Traceability artifact directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Safely rebuild and replace existing outputs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        summary = prepare_dataset(
            arguments.source,
            arguments.output,
            arguments.artifacts,
            overwrite=arguments.overwrite,
        )
    except PreparationError as error:
        raise SystemExit(f"Phase 2A preparation failed: {error}") from error

    print(
        f"Prepared {summary['images']} images and {summary['annotations']} detection annotations "
        f"({summary['empty_labels']} empty labels)."
    )
    print(f"Dataset: {summary['prepared_dataset_path']}")
    print(f"Traceability: {summary['traceability_artifacts_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
