from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

from bone_fracture_audit.audit import _read_dataset_config, match_images_and_labels, sha256_file
from bone_fracture_audit.yolo import parse_yolo_label
from bone_fracture_pipeline.prepare_dataset import (
    CANONICAL_EXPECTATIONS,
    DatasetExpectations,
    SPLITS,
    SplitExpectation,
    fingerprint_dataset,
)


SINGLE_CLASS_NAMES = ("fracture",)
SINGLE_CLASS_EXPECTATIONS = DatasetExpectations(
    class_names=SINGLE_CLASS_NAMES,
    splits={
        split: SplitExpectation(
            images=expected.images,
            annotations=expected.annotations,
            empty_labels=expected.empty_labels,
            class_counts=(expected.annotations,),
        )
        for split, expected in CANONICAL_EXPECTATIONS.splits.items()
    },
)
SOURCE_CLAHE_FINGERPRINT = "ca8286b35d3d31f0b8074aa387a44ca6392178926c23bde61ea6d99be70aba5f"
DATA_YAML = (
    "path: .\n"
    "train: train/images\n"
    "val: valid/images\n"
    "test: test/images\n"
    "names:\n"
    "  0: fracture\n"
)


class SingleClassDatasetError(RuntimeError):
    """Raised when the six-to-one derived dataset does not preserve its source."""


def _relative_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _remap_label(source_bytes: bytes, source_class_count: int) -> tuple[bytes, int]:
    if not source_bytes:
        return b"", 0
    lines = source_bytes.splitlines(keepends=True)
    remapped: list[bytes] = []
    for line in lines:
        tokens = line.split()
        if len(tokens) != 5 or tokens[0] not in {str(index).encode("ascii") for index in range(source_class_count)}:
            raise SingleClassDatasetError("Source label is not a valid six-class detection row.")
        if line[1:2] != b" ":
            raise SingleClassDatasetError("Source label has an unexpected class-token separator.")
        # Keep every coordinate byte and line ending unchanged.
        remapped.append(b"0" + line[1:])
    return b"".join(remapped), len(lines)


# Independently compares every source and derived path, image hash, and label row.
def validate_single_class_dataset(
    source: Path,
    derived: Path,
    *,
    expected_source_fingerprint: str | None = SOURCE_CLAHE_FINGERPRINT,
    expected_derived_fingerprint: str | None = None,
    expectations: DatasetExpectations = CANONICAL_EXPECTATIONS,
) -> dict[str, object]:
    source, derived = source.resolve(), derived.resolve()
    # The source is a fixed six-class input, never a destination for this conversion.
    if source == derived or source in derived.parents or derived in source.parents:
        raise SingleClassDatasetError("Source and derived dataset paths must be separate.")
    if not source.is_dir() or not derived.is_dir():
        raise SingleClassDatasetError("Source and derived dataset directories must both exist.")
    if _read_dataset_config(source / "data.yaml") != expectations.class_names:
        raise SingleClassDatasetError("Source class mapping differs from the six approved classes.")
    if (derived / "data.yaml").read_bytes() != DATA_YAML.encode("utf-8"):
        raise SingleClassDatasetError("Derived data.yaml must declare only class 0: fracture and fixed split paths.")
    if _read_dataset_config(derived / "data.yaml") != SINGLE_CLASS_NAMES:
        raise SingleClassDatasetError("Derived dataset has an invalid class mapping.")

    source_fingerprint = fingerprint_dataset(source)
    if expected_source_fingerprint and source_fingerprint.digest != expected_source_fingerprint:
        raise SingleClassDatasetError("The CLAHE source fingerprint differs from Experiment D.")
    source_paths = _relative_files(source)
    derived_paths = _relative_files(derived)
    if source_paths != derived_paths:
        raise SingleClassDatasetError("Derived dataset has missing or extra relative files.")

    split_results: dict[str, dict[str, int]] = {}
    image_hash_matches = 0
    label_matches = 0
    remapped_rows = 0
    empty_labels = 0
    # Held-out files are checked for integrity here; they are not decoded or evaluated.
    for split in SPLITS:
        expected = expectations.splits[split]
        source_pairs = match_images_and_labels(source / split / "images", source / split / "labels")
        derived_pairs = match_images_and_labels(derived / split / "images", derived / split / "labels")
        for matching in (source_pairs, derived_pairs):
            if any(matching[key] for key in (
                "images_without_labels", "labels_without_images", "ambiguous_image_stems", "ambiguous_label_stems"
            )):
                raise SingleClassDatasetError(f"Unmatched or ambiguous image/label files in {split}.")
        source_by_name = {image.name: (image, label) for image, label in source_pairs["pairs"]}
        derived_by_name = {image.name: (image, label) for image, label in derived_pairs["pairs"]}
        if set(source_by_name) != set(derived_by_name) or len(source_by_name) != expected.images:
            raise SingleClassDatasetError(f"Image identities or counts changed in {split}.")

        split_annotations = 0
        split_empty = 0
        for image_name, (source_image, source_label) in source_by_name.items():
            derived_image, derived_label = derived_by_name[image_name]
            if sha256_file(source_image) != sha256_file(derived_image):
                raise SingleClassDatasetError(f"Image content changed: {split}/{image_name}")
            image_hash_matches += 1
            source_parsed = parse_yolo_label(source_label, set(range(len(expectations.class_names))))
            derived_parsed = parse_yolo_label(derived_label, {0})
            if (source_parsed.issues or derived_parsed.issues or
                any(row.annotation_type != "box" for row in (*source_parsed.annotations, *derived_parsed.annotations))):
                raise SingleClassDatasetError(f"Invalid detection annotation: {split}/{source_label.name}")
            expected_bytes, row_count = _remap_label(source_label.read_bytes(), len(expectations.class_names))
            if derived_label.read_bytes() != expected_bytes or len(derived_parsed.annotations) != row_count:
                raise SingleClassDatasetError(f"A label changed beyond its class token: {split}/{source_label.name}")
            split_annotations += row_count
            split_empty += int(row_count == 0)
            label_matches += 1
        if (split_annotations, split_empty) != (expected.annotations, expected.empty_labels):
            raise SingleClassDatasetError(f"Annotation or empty-label counts changed in {split}.")
        split_results[split] = {
            "images": len(source_by_name), "labels": len(source_by_name),
            "annotations": split_annotations, "empty_labels": split_empty,
        }
        remapped_rows += split_annotations
        empty_labels += split_empty

    derived_fingerprint = fingerprint_dataset(derived)
    if expected_derived_fingerprint and derived_fingerprint.digest != expected_derived_fingerprint:
        raise SingleClassDatasetError("Derived dataset fingerprint differs from Experiment E's configuration.")
    # Detect concurrent source changes before accepting the lineage result.
    if fingerprint_dataset(source).digest != source_fingerprint.digest:
        raise SingleClassDatasetError("The CLAHE source changed during validation.")
    return {
        "valid": True,
        "source_dataset": source.as_posix(),
        "source_fingerprint": source_fingerprint.digest,
        "derived_dataset": derived.as_posix(),
        "derived_fingerprint": derived_fingerprint.digest,
        "fingerprint_file_count": derived_fingerprint.file_count,
        "class_names": list(SINGLE_CLASS_NAMES),
        "class_mapping": {str(index): 0 for index in range(len(expectations.class_names))},
        "images": image_hash_matches,
        "labels": label_matches,
        "annotations": remapped_rows,
        "empty_labels": empty_labels,
        "image_hash_matches": image_hash_matches,
        "token_exact_label_matches": label_matches,
        "splits": split_results,
        "test_use": "file integrity only; no test training, inference, or visual review",
    }


def prepare_single_class_dataset(
    source: Path,
    output: Path,
    *,
    expected_source_fingerprint: str | None = SOURCE_CLAHE_FINGERPRINT,
    expectations: DatasetExpectations = CANONICAL_EXPECTATIONS,
) -> dict[str, object]:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise SingleClassDatasetError("Source and output paths must be separate.")
    if output.exists():
        raise SingleClassDatasetError(f"Derived dataset already exists: {output}")
    if _read_dataset_config(source / "data.yaml") != expectations.class_names:
        raise SingleClassDatasetError("Source class mapping differs from the approved six classes.")
    before = fingerprint_dataset(source)
    if expected_source_fingerprint and before.digest != expected_source_fingerprint:
        raise SingleClassDatasetError("The CLAHE source fingerprint differs from Experiment D.")

    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a fully validated build; an interrupted build cleans up its temporary tree.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}_building_", dir=output.parent) as directory:
        build = Path(directory)
        for split in SPLITS:
            image_source = source / split / "images"
            label_source = source / split / "labels"
            image_output = build / split / "images"
            label_output = build / split / "labels"
            image_output.mkdir(parents=True)
            label_output.mkdir(parents=True)
            for image in image_source.iterdir():
                if not image.is_file():
                    raise SingleClassDatasetError(f"Unexpected source image entry: {image}")
                shutil.copy2(image, image_output / image.name)
            for label in label_source.iterdir():
                if not label.is_file() or label.suffix.lower() != ".txt":
                    raise SingleClassDatasetError(f"Unexpected source label entry: {label}")
                parsed = parse_yolo_label(label, set(range(len(expectations.class_names))))
                if parsed.issues or any(row.annotation_type != "box" for row in parsed.annotations):
                    raise SingleClassDatasetError(f"Invalid source detection label: {label}")
                remapped, count = _remap_label(label.read_bytes(), len(expectations.class_names))
                if count != len(parsed.annotations):
                    raise SingleClassDatasetError(f"Source label row count differs from parser: {label}")
                (label_output / label.name).write_bytes(remapped)
        (build / "data.yaml").write_text(DATA_YAML, encoding="utf-8", newline="\n")
        result = validate_single_class_dataset(
            source, build, expected_source_fingerprint=expected_source_fingerprint, expectations=expectations
        )
        if output.exists():
            raise SingleClassDatasetError(f"Derived dataset appeared during preparation: {output}")
        build.rename(output)
    result["derived_dataset"] = output.as_posix()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or validate Experiment E's single-class CLAHE dataset.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = arguments.project_root.resolve()
    source = root / "data/prepared/v3_detection_clahe"
    output = root / "data/prepared/v3_detection_clahe_single_class"
    try:
        if arguments.validate_only:
            result = validate_single_class_dataset(source, output)
        else:
            result = prepare_single_class_dataset(source, output)
    except SingleClassDatasetError as error:
        print(f"Experiment E dataset validation failed: {error}")
        return 1
    result["checked_at_utc"] = datetime.now(UTC).isoformat()
    evidence = root / "outputs/phase3e/single_class/preparation_validation.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Experiment E dataset validated: {result['derived_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
