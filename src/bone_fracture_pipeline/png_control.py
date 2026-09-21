from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from bone_fracture_audit.audit import sha256_file
from bone_fracture_pipeline.prepare_dataset import (
    CANONICAL_EXPECTATIONS,
    DatasetExpectations,
    fingerprint_dataset,
)
from bone_fracture_pipeline.preprocess_dataset import (
    APPROVED_SOURCE_FINGERPRINT,
    DetectionPair,
    _display_path,
    _finalize_directories,
    _inspect_source_dataset,
    _prepare_build_locations,
    _remove_build_directory,
    _repository_commit,
    _write_csv,
    _write_data_yaml,
    _write_json,
)
from bone_fracture_pipeline.preprocessing_core import (
    PNG_COMPRESSION,
    PNG_SIGNATURE,
    PreprocessingError,
    decode_without_orientation_transform,
)


TOOL_VERSION = "1.0.0"


@dataclass(frozen=True)
class ControlImageRecord:
    split: str
    source_filename: str
    control_filename: str
    width: int
    height: int
    channels: int
    dtype: str
    exif_orientation: int
    source_image_sha256: str
    control_image_sha256: str
    decoded_pixel_sha256: str
    source_label_sha256: str
    control_label_sha256: str
    annotation_count: int

    @property
    def sample_id(self) -> str:
        return f"{self.split}/{Path(self.source_filename).stem}"


# Creates the format-normalized control without changing the decoded image matrix.
def build_png_control_dataset(
    source: Path,
    output: Path,
    artifacts: Path,
    canonical_summary: Path,
    *,
    overwrite: bool = False,
    expectations: DatasetExpectations = CANONICAL_EXPECTATIONS,
    expected_source_fingerprint: str = APPROVED_SOURCE_FINGERPRINT,
    project_root: Path | None = None,
) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    artifacts = artifacts.resolve()
    canonical_summary = canonical_summary.resolve()
    project_root = (project_root or Path.cwd()).resolve()
    _validate_locations(source, output, artifacts, canonical_summary)

    for path in (output, artifacts, canonical_summary):
        if path.exists() and not overwrite:
            raise PreprocessingError(f"Output already exists: {path}. Use --overwrite to rebuild safely.")

    # The approved fingerprint guard prevents accidental conversion of a drifted base dataset.
    source_before = fingerprint_dataset(source)
    if source_before.digest != expected_source_fingerprint:
        raise PreprocessingError(
            "Approved detection dataset fingerprint changed: "
            f"{source_before.digest} != {expected_source_fingerprint}"
        )
    pairs, split_summary = _inspect_source_dataset(source, expectations)

    output_build = output.parent / f".{output.name}_building"
    artifacts_build = artifacts.parent / f".{artifacts.name}_building"
    canonical_build = canonical_summary.with_name(f".{canonical_summary.name}.building")
    _prepare_build_locations((output_build, artifacts_build), (output, artifacts))
    _remove_staged_file(canonical_build)
    canonical_summary.parent.mkdir(parents=True, exist_ok=True)

    cv2.setNumThreads(1)
    try:
        output_build.mkdir(parents=True)
        artifacts_build.mkdir(parents=True)

        records = _build_control_files(pairs, output_build)
        _write_data_yaml(output_build / "data.yaml", expectations.class_names)
        validation = _validate_control_dataset(output_build, records, expectations)
        output_fingerprint = fingerprint_dataset(output_build)

        # A complete second build detects nondeterministic image encoding or traversal behavior.
        reproduction = _verify_reproducibility(
            output.parent,
            pairs,
            expectations,
            output_fingerprint.digest,
        )
        source_after = fingerprint_dataset(source)
        if source_after.digest != source_before.digest:
            raise PreprocessingError("Approved detection dataset changed during PNG conversion.")

        summary = _build_summary(
            source,
            output,
            artifacts,
            source_before.digest,
            source_after.digest,
            output_fingerprint.digest,
            output_fingerprint.file_count,
            split_summary,
            validation,
            reproduction,
            records,
            expectations,
            project_root,
        )
        _write_csv(artifacts_build / "png_control_manifest.csv", _manifest_rows(records))
        _write_json(artifacts_build / "png_control_summary.json", summary)
        _write_json(canonical_build, summary)

        _finalize_directories(
            ((output_build, output), (artifacts_build, artifacts)),
            overwrite=overwrite,
        )
        os.replace(canonical_build, canonical_summary)
        return summary
    except Exception:
        for build in (output_build, artifacts_build):
            _remove_build_directory(build)
        _remove_staged_file(canonical_build)
        raise


# Builds every output from the same safe decoder used by the CLAHE condition.
def _build_control_files(
    pairs: Sequence[DetectionPair],
    output: Path,
) -> list[ControlImageRecord]:
    for split in ("train", "valid", "test"):
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)

    records: list[ControlImageRecord] = []
    for pair in pairs:
        decoded_source, orientation = decode_without_orientation_transform(pair.image_path)
        if decoded_source.dtype != np.uint8 or decoded_source.ndim != 3 or decoded_source.shape[2] != 3:
            raise PreprocessingError(
                "PNG control requires the approved three-channel uint8 source profile: "
                f"{pair.image_path} has {decoded_source.dtype} {decoded_source.shape!r}."
            )

        height, width = decoded_source.shape[:2]
        control_image = output / pair.split / "images" / f"{pair.image_path.stem}.png"
        control_label = output / pair.split / "labels" / pair.label_path.name
        if not cv2.imwrite(
            str(control_image),
            decoded_source,
            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
        ):
            raise PreprocessingError(f"OpenCV could not write {control_image}.")
        shutil.copyfile(pair.label_path, control_label)

        decoded_control = cv2.imread(str(control_image), cv2.IMREAD_UNCHANGED)
        if decoded_control is None:
            raise PreprocessingError(f"PNG control image cannot be decoded: {control_image}")
        if (
            decoded_control.shape != decoded_source.shape
            or decoded_control.dtype != decoded_source.dtype
            or not np.array_equal(decoded_control, decoded_source)
        ):
            raise PreprocessingError(f"Decoded pixels changed during PNG conversion: {pair.image_path}")
        if control_image.read_bytes()[:8] != PNG_SIGNATURE:
            raise PreprocessingError(f"Control image is not a PNG file: {control_image}")

        source_label_hash = sha256_file(pair.label_path)
        control_label_hash = sha256_file(control_label)
        if source_label_hash != control_label_hash:
            raise PreprocessingError(f"Label bytes changed for {pair.label_path}.")

        records.append(
            ControlImageRecord(
                split=pair.split,
                source_filename=pair.image_path.name,
                control_filename=control_image.name,
                width=width,
                height=height,
                channels=decoded_source.shape[2],
                dtype=str(decoded_source.dtype),
                exif_orientation=orientation,
                source_image_sha256=sha256_file(pair.image_path),
                control_image_sha256=sha256_file(control_image),
                decoded_pixel_sha256=hashlib.sha256(decoded_source.tobytes()).hexdigest(),
                source_label_sha256=source_label_hash,
                control_label_sha256=control_label_hash,
                annotation_count=len(pair.parsed_label.annotations),
            )
        )
    return records


# Re-decodes the finished tree so validation does not trust only the write-time checks.
def _validate_control_dataset(
    output: Path,
    records: Sequence[ControlImageRecord],
    expectations: DatasetExpectations,
) -> dict[str, object]:
    by_sample = {(record.split, Path(record.control_filename).stem.casefold()): record for record in records}
    image_count = 0
    label_count = 0
    annotation_count = 0
    empty_count = 0

    for split, expected in expectations.splits.items():
        image_paths = sorted((output / split / "images").glob("*.png"), key=lambda path: path.name.casefold())
        label_paths = sorted((output / split / "labels").glob("*.txt"), key=lambda path: path.name.casefold())
        if len(image_paths) != expected.images or len(label_paths) != expected.images:
            raise PreprocessingError(f"PNG control split {split!r} has unexpected file counts.")
        if {path.stem.casefold() for path in image_paths} != {path.stem.casefold() for path in label_paths}:
            raise PreprocessingError(f"PNG control split {split!r} has unmatched stems.")

        split_annotations = 0
        split_empty = 0
        for image_path in image_paths:
            record = by_sample.get((split, image_path.stem.casefold()))
            if record is None:
                raise PreprocessingError(f"Control image is missing from the manifest: {image_path}")
            decoded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.shape != (record.height, record.width, record.channels):
                raise PreprocessingError(f"Control image geometry changed: {image_path}")
            if str(decoded.dtype) != record.dtype:
                raise PreprocessingError(f"Control image dtype changed: {image_path}")
            if hashlib.sha256(decoded.tobytes()).hexdigest() != record.decoded_pixel_sha256:
                raise PreprocessingError(f"Control image pixels changed after creation: {image_path}")
            if image_path.read_bytes()[:8] != PNG_SIGNATURE:
                raise PreprocessingError(f"Control image is not a valid PNG: {image_path}")

            label_path = output / split / "labels" / f"{image_path.stem}.txt"
            if sha256_file(label_path) != record.source_label_sha256:
                raise PreprocessingError(f"Control label differs from its source: {label_path}")
            split_annotations += record.annotation_count
            split_empty += int(record.annotation_count == 0)

        if split_annotations != expected.annotations or split_empty != expected.empty_labels:
            raise PreprocessingError(f"PNG control split {split!r} changed annotation counts.")
        image_count += len(image_paths)
        label_count += len(label_paths)
        annotation_count += split_annotations
        empty_count += split_empty

    if (
        image_count != expectations.images
        or label_count != expectations.images
        or annotation_count != expectations.annotations
        or empty_count != expectations.empty_labels
    ):
        raise PreprocessingError("PNG control totals do not match approved expectations.")
    return {
        "images": image_count,
        "labels": label_count,
        "annotations": annotation_count,
        "empty_labels": empty_count,
        "decoded_pixel_equal_images": image_count,
        "pixel_mismatches": 0,
        "dimension_mismatches": 0,
        "channel_mismatches": 0,
        "dtype_mismatches": 0,
        "label_hash_mismatches": 0,
        "decode_failures": 0,
        "non_png_images": 0,
    }


def _verify_reproducibility(
    temporary_parent: Path,
    pairs: Sequence[DetectionPair],
    expectations: DatasetExpectations,
    expected_fingerprint: str,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="phase2e-png-reproduction-", dir=temporary_parent) as directory:
        reproduced = Path(directory) / "dataset"
        reproduced.mkdir()
        records = _build_control_files(pairs, reproduced)
        _write_data_yaml(reproduced / "data.yaml", expectations.class_names)
        _validate_control_dataset(reproduced, records, expectations)
        fingerprint = fingerprint_dataset(reproduced)
        if fingerprint.digest != expected_fingerprint:
            raise PreprocessingError("Independent PNG control rebuild produced a different fingerprint.")
    return {
        "success": True,
        "first_build_fingerprint": expected_fingerprint,
        "reproduced_fingerprint": fingerprint.digest,
        "fingerprints_identical": True,
        "files_compared": fingerprint.file_count,
        "temporary_rebuild_removed": True,
    }


def _manifest_rows(records: Sequence[ControlImageRecord]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": record.sample_id,
            "split": record.split,
            "source_filename": record.source_filename,
            "control_filename": record.control_filename,
            "width": record.width,
            "height": record.height,
            "channels": record.channels,
            "dtype": record.dtype,
            "exif_orientation": record.exif_orientation,
            "source_image_sha256": record.source_image_sha256,
            "control_image_sha256": record.control_image_sha256,
            "decoded_pixel_sha256": record.decoded_pixel_sha256,
            "source_label_sha256": record.source_label_sha256,
            "control_label_sha256": record.control_label_sha256,
            "label_sha256_equal": record.source_label_sha256 == record.control_label_sha256,
            "annotation_count": record.annotation_count,
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
    split_summary: Mapping[str, object],
    validation: Mapping[str, object],
    reproduction: Mapping[str, object],
    records: Sequence[ControlImageRecord],
    expectations: DatasetExpectations,
    project_root: Path,
) -> dict[str, object]:
    return {
        "tool_version": TOOL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit_sha": _repository_commit(project_root),
        "source_dataset_path": _display_path(source, project_root),
        "control_dataset_path": _display_path(output, project_root),
        "phase2e_png_output_path": _display_path(artifacts, project_root),
        "source_dataset_fingerprint_before": source_before,
        "source_dataset_fingerprint_after": source_after,
        "source_unchanged": source_before == source_after,
        "control_dataset_fingerprint": output_fingerprint,
        "control_dataset_file_count": output_file_count,
        "conversion": {
            "operation": "decode approved JPEG and encode the identical pixel matrix as lossless PNG",
            "png_compression": PNG_COMPRESSION,
            "pixel_transformations": [],
            "geometry_transformations": [],
            "opencv_version": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
        },
        "source_profile": {
            "images": len(records),
            "suffixes": sorted({Path(record.source_filename).suffix.lower() for record in records}),
            "dtypes": sorted({record.dtype for record in records}),
            "channels": sorted({record.channels for record in records}),
            "exif_orientations": sorted({record.exif_orientation for record in records}),
        },
        "class_mapping": {str(index): name for index, name in enumerate(expectations.class_names)},
        "source_splits": split_summary,
        "validation": dict(validation),
        "reproducibility": dict(reproduction),
        "stable_sample_identity": "<split>/<source stem>",
        "automated_validation_passed": True,
        "official_training_performed": False,
    }


def _validate_locations(source: Path, output: Path, artifacts: Path, canonical_summary: Path) -> None:
    if not source.is_dir():
        raise PreprocessingError(f"Approved source dataset does not exist: {source}")
    for label, path in (("PNG control", output), ("artifacts", artifacts), ("canonical evidence", canonical_summary)):
        if path == source or path.is_relative_to(source):
            raise PreprocessingError(f"{label} must not be inside the approved source dataset: {path}")
    if output == artifacts or output.is_relative_to(artifacts) or artifacts.is_relative_to(output):
        raise PreprocessingError("PNG control data and generated artifacts must be separate.")


def _remove_staged_file(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file() or not path.name.startswith(".") or not path.name.endswith(".building"):
        raise PreprocessingError(f"Refusing to remove unsafe staged file: {path}")
    path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the pixel-preserving PNG control dataset for Experiments A and C."
    )
    parser.add_argument("--source", type=Path, default=Path("data/prepared/v3_detection"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared/v3_detection_png"))
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/phase2/phase2e/png_control"))
    parser.add_argument(
        "--canonical-summary",
        type=Path,
        default=Path("docs/evidence/phase2e/png_control_summary.json"),
    )
    parser.add_argument("--overwrite", action="store_true", help="Safely rebuild and replace existing outputs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = build_png_control_dataset(
            arguments.source,
            arguments.output,
            arguments.artifacts,
            arguments.canonical_summary,
            overwrite=arguments.overwrite,
        )
    except PreprocessingError as error:
        print(f"PNG control preparation failed: {error}")
        return 1
    print("PNG control preparation completed without training.")
    print(f"Control dataset: {summary['control_dataset_path']}")
    print(f"Dataset fingerprint: {summary['control_dataset_fingerprint']}")
    print(f"Pixel-equal images: {summary['validation']['decoded_pixel_equal_images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
