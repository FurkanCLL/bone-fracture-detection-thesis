from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from bone_fracture_audit.audit import sha256_file
from bone_fracture_audit.yolo import Annotation
from bone_fracture_pipeline.prepare_dataset import (
    DatasetExpectations,
    SplitExpectation,
    fingerprint_dataset,
)
from bone_fracture_pipeline.preprocess_dataset import (
    PNG_SIGNATURE,
    PreprocessingError,
    ProcessedImageRecord,
    channels_are_identical,
    preprocess_dataset,
    preprocess_image_array,
    select_visual_review,
    validate_processed_array,
)


TEST_CLASSES = ("class zero", "class one")
TEST_EXPECTATIONS = DatasetExpectations(
    class_names=TEST_CLASSES,
    splits={
        "train": SplitExpectation(2, 1, 1, (1, 0)),
        "valid": SplitExpectation(1, 1, 0, (0, 1)),
        "test": SplitExpectation(1, 0, 1, (0, 0)),
    },
)


class PreprocessingCoreTests(unittest.TestCase):
    def test_clahe_output_is_deterministic_and_preserves_geometry(self) -> None:
        source = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)

        original_first, processed_first = preprocess_image_array(source)
        original_second, processed_second = preprocess_image_array(source.copy())

        self.assertTrue(np.array_equal(original_first, original_second))
        self.assertTrue(np.array_equal(processed_first, processed_second))
        self.assertEqual(processed_first.shape, (12, 16, 3))
        self.assertEqual(processed_first.dtype, np.uint8)
        self.assertTrue(channels_are_identical(processed_first))

    def test_uint16_standardization_is_fixed_and_not_data_dependent(self) -> None:
        source = np.array([[0, 32768, 65535]], dtype=np.uint16)

        grayscale, processed = preprocess_image_array(source)

        self.assertEqual(grayscale.tolist(), [[0, 128, 255]])
        self.assertTrue(channels_are_identical(processed))

    def test_geometry_mismatch_is_rejected(self) -> None:
        processed = np.zeros((8, 10, 3), dtype=np.uint8)

        with self.assertRaisesRegex(PreprocessingError, "geometry"):
            validate_processed_array(processed, expected_width=11, expected_height=8)

    def test_visual_selection_is_reproducible_and_excludes_test(self) -> None:
        records = [
            _record("train", "a.jpg", 20.0, 5.0, (0,)),
            _record("valid", "b.jpg", 220.0, 25.0, (1,)),
            _record("test", "c.jpg", 2.0, 1.0, (0, 1)),
            _record("train", "d.jpg", 100.0, 2.0, (0, 1), annotations=2),
        ]

        first = select_visual_review(records, class_count=2, per_category=1)
        second = select_visual_review(list(reversed(records)), class_count=2, per_category=1)

        first_keys = [(record.image_key, reasons) for record, reasons in first]
        second_keys = [(record.image_key, reasons) for record, reasons in second]
        self.assertEqual(first_keys, second_keys)
        self.assertNotIn("test", {record.split for record, _ in first})
        covered_classes = {class_id for record, _ in first for class_id in record.class_ids}
        self.assertEqual(covered_classes, {0, 1})


class PreprocessingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "data" / "prepared" / "v3_detection"
        self.output = self.root / "data" / "prepared" / "v3_detection_clahe"
        self.artifacts = self.root / "outputs" / "phase2" / "phase2d"
        self.canonical = self.root / "docs" / "evidence" / "phase2d"
        self._create_source()
        self.source_fingerprint = fingerprint_dataset(self.source).digest

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_build_preserves_labels_geometry_and_source(self) -> None:
        source_before = fingerprint_dataset(self.source).digest

        summary = self._run()

        self.assertEqual(summary["validation"]["images"], 4)
        self.assertEqual(summary["validation"]["annotations"], 2)
        self.assertEqual(summary["validation"]["empty_labels"], 2)
        self.assertEqual(summary["label_integrity"]["hash_mismatches"], 0)
        self.assertTrue(summary["reproducibility"]["fingerprints_identical"])
        self.assertEqual(fingerprint_dataset(self.source).digest, source_before)

        for split in ("train", "valid", "test"):
            source_labels = sorted((self.source / split / "labels").glob("*.txt"))
            for source_label in source_labels:
                copied_label = self.output / split / "labels" / source_label.name
                self.assertEqual(sha256_file(copied_label), sha256_file(source_label))
            for processed_image in (self.output / split / "images").glob("*.png"):
                self.assertEqual(processed_image.read_bytes()[:8], PNG_SIGNATURE)
                decoded = cv2.imread(str(processed_image), cv2.IMREAD_UNCHANGED)
                self.assertIsNotNone(decoded)
                self.assertTrue(channels_are_identical(decoded))

        with (self.artifacts / "visual_review_index.csv").open(encoding="utf-8", newline="") as source:
            visual_rows = list(csv.DictReader(source))
        self.assertNotIn("test", {row["split"] for row in visual_rows})
        self.assertTrue((self.artifacts / "visual_review" / "contact_sheet.png").is_file())
        self.assertTrue((self.canonical / "preprocessing_summary.json").is_file())
        self.assertTrue((self.canonical / "intensity_summary.csv").is_file())

    def test_existing_outputs_require_explicit_overwrite(self) -> None:
        first = self._run()

        with self.assertRaisesRegex(PreprocessingError, "--overwrite"):
            self._run()

        rebuilt = self._run(overwrite=True)
        self.assertEqual(
            first["processed_dataset_fingerprint"],
            rebuilt["processed_dataset_fingerprint"],
        )

    def _run(self, *, overwrite: bool = False) -> dict[str, object]:
        return preprocess_dataset(
            self.source,
            self.output,
            self.artifacts,
            self.canonical,
            overwrite=overwrite,
            expectations=TEST_EXPECTATIONS,
            expected_source_fingerprint=self.source_fingerprint,
            project_root=self.root,
        )

    def _create_source(self) -> None:
        self.source.mkdir(parents=True)
        (self.source / "data.yaml").write_text(
            "path: .\n"
            "train: train/images\n"
            "val: valid/images\n"
            "test: test/images\n"
            "names:\n"
            "  0: class zero\n"
            "  1: class one\n",
            encoding="utf-8",
        )
        examples = {
            "train": [
                ("train-a.jpg", "0 0.5000000000 0.5000000000 0.2000000000 0.3000000000\n", 20),
                ("train-b.jpg", "", 110),
            ],
            "valid": [
                ("valid-a.jpg", "1 0.4000000000 0.4000000000 0.3000000000 0.2000000000\n", 210),
            ],
            "test": [("test-a.jpg", "", 70)],
        }
        for split, split_examples in examples.items():
            image_dir = self.source / split / "images"
            label_dir = self.source / split / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for index, (filename, label, level) in enumerate(split_examples):
                gradient = np.tile(np.arange(24, dtype=np.uint8), (18, 1))
                image = np.clip(level + gradient + index, 0, 255).astype(np.uint8)
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
                self.assertTrue(cv2.imwrite(str(image_dir / filename), image))
                (label_dir / f"{Path(filename).stem}.txt").write_text(label, encoding="utf-8")


def _record(
    split: str,
    filename: str,
    mean: float,
    standard_deviation: float,
    class_ids: tuple[int, ...],
    *,
    annotations: int = 1,
) -> ProcessedImageRecord:
    boxes = tuple(
        Annotation(
            line_number=index + 1,
            class_id=class_ids[index % len(class_ids)],
            annotation_type="box",
            x_center=0.5,
            y_center=0.5,
            width=0.1 + index * 0.01,
            height=0.1,
            polygon_points=(),
        )
        for index in range(annotations)
    )
    return ProcessedImageRecord(
        split=split,
        source_filename=filename,
        processed_filename=f"{Path(filename).stem}.png",
        width=100,
        height=80,
        source_suffix=".jpg",
        source_dtype="uint8",
        source_channels=3,
        exif_orientation=1,
        original_mean=mean,
        processed_mean=mean + 1,
        original_std=standard_deviation,
        processed_std=standard_deviation + 1,
        original_min=0,
        original_max=255,
        processed_min=0,
        processed_max=255,
        pixel_count=8000,
        original_sum=mean * 8000,
        original_square_sum=(standard_deviation**2 + mean**2) * 8000,
        processed_sum=(mean + 1) * 8000,
        processed_square_sum=((standard_deviation + 1) ** 2 + (mean + 1) ** 2) * 8000,
        original_zero_count=0,
        original_full_count=0,
        processed_zero_count=0,
        processed_full_count=0,
        source_image_sha256="a" * 64,
        processed_image_sha256="b" * 64,
        source_label_sha256="c" * 64,
        processed_label_sha256="c" * 64,
        annotations=boxes,
    )


if __name__ == "__main__":
    unittest.main()
