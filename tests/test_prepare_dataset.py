from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bone_fracture_audit.yolo import Annotation
from bone_fracture_pipeline.prepare_dataset import (
    DatasetExpectations,
    PreparationError,
    SplitExpectation,
    fingerprint_dataset,
    polygon_to_box,
    prepare_dataset,
    serialize_detection_row,
)


TEST_CLASSES = ("class zero", "class one")
TEST_EXPECTATIONS = DatasetExpectations(
    class_names=TEST_CLASSES,
    splits={
        "train": SplitExpectation(1, 2, 0, (1, 1)),
        "valid": SplitExpectation(1, 0, 1, (0, 0)),
        "test": SplitExpectation(1, 1, 0, (0, 1)),
    },
)


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "data" / "raw" / "canonical-v3"
        self.output = self.root / "data" / "prepared" / "v3_detection"
        self.artifacts = self.root / "outputs" / "phase2a" / "v3_detection"
        self._create_source()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_source(self) -> None:
        self.source.mkdir(parents=True)
        (self.source / "data.yaml").write_text(
            "names: ['class zero', 'class one']\nnc: 2\n", encoding="utf-8"
        )
        labels = {
            "train": (
                "train-image.jpg",
                "0 0.1000000000 0.2000000000 0.5000000000 0.2000000000 0.4000000000 0.8000000000\n"
                "1 0.6000000000 0.1000000000 0.9000000000 0.4000000000 0.7000000000 0.9000000000\n",
            ),
            "valid": ("valid-image.jpg", ""),
            "test": (
                "test-image.jpg",
                "1 0.2000000000 0.3000000000 0.8000000000 0.2000000000 0.9000000000 0.7000000000 "
                "0.5000000000 0.9000000000 0.1000000000 0.6000000000\n",
            ),
        }
        for split, (image_name, label_text) in labels.items():
            image_dir = self.source / split / "images"
            label_dir = self.source / split / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (image_dir / image_name).write_bytes(f"independent-{split}-image".encode())
            (label_dir / f"{Path(image_name).stem}.txt").write_text(label_text, encoding="utf-8")

    def run_preparation(self, *, overwrite: bool = False) -> dict[str, object]:
        return prepare_dataset(
            self.source,
            self.output,
            self.artifacts,
            overwrite=overwrite,
            expectations=TEST_EXPECTATIONS,
            project_root=self.root,
        )

    def test_converts_irregular_polygon_and_serializes_exact_precision(self) -> None:
        annotation = Annotation(
            line_number=7,
            class_id=1,
            annotation_type="polygon",
            x_center=0.5,
            y_center=0.55,
            width=0.8,
            height=0.7,
            polygon_points=((0.2, 0.3), (0.8, 0.2), (0.9, 0.7), (0.5, 0.9), (0.1, 0.6)),
        )

        box = polygon_to_box(annotation)

        self.assertEqual(box, (0.5, 0.55, 0.8, 0.7))
        self.assertEqual(serialize_detection_row(1, box), "1 0.5000000000 0.5500000000 0.8000000000 0.7000000000")

    def test_build_preserves_images_empty_labels_classes_and_annotation_count(self) -> None:
        summary = self.run_preparation()

        self.assertEqual(summary["images"], 3)
        self.assertEqual(summary["annotations"], 3)
        self.assertEqual(summary["empty_labels"], 1)
        self.assertEqual((self.output / "valid" / "labels" / "valid-image.txt").stat().st_size, 0)
        train_rows = (self.output / "train" / "labels" / "train-image.txt").read_text().splitlines()
        self.assertEqual(len(train_rows), 2)
        self.assertTrue(train_rows[0].startswith("0 "))
        self.assertTrue(train_rows[1].startswith("1 "))
        self.assertEqual(
            train_rows[0],
            "0 0.3000000000 0.5000000000 0.4000000000 0.6000000000",
        )
        self.assertEqual(
            (self.source / "train" / "images" / "train-image.jpg").read_bytes(),
            (self.output / "train" / "images" / "train-image.jpg").read_bytes(),
        )
        self.assertFalse((self.output.parent / ".v3_detection_building").exists())
        self.assertFalse((self.artifacts.parent / ".v3_detection_building").exists())

        with (self.artifacts / "file_manifest.csv").open(encoding="utf-8", newline="") as source:
            manifest = list(csv.DictReader(source))
        with (self.artifacts / "annotation_conversion.csv").open(encoding="utf-8", newline="") as source:
            conversions = list(csv.DictReader(source))
        persisted_summary = json.loads((self.artifacts / "preparation_summary.json").read_text())
        self.assertEqual(len(manifest), 3)
        self.assertEqual(len(conversions), 3)
        self.assertTrue(all(row["source_image_sha256"] == row["prepared_image_sha256"] for row in manifest))
        self.assertTrue(all(row["source_annotation_count"] == row["prepared_annotation_count"] for row in manifest))
        self.assertEqual(persisted_summary["class_mapping"], {"0": "class zero", "1": "class one"})

    def test_rejects_detection_row_as_dataset_drift(self) -> None:
        label = self.source / "test" / "labels" / "test-image.txt"
        label.write_text("1 0.5000000000 0.5000000000 0.2000000000 0.2000000000\n", encoding="utf-8")

        with self.assertRaisesRegex(PreparationError, "expected a polygon"):
            self.run_preparation()

    def test_rejects_invalid_polygon(self) -> None:
        label = self.source / "test" / "labels" / "test-image.txt"
        label.write_text("1 0.2 0.3 1.2 0.4 0.5 0.8\n", encoding="utf-8")

        with self.assertRaisesRegex(PreparationError, "polygon_coordinate_out_of_range"):
            self.run_preparation()

    def test_rejects_output_inside_raw_directory(self) -> None:
        unsafe_output = self.source.parent / "prepared"

        with self.assertRaisesRegex(PreparationError, "must not be inside the raw-data directory"):
            prepare_dataset(
                self.source,
                unsafe_output,
                self.artifacts,
                expectations=TEST_EXPECTATIONS,
                project_root=self.root,
            )

    def test_refuses_existing_output_without_overwrite(self) -> None:
        self.run_preparation()

        with self.assertRaisesRegex(PreparationError, "Use --overwrite"):
            self.run_preparation()

    def test_overwrite_rebuilds_without_modifying_source(self) -> None:
        first_summary = self.run_preparation()
        source_before = fingerprint_dataset(self.source).digest
        (self.output / "unexpected.txt").write_text("old output", encoding="utf-8")

        second_summary = self.run_preparation(overwrite=True)

        self.assertFalse((self.output / "unexpected.txt").exists())
        self.assertEqual(fingerprint_dataset(self.source).digest, source_before)
        self.assertEqual(second_summary["source_dataset_fingerprint"], first_summary["source_dataset_fingerprint"])

    def test_fingerprint_is_deterministic_and_content_sensitive(self) -> None:
        first = fingerprint_dataset(self.source)
        second = fingerprint_dataset(self.source)
        image = self.source / "test" / "images" / "test-image.jpg"
        image.write_bytes(image.read_bytes() + b"changed")
        changed = fingerprint_dataset(self.source)

        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.file_hashes, second.file_hashes)
        self.assertNotEqual(first.digest, changed.digest)

    def test_hash_mismatch_cleans_temporary_build_without_finalizing(self) -> None:
        original_copy = shutil.copyfile

        def corrupting_copy(source: Path, destination: Path) -> str:
            result = original_copy(source, destination)
            Path(destination).write_bytes(Path(destination).read_bytes() + b"corrupt")
            return result

        with patch("bone_fracture_pipeline.prepare_dataset.shutil.copyfile", side_effect=corrupting_copy):
            with self.assertRaisesRegex(PreparationError, "hash mismatch"):
                self.run_preparation()

        self.assertFalse(self.output.exists())
        self.assertFalse(self.artifacts.exists())
        self.assertFalse((self.output.parent / ".v3_detection_building").exists())
        self.assertFalse((self.artifacts.parent / ".v3_detection_building").exists())


if __name__ == "__main__":
    unittest.main()
