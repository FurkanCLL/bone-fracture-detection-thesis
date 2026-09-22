from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from bone_fracture_pipeline.prepare_dataset import DatasetExpectations, SplitExpectation, fingerprint_dataset
from bone_fracture_pipeline.single_class_dataset import (
    DATA_YAML,
    SingleClassDatasetError,
    prepare_single_class_dataset,
    validate_single_class_dataset,
)


CLASSES = (
    "elbow positive", "fingers positive", "forearm fracture",
    "humerus fracture", "shoulder fracture", "wrist positive",
)
EXPECTATIONS = DatasetExpectations(
    class_names=CLASSES,
    splits={
        "train": SplitExpectation(2, 6, 1, (1, 1, 1, 1, 1, 1)),
        "valid": SplitExpectation(1, 1, 0, (0, 0, 1, 0, 0, 0)),
        "test": SplitExpectation(1, 0, 1, (0, 0, 0, 0, 0, 0)),
    },
)


class SingleClassDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "clahe"
        self.derived = self.root / "single_class"
        (self.source / "data.yaml").parent.mkdir(parents=True)
        (self.source / "data.yaml").write_text(
            "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASSES)),
            encoding="utf-8",
        )
        label_rows = {
            "train": {
                "multiple": "".join(
                    f"{class_id} {0.2 + class_id * 0.1:.10f} 0.5000000000 0.1000000000 0.1000000000\n"
                    for class_id in range(6)
                ),
                "empty": "",
            },
            "valid": {"validation": "2 0.4000000000 0.3000000000 0.1000000000 0.1000000000\n"},
            "test": {"heldout": ""},
        }
        for split, labels in label_rows.items():
            image_dir = self.source / split / "images"
            label_dir = self.source / split / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for stem, text in labels.items():
                (image_dir / f"{stem}.png").write_bytes(f"unchanged-{split}-{stem}".encode())
                (label_dir / f"{stem}.txt").write_text(text, encoding="utf-8", newline="\n")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def build(self) -> dict[str, object]:
        return prepare_single_class_dataset(
            self.source, self.derived,
            expected_source_fingerprint=None, expectations=EXPECTATIONS,
        )

    def validate(self, fingerprint: str | None = None) -> dict[str, object]:
        return validate_single_class_dataset(
            self.source, self.derived, expected_source_fingerprint=None,
            expected_derived_fingerprint=fingerprint, expectations=EXPECTATIONS,
        )

    def test_remaps_every_box_and_preserves_images_empty_files_and_splits(self) -> None:
        source_before = fingerprint_dataset(self.source).digest
        summary = self.build()
        self.assertEqual(summary["images"], 4)
        self.assertEqual(summary["annotations"], 7)
        self.assertEqual(summary["empty_labels"], 2)
        self.assertEqual(summary["class_mapping"], {str(index): 0 for index in range(6)})
        self.assertEqual(summary["splits"]["train"], {
            "images": 2, "labels": 2, "annotations": 6, "empty_labels": 1,
        })
        self.assertEqual((self.derived / "data.yaml").read_text(encoding="utf-8"), DATA_YAML)
        for split in ("train", "valid", "test"):
            for image in (self.source / split / "images").iterdir():
                self.assertEqual(image.read_bytes(), (self.derived / split / "images" / image.name).read_bytes())
        source_lines = (self.source / "train/labels/multiple.txt").read_text().splitlines()
        derived_lines = (self.derived / "train/labels/multiple.txt").read_text().splitlines()
        self.assertEqual(len(derived_lines), 6)
        for source_line, derived_line in zip(source_lines, derived_lines):
            self.assertEqual(derived_line.split()[0], "0")
            self.assertEqual(source_line.split()[1:], derived_line.split()[1:])
        self.assertEqual((self.derived / "train/labels/empty.txt").stat().st_size, 0)
        self.assertEqual((self.derived / "test/labels/heldout.txt").stat().st_size, 0)
        self.assertEqual(fingerprint_dataset(self.source).digest, source_before)
        self.assertEqual(self.validate(summary["derived_fingerprint"])["derived_fingerprint"], summary["derived_fingerprint"])

    def test_rejects_geometry_or_image_content_drift(self) -> None:
        self.build()
        label = self.derived / "valid/labels/validation.txt"
        label.write_text(label.read_text().replace("0.4000000000", "0.4500000000"), encoding="utf-8")
        with self.assertRaisesRegex(SingleClassDatasetError, "changed beyond its class token"):
            self.validate()
        label.write_bytes((self.source / "valid/labels/validation.txt").read_bytes().replace(b"2 ", b"0 ", 1))
        image = self.derived / "train/images/multiple.png"
        image.write_bytes(b"modified-image")
        with self.assertRaisesRegex(SingleClassDatasetError, "Image content changed"):
            self.validate()

    def test_rejects_missing_identity_or_wrong_source_class(self) -> None:
        self.build()
        (self.derived / "train/labels/empty.txt").unlink()
        with self.assertRaisesRegex(SingleClassDatasetError, "missing or extra relative files"):
            self.validate()

        bad_source = self.source / "valid/labels/validation.txt"
        bad_source.write_text(bad_source.read_text().replace("2 ", "6 ", 1), encoding="utf-8")
        second_output = self.root / "bad_output"
        with self.assertRaises(SingleClassDatasetError):
            prepare_single_class_dataset(
                self.source, second_output, expected_source_fingerprint=None, expectations=EXPECTATIONS
            )
        self.assertFalse(second_output.exists())

    def test_rebuilds_an_existing_derived_dataset_without_changing_the_result(self) -> None:
        self.build()
        original = fingerprint_dataset(self.derived).digest
        result = self.build()
        self.assertTrue(result["replaced_existing"])
        self.assertEqual(fingerprint_dataset(self.derived).digest, original)

    def test_rebuilds_a_partial_or_corrupt_derived_dataset(self) -> None:
        self.derived.mkdir()
        (self.derived / "partial.txt").write_text("incomplete", encoding="utf-8")
        result = self.build()
        self.assertTrue(result["replaced_existing"])
        self.assertEqual(result["fingerprint_file_count"], 9)
        self.assertFalse((self.derived / "partial.txt").exists())
        self.assertEqual(self.validate(result["derived_fingerprint"])["images"], 4)

    def test_failed_rebuild_keeps_the_existing_output(self) -> None:
        self.derived.mkdir()
        (self.derived / "partial.txt").write_text("keep", encoding="utf-8")
        (self.source / "valid/labels/validation.txt").write_text("6 0.4 0.3 0.1 0.1\n", encoding="utf-8")
        with self.assertRaises(SingleClassDatasetError):
            self.build()
        self.assertEqual((self.derived / "partial.txt").read_text(encoding="utf-8"), "keep")
        self.assertEqual(len(list(self.root.glob(".single_class_building_*"))), 0)

    @unittest.skipUnless(os.name == "nt", "Windows ACL regression check")
    def test_rebuild_replaces_a_private_directory_with_an_inheriting_one(self) -> None:
        def has_inherited_access() -> bool:
            return subprocess.run(
                ["icacls", str(self.derived)], check=True, capture_output=True, text=True,
            ).stdout.find("(I)") >= 0

        self.derived = Path(tempfile.mkdtemp(prefix="single_class_", dir=self.root))
        self.assertFalse(has_inherited_access())
        self.build()
        self.assertTrue(has_inherited_access())


if __name__ == "__main__":
    unittest.main()
