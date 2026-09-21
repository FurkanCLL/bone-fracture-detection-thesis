from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from bone_fracture_audit.audit import sha256_file
from bone_fracture_pipeline.png_control import build_png_control_dataset
from bone_fracture_pipeline.prepare_dataset import DatasetExpectations, SplitExpectation, fingerprint_dataset
from bone_fracture_pipeline.preprocessing_core import PNG_SIGNATURE, PreprocessingError


TEST_CLASSES = ("class zero", "class one")
TEST_EXPECTATIONS = DatasetExpectations(
    class_names=TEST_CLASSES,
    splits={
        "train": SplitExpectation(2, 1, 1, (1, 0)),
        "valid": SplitExpectation(1, 1, 0, (0, 1)),
        "test": SplitExpectation(1, 0, 1, (0, 0)),
    },
)


class PngControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "data" / "prepared" / "v3_detection"
        self.output = self.root / "data" / "prepared" / "v3_detection_png"
        self.artifacts = self.root / "outputs" / "phase2" / "phase2e" / "png_control"
        self.canonical = self.root / "docs" / "evidence" / "phase2e" / "png_control_summary.json"
        self._create_source()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_build_preserves_decoded_pixels_labels_and_source(self) -> None:
        source_before = fingerprint_dataset(self.source).digest

        summary = self._run(source_before)

        self.assertEqual(summary["validation"]["decoded_pixel_equal_images"], 4)
        self.assertEqual(summary["validation"]["pixel_mismatches"], 0)
        self.assertTrue(summary["reproducibility"]["fingerprints_identical"])
        self.assertEqual(fingerprint_dataset(self.source).digest, source_before)
        self.assertTrue(self.canonical.is_file())

        for split in ("train", "valid", "test"):
            for source_image in (self.source / split / "images").glob("*.jpg"):
                control_image = self.output / split / "images" / f"{source_image.stem}.png"
                source_pixels = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
                control_pixels = cv2.imread(str(control_image), cv2.IMREAD_UNCHANGED)
                self.assertTrue(np.array_equal(source_pixels, control_pixels))
                self.assertEqual(control_image.read_bytes()[:8], PNG_SIGNATURE)
                source_label = self.source / split / "labels" / f"{source_image.stem}.txt"
                control_label = self.output / split / "labels" / source_label.name
                self.assertEqual(sha256_file(source_label), sha256_file(control_label))

    def test_overwrite_guard_and_rebuild_fingerprint(self) -> None:
        source_fingerprint = fingerprint_dataset(self.source).digest
        first = self._run(source_fingerprint)

        with self.assertRaisesRegex(PreprocessingError, "--overwrite"):
            self._run(source_fingerprint)

        second = self._run(source_fingerprint, overwrite=True)
        self.assertEqual(first["control_dataset_fingerprint"], second["control_dataset_fingerprint"])

    def test_unexpected_grayscale_source_fails_instead_of_normalizing(self) -> None:
        grayscale_path = self.source / "train" / "images" / "train-a.jpg"
        grayscale = np.full((18, 24), 80, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(grayscale_path), grayscale))
        source_fingerprint = fingerprint_dataset(self.source).digest

        with self.assertRaisesRegex(PreprocessingError, "three-channel uint8"):
            self._run(source_fingerprint)

    def _run(self, source_fingerprint: str, *, overwrite: bool = False) -> dict[str, object]:
        return build_png_control_dataset(
            self.source,
            self.output,
            self.artifacts,
            self.canonical,
            overwrite=overwrite,
            expectations=TEST_EXPECTATIONS,
            expected_source_fingerprint=source_fingerprint,
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
                ("train-a.jpg", "0 0.5 0.5 0.2 0.3\n", 20),
                ("train-b.jpg", "", 100),
            ],
            "valid": [("valid-a.jpg", "1 0.4 0.4 0.3 0.2\n", 180)],
            "test": [("test-a.jpg", "", 60)],
        }
        for split, entries in examples.items():
            images = self.source / split / "images"
            labels = self.source / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for index, (filename, label, level) in enumerate(entries):
                x = np.arange(24, dtype=np.uint8)[None, :, None]
                image = np.clip(level + x + np.array([0, 2, 4], dtype=np.uint8), 0, 255)
                image = np.repeat(image, 18, axis=0)
                image = np.roll(image, index, axis=1)
                self.assertTrue(cv2.imwrite(str(images / filename), image))
                (labels / f"{Path(filename).stem}.txt").write_text(label, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
