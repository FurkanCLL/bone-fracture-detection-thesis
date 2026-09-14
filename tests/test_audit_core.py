from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from bone_fracture_audit.audit import (
    difference_hash,
    hamming_distance,
    match_images_and_labels,
    normalized_pixel_difference,
    run_audit,
    sha256_file,
)


class AuditCoreTests(unittest.TestCase):
    def test_matches_images_and_labels_by_stem(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            image_dir.mkdir()
            label_dir.mkdir()
            (image_dir / "paired.jpg").write_bytes(b"image")
            (image_dir / "image_only.png").write_bytes(b"image")
            (label_dir / "paired.txt").write_text("", encoding="utf-8")
            (label_dir / "label_only.txt").write_text("", encoding="utf-8")

            result = match_images_and_labels(image_dir, label_dir)

            self.assertEqual([(image.name, label.name) for image, label in result["pairs"]], [("paired.jpg", "paired.txt")])
            self.assertEqual([path.name for path in result["images_without_labels"]], ["image_only.png"])
            self.assertEqual([path.name for path in result["labels_without_images"]], ["label_only.txt"])

    def test_sha256_depends_on_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "first.bin"
            second = Path(temporary_directory) / "second.bin"
            first.write_bytes(b"same")
            second.write_bytes(b"same")
            self.assertEqual(sha256_file(first), sha256_file(second))
            second.write_bytes(b"different")
            self.assertNotEqual(sha256_file(first), sha256_file(second))

    def test_difference_hash_and_hamming_distance(self) -> None:
        first = Image.new("L", (16, 16))
        second = Image.new("L", (16, 16))
        for x in range(16):
            for y in range(16):
                first.putpixel((x, y), x * 10)
                second.putpixel((x, y), x * 10)
        first_hash = difference_hash(first)
        second_hash = difference_hash(second)
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(hamming_distance(first_hash, second_hash), 0)

    def test_normalized_pixel_difference_is_zero_for_matching_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first_path = Path(temporary_directory) / "first.png"
            second_path = Path(temporary_directory) / "second.png"
            Image.new("L", (20, 30), 120).save(first_path)
            Image.new("L", (20, 30), 120).save(second_path)
            self.assertEqual(normalized_pixel_difference(first_path, second_path), 0.0)

    def test_rejects_output_inside_raw_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_root = root / "raw"
            raw_root.mkdir()
            with self.assertRaisesRegex(ValueError, "outside the immutable raw-data directory"):
                run_audit(raw_root, raw_root / "audit", root / "report.md")


if __name__ == "__main__":
    unittest.main()
