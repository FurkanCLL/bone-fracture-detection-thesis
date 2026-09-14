from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from bone_fracture_audit.review import prepare_review_set


# Draws distinct synthetic images for the selection test.
def _write_pattern(path: Path, offset: int) -> None:
    image = Image.new("L", (80, 60), 10)
    draw = ImageDraw.Draw(image)
    draw.rectangle((5 + offset, 5, 25 + offset, 50), fill=100 + offset * 10)
    draw.line((0, offset * 5, 79, 59 - offset * 5), fill=220, width=3)
    image.save(path)


class ReviewSetTests(unittest.TestCase):
    def test_creates_seeded_train_and_validation_copies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "data" / "raw" / "dataset"
            for split in ("train", "valid", "test"):
                (dataset_root / split / "images").mkdir(parents=True)
                (dataset_root / split / "labels").mkdir()
                for index in range(4):
                    image_path = dataset_root / split / "images" / f"image_{split}_{index}.jpg"
                    label_path = dataset_root / split / "labels" / f"image_{split}_{index}.txt"
                    _write_pattern(image_path, index)
                    label_path.write_text("", encoding="utf-8")

            output_dir = root / "outputs" / "review"
            rows = prepare_review_set(dataset_root, output_dir, seed=7, train_count=2, valid_count=1)

            self.assertEqual([row["split"] for row in rows].count("train"), 2)
            self.assertEqual([row["split"] for row in rows].count("valid"), 1)
            self.assertNotIn("test", {row["split"] for row in rows})
            self.assertEqual(len({row["source_key"] for row in rows}), 3)
            self.assertTrue(all(row["source_sha256"] == row["review_sha256"] for row in rows))
            self.assertEqual(len(list(output_dir.glob("empty_*.jpg"))), 3)

            repeated_rows = prepare_review_set(dataset_root, output_dir, seed=7, train_count=2, valid_count=1)
            self.assertEqual(rows, repeated_rows)


if __name__ == "__main__":
    unittest.main()
