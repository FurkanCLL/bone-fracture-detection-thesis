from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from bone_fracture_audit.audit import sha256_file
from bone_fracture_pipeline.dataset_difficulty import (
    DifficultyAnalysisError,
    analyze_dataset,
    box_measurements,
    describe,
    orientation,
)
from bone_fracture_pipeline.dataset_difficulty_figures import generate_figures
from bone_fracture_pipeline.prepare_dataset import CLASS_NAMES


class DatasetDifficultyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "v3_detection"
        self.source.mkdir()
        (self.source / "data.yaml").write_text(
            "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES)),
            encoding="utf-8",
        )
        self._image("train", "two_boxes", (1000, 500),
                    "0 0.50 0.50 0.10 0.20\n1 0.25 0.25 0.10 0.10\n")
        self._image("train", "empty", (500, 500), "")
        self._image("valid", "one_box", (600, 1200), "2 0.50 0.50 0.05 0.10\n")
        self._image("valid", "empty", (800, 800), "")
        # These deliberately invalid held-out files must never enter analysis.
        (self.source / "test/images").mkdir(parents=True)
        (self.source / "test/labels").mkdir(parents=True)
        (self.source / "test/images/heldout.png").write_bytes(b"invalid-image")
        (self.source / "test/labels/heldout.txt").write_text("not a YOLO row", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _image(self, split: str, stem: str, size: tuple[int, int], label: str) -> None:
        image_dir = self.source / split / "images"
        label_dir = self.source / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color=(20, 40, 60)).save(image_dir / f"{stem}.png")
        (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8", newline="\n")

    def test_box_geometry_uses_aspect_ratio_preserving_letterbox_scale(self) -> None:
        box = box_measurements(0.1, 0.2, 1000, 500)
        self.assertAlmostEqual(box["normalized_area"], 0.02)
        self.assertEqual(box["width_px"], 100)
        self.assertEqual(box["height_px"], 100)
        self.assertEqual(box["area_px"], 10000)
        self.assertEqual(box["letterbox_scale_640"], 0.64)
        self.assertEqual(box["effective_width_640"], 64)
        self.assertEqual(box["effective_height_640"], 64)
        self.assertEqual(box["effective_short_side_640"], 64)
        self.assertEqual(orientation(1000, 500), "landscape")
        self.assertEqual(orientation(600, 1200), "portrait")
        self.assertEqual(orientation(1000, 975), "approximately_square")

    def test_composition_class_aggregation_and_empty_labels(self) -> None:
        result = analyze_dataset(self.source)
        train = result["summary"]["splits"]["train"]
        valid = result["summary"]["splits"]["valid"]
        self.assertEqual((train["images"], train["positive_images"], train["empty_label_images"]), (2, 1, 1))
        self.assertEqual((valid["images"], valid["positive_images"], valid["empty_label_images"]), (2, 1, 1))
        self.assertEqual((train["annotations"], valid["annotations"]), (2, 1))
        self.assertEqual(train["annotations_per_positive_image"]["median"], 2)
        self.assertEqual(train["class_annotation_counts"]["0"], 1)
        self.assertEqual(train["class_annotation_counts"]["1"], 1)
        self.assertEqual(valid["class_annotation_counts"]["2"], 1)
        self.assertEqual(train["normalized_area_below_fraction_of_image"]["0.01"]["count"], 0)
        self.assertEqual(train["normalized_area_below_fraction_of_image"]["0.05"]["count"], 2)
        self.assertEqual(valid["effective_short_side_below_px"]["32"]["count"], 1)
        class_zero = next(row for row in result["class_rows"] if row["split"] == "train" and row["class_id"] == 0)
        self.assertEqual((class_zero["annotations"], class_zero["positive_images"]), (1, 1))
        self.assertAlmostEqual(class_zero["median_normalized_area_pct"], 2.0)
        self.assertIsNone(describe([])["median"])

    def test_test_split_is_untouched_and_result_is_deterministic(self) -> None:
        source_hash = sha256_file(self.source / "train/images/two_boxes.png")
        first = analyze_dataset(self.source)
        (self.source / "test/labels/heldout.txt").write_text("still invalid and different", encoding="utf-8")
        second = analyze_dataset(self.source)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(first["summary"]["analyzed_splits"], ["train", "valid"])
        self.assertFalse(first["summary"]["test_split_opened_or_analyzed"])
        self.assertEqual(first["summary"]["source_train_valid_fingerprint_before"],
                         first["summary"]["source_train_valid_fingerprint_after"])
        self.assertEqual(sha256_file(self.source / "train/images/two_boxes.png"), source_hash)

    def test_condition_check_rejects_changed_box_geometry(self) -> None:
        condition = self.root / "png_control"
        for split in ("train", "valid"):
            shutil.copytree(self.source / split, condition / split)
        shutil.copy2(self.source / "data.yaml", condition / "data.yaml")
        result = analyze_dataset(self.source, condition_roots={"png_control": condition})
        self.assertEqual(result["summary"]["condition_geometry_checks"]["png_control"]["matching_label_hashes"], 4)
        (condition / "valid/labels/one_box.txt").write_text("2 0.50 0.50 0.30 0.10\n", encoding="utf-8")
        with self.assertRaisesRegex(DifficultyAnalysisError, "label geometry differs"):
            analyze_dataset(self.source, condition_roots={"png_control": condition})

    def test_figure_boundary_rejects_a_test_row(self) -> None:
        figure_dir = self.root / "figures"
        with self.assertRaisesRegex(ValueError, "train and valid rows only"):
            generate_figures([{"split": "test"}], [], [], figure_dir)
        self.assertFalse(figure_dir.exists())


if __name__ == "__main__":
    unittest.main()
