from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from bone_fracture_audit.yolo import Annotation
from bone_fracture_pipeline.prepare_dataset import DatasetExpectations, SplitExpectation, prepare_dataset
from bone_fracture_pipeline.validate_conversion import (
    SERIALIZATION_TOLERANCE,
    ValidatedAnnotation,
    ValidationError,
    compare_reproduced_datasets,
    minimum_axis_aligned_box,
    minimum_box_matches,
    occupancy_ratio,
    polygon_area,
    run_validation,
    select_review_candidates,
    validate_annotation_geometry,
    vertices_are_contained,
)


TEST_EXPECTATIONS = DatasetExpectations(
    class_names=("class zero", "class one"),
    splits={
        "train": SplitExpectation(1, 1, 0, (1, 0)),
        "valid": SplitExpectation(1, 1, 0, (0, 1)),
        "test": SplitExpectation(1, 0, 1, (0, 0)),
    },
)


class ConversionGeometryTests(unittest.TestCase):
    def test_polygon_area_uses_shoelace_formula(self) -> None:
        self.assertAlmostEqual(polygon_area(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))), 0.5)

    def test_exact_containment_and_minimum_box(self) -> None:
        points = ((0.2, 0.3), (0.8, 0.2), (0.9, 0.7), (0.5, 0.9), (0.1, 0.6))
        expected = (0.5, 0.55, 0.8, 0.7)

        self.assertEqual(minimum_axis_aligned_box(points), expected)
        self.assertTrue(vertices_are_contained(points, expected))
        self.assertTrue(minimum_box_matches(points, expected))

    def test_serialization_tolerance_accepts_only_rounding_scale_difference(self) -> None:
        points = ((0.12345678904, 0.2), (0.8, 0.2), (0.8, 0.9), (0.12345678904, 0.9))
        rounded_box = tuple(float(f"{value:.10f}") for value in minimum_axis_aligned_box(points))

        self.assertTrue(minimum_box_matches(points, rounded_box, tolerance=SERIALIZATION_TOLERANCE))
        self.assertFalse(minimum_box_matches(points, (0.5, 0.55, 0.6, 0.7)))

    def test_occupancy_ratio_compares_polygon_and_box_area(self) -> None:
        triangle = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

        self.assertAlmostEqual(occupancy_ratio(triangle, (0.5, 0.5, 1.0, 1.0)), 0.5)

    def test_rejects_mismatched_geometry(self) -> None:
        source = Annotation(
            1,
            0,
            "polygon",
            0.5,
            0.5,
            0.6,
            0.6,
            ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)),
        )
        prepared = Annotation(1, 0, "box", 0.5, 0.5, 0.5, 0.5)

        with self.assertRaisesRegex(ValidationError, "does not contain"):
            validate_annotation_geometry(source, prepared)

    def test_review_candidate_selection_is_stable_and_excludes_test(self) -> None:
        records = [
            self._record("train", "a.jpg", 1, 0, 0.2, 0.01),
            self._record("valid", "b.jpg", 1, 1, 0.8, 0.04),
            self._record("test", "c.jpg", 1, 1, 0.1, 0.02),
        ]

        first = select_review_candidates(records, rank_count=1)
        second = select_review_candidates(records, rank_count=1)

        self.assertEqual(first, second)
        self.assertEqual({candidate.annotation.split for candidate in first}, {"train", "valid"})
        self.assertEqual({candidate.annotation.class_id for candidate in first}, {0, 1})

    @staticmethod
    def _record(
        split: str,
        filename: str,
        line: int,
        class_id: int,
        occupancy: float,
        box_area: float,
    ) -> ValidatedAnnotation:
        side = box_area**0.5
        return ValidatedAnnotation(
            split,
            filename,
            line,
            line,
            class_id,
            ((0.2, 0.2), (0.8, 0.2), (0.5, 0.8)),
            (0.5, 0.5, side, side),
            (0.5, 0.5, side, side),
            occupancy * box_area,
            box_area,
            occupancy,
            0.1,
            100,
            100,
            1,
        )


class Phase2BIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "data" / "raw" / "canonical-v3"
        self.prepared = self.root / "data" / "prepared" / "v3_detection"
        self.phase2a_artifacts = self.root / "outputs" / "phase2a" / "v3_detection"
        self.phase2b_output = self.root / "outputs" / "phase2b" / "v3_detection"
        self._create_source()
        prepare_dataset(
            self.source,
            self.prepared,
            self.phase2a_artifacts,
            expectations=TEST_EXPECTATIONS,
            project_root=self.root,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_source(self) -> None:
        self.source.mkdir(parents=True)
        (self.source / "data.yaml").write_text(
            "names: ['class zero', 'class one']\nnc: 2\n", encoding="utf-8"
        )
        labels = {
            "train": ("train.jpg", "0 0.2 0.2 0.8 0.2 0.7 0.8 0.3 0.7\n"),
            "valid": ("valid.jpg", "1 0.1 0.3 0.6 0.1 0.9 0.7 0.4 0.9\n"),
            "test": ("test.jpg", ""),
        }
        for split, (filename, label) in labels.items():
            image_dir = self.source / split / "images"
            label_dir = self.source / split / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("L", (32, 24), color=80 + len(split)).save(image_dir / filename, format="JPEG")
            (label_dir / f"{Path(filename).stem}.txt").write_text(label, encoding="utf-8")

    def test_full_validation_writes_reproducible_train_valid_review(self) -> None:
        summary = run_validation(
            self.source,
            self.prepared,
            self.phase2a_artifacts,
            self.phase2b_output,
            expectations=TEST_EXPECTATIONS,
            project_root=self.root,
        )

        self.assertTrue(summary["technically_ready_for_phase2c"])
        self.assertTrue(summary["reproducibility"]["success"])
        self.assertEqual(summary["geometry"]["annotations_checked"], 2)
        self.assertEqual(summary["visual_review_images"], 2)
        self.assertTrue((self.phase2b_output / "visual_review" / "contact_sheet.jpg").is_file())
        index_text = (self.phase2b_output / "visual_review_index.csv").read_text(encoding="utf-8")
        self.assertIn("train", index_text)
        self.assertIn("valid", index_text)
        self.assertNotIn("test.jpg", index_text)
        persisted = json.loads((self.phase2b_output / "validation_summary.json").read_text())
        self.assertEqual(persisted["independent_consistency"]["empty_labels"], 1)

    def test_reproduction_comparison_detects_changed_label(self) -> None:
        with tempfile.TemporaryDirectory() as reproduction_name:
            reproduction = Path(reproduction_name) / "prepared"
            # copytree is appropriate here because this test needs an independent comparison tree.
            shutil.copytree(self.prepared, reproduction)
            matching = compare_reproduced_datasets(self.prepared, reproduction, class_count=2)
            self.assertTrue(matching["success"])

            changed_label = reproduction / "train" / "labels" / "train.txt"
            changed_label.write_text("0 0.5000000000 0.5000000000 0.5000000000 0.5000000000\n", encoding="utf-8")
            changed = compare_reproduced_datasets(self.prepared, reproduction, class_count=2)
            self.assertFalse(changed["success"])
            self.assertEqual(changed["label_hash_mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()
