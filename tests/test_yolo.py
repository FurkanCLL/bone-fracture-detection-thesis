from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bone_fracture_audit.yolo import parse_yolo_label


class YoloLabelTests(unittest.TestCase):
    def parse(self, text: str):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "label.txt"
            path.write_text(text, encoding="utf-8")
            return parse_yolo_label(path, {0, 1})

    def test_parses_valid_annotation(self) -> None:
        result = self.parse("1 0.5 0.4 0.2 0.3\n")
        self.assertFalse(result.is_empty)
        self.assertEqual(len(result.annotations), 1)
        self.assertEqual(result.annotations[0].class_id, 1)
        self.assertEqual(result.annotations[0].annotation_type, "box")
        self.assertEqual(result.issues, ())

    def test_recognizes_empty_label(self) -> None:
        result = self.parse("\n  \n")
        self.assertTrue(result.is_empty)
        self.assertEqual(result.annotations, ())
        self.assertEqual(result.issues, ())

    def test_reports_malformed_and_non_numeric_rows(self) -> None:
        result = self.parse("0 0.5 0.5 0.2\n0 x 0.5 0.2 0.2\n")
        self.assertEqual([issue.code for issue in result.issues], ["incorrect_value_count", "non_numeric_value"])

    def test_parses_polygon_and_derives_box(self) -> None:
        result = self.parse("1 0.2 0.3 0.8 0.3 0.8 0.9 0.2 0.9\n")
        annotation = result.annotations[0]
        self.assertEqual(annotation.annotation_type, "polygon")
        self.assertEqual(len(annotation.polygon_points), 4)
        self.assertAlmostEqual(annotation.x_center, 0.5)
        self.assertAlmostEqual(annotation.y_center, 0.6)
        self.assertAlmostEqual(annotation.width, 0.6)
        self.assertAlmostEqual(annotation.height, 0.6)
        self.assertEqual(result.issues, ())

    def test_reports_invalid_polygon_coordinate(self) -> None:
        result = self.parse("0 0.2 0.3 1.2 0.3 0.8 0.9\n")
        self.assertEqual([issue.code for issue in result.issues], ["polygon_coordinate_out_of_range"])
        self.assertEqual(result.annotations, ())

    def test_reports_invalid_class_and_geometry(self) -> None:
        result = self.parse("3 1.2 0.5 0 -0.1\n")
        self.assertEqual(
            {issue.code for issue in result.issues},
            {"unexpected_class_id", "center_out_of_range", "non_positive_size", "box_outside_image"},
        )
        self.assertEqual(result.annotations, ())

    def test_reports_box_that_crosses_image_boundary(self) -> None:
        result = self.parse("0 0.95 0.5 0.2 0.4\n")
        self.assertEqual([issue.code for issue in result.issues], ["box_outside_image"])
        self.assertEqual(result.annotations, ())

    def test_reports_duplicate_row_without_rewriting_it(self) -> None:
        result = self.parse("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
        self.assertEqual(len(result.annotations), 2)
        self.assertEqual([issue.code for issue in result.issues], ["duplicate_annotation"])


if __name__ == "__main__":
    unittest.main()
