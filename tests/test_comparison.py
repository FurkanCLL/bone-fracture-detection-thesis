import tempfile
import unittest
from pathlib import Path

from bone_fracture_audit.comparison import _split_relative_path, compare_label_pair


class ComparisonTests(unittest.TestCase):
    def test_split_relative_path_removes_export_parents(self) -> None:
        self.assertEqual(_split_relative_path("export/version/train/images/example.jpg"), "train/images/example.jpg")

    def test_label_comparison_separates_class_remap_from_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            v3_label = root / "v3.txt"
            v4_label = root / "v4.txt"
            v3_label.write_text("3 0.1 0.2 0.3 0.4 0.5 0.6\n", encoding="utf-8")
            v4_label.write_text("4 0.1 0.2 0.3 0.4 0.5 0.6\n", encoding="utf-8")

            result = compare_label_pair(
                v3_label,
                v4_label,
                ["a", "b", "c", "humerus fracture"],
                ["a", "b", "c", "x", "humerus"],
            )

            self.assertFalse(result["byte_equal"])
            self.assertFalse(result["semantic_equal"])
            self.assertTrue(result["coordinates_equal"])
            self.assertEqual(result["transitions"][("humerus fracture", "humerus")], 1)
