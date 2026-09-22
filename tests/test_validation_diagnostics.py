from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from bone_fracture_pipeline.validation_diagnostics import (
    _finite_number,
    _json_safe,
    _slice_batch,
    _without_targets,
    summarize_batch_targets,
    summarize_prediction_tensors,
    summarize_tensor,
    summarize_existing_diagnostics,
)


class ValidationDiagnosticTests(unittest.TestCase):
    def test_tensor_summary_counts_nonfinite_values(self) -> None:
        summary = summarize_tensor(torch.tensor([1.0, float("nan"), float("inf")]))

        self.assertFalse(summary["all_finite"])
        self.assertEqual(summary["finite_values"], 1)
        self.assertEqual(summary["nonfinite_values"], 2)
        self.assertEqual(summary["nan_values"], 1)
        self.assertEqual(summary["positive_infinity_values"], 1)

    def test_nested_prediction_summary_names_nonfinite_path(self) -> None:
        summary = summarize_prediction_tensors(
            {"boxes": torch.ones(1), "feats": [torch.tensor([float("nan")])]}
        )

        self.assertFalse(summary["all_finite"])
        self.assertEqual(summary["nonfinite_paths"], ["predictions.feats[0]"])

    def test_target_summary_preserves_empty_images(self) -> None:
        batch = {
            "im_file": ["a.png", "b.png"],
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([[3.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }

        summary = summarize_batch_targets(batch)

        self.assertEqual(summary["target_count"], 1)
        self.assertEqual(summary["empty_image_count"], 1)
        self.assertFalse(summary["images"][0]["empty_label"])
        self.assertTrue(summary["images"][1]["empty_label"])

    def test_slice_batch_remaps_target_index(self) -> None:
        batch = {
            "img": torch.zeros(2, 3, 4, 4),
            "im_file": ["a.png", "b.png"],
            "batch_idx": torch.tensor([0, 1, 1]),
            "cls": torch.tensor([[1.0], [2.0], [3.0]]),
            "bboxes": torch.ones(3, 4),
            "ori_shape": [(4, 4), (4, 4)],
        }

        sliced = _slice_batch(batch, 1)

        self.assertEqual(tuple(sliced["img"].shape), (1, 3, 4, 4))
        self.assertEqual(sliced["im_file"], ["b.png"])
        self.assertEqual(sliced["cls"].view(-1).tolist(), [2.0, 3.0])
        self.assertEqual(sliced["batch_idx"].tolist(), [0, 0])

    def test_without_targets_keeps_images_and_empties_annotations(self) -> None:
        batch = {
            "img": torch.ones(2, 3, 4, 4),
            "batch_idx": torch.tensor([0, 1]),
            "cls": torch.tensor([[1.0], [2.0]]),
            "bboxes": torch.ones(2, 4),
        }

        empty = _without_targets(batch)

        self.assertIs(empty["img"], batch["img"])
        self.assertEqual(empty["batch_idx"].numel(), 0)
        self.assertEqual(tuple(empty["cls"].shape), (0, 1))
        self.assertEqual(tuple(empty["bboxes"].shape), (0, 4))

    def test_finite_number_rejects_nan_and_text(self) -> None:
        self.assertTrue(_finite_number(1.25))
        self.assertFalse(_finite_number(float("nan")))
        self.assertFalse(_finite_number("not-a-number"))

    def test_json_safe_replaces_nested_nonfinite_numbers(self) -> None:
        converted = _json_safe({"values": [1.0, float("nan"), float("inf"), float("-inf")]})

        self.assertEqual(converted, {"values": [1.0, "nan", "infinity", "-infinity"]})

    def test_compact_summary_requires_all_diagnostic_inputs(self) -> None:
        with TemporaryDirectory() as directory, patch(
            "bone_fracture_pipeline.validation_diagnostics.installed_source_evidence",
            return_value={},
        ):
            with self.assertRaisesRegex(Exception, "Missing diagnostic report"):
                summarize_existing_diagnostics(Path(directory))


if __name__ == "__main__":
    unittest.main()
