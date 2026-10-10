from __future__ import annotations

import itertools
import json
import math
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import yaml
from PIL import Image

from bone_fracture_pipeline.detection_evaluation import (
    GroundTruth, Prediction, box_iou, detection_metrics, match_detections,
)
from bone_fracture_pipeline.error_analysis import (
    ErrorAnalysisError, _keep_unfused, best_official_row, compare_official_metrics, load_validation_samples,
    read_prediction_export, validate_export_rows, validate_image_paths, validation_fingerprint,
    verify_preprocessing_lineage, write_validation_snapshot,
)
from bone_fracture_pipeline.prepare_dataset import CLASS_NAMES


class DetectionEvaluationTests(unittest.TestCase):
    def match(self, predictions=(), targets=(), **overrides):
        return match_detections(predictions, targets, **{
            "confidence_threshold": 0.25, "iou_threshold": 0.5, **overrides,
        })

    def test_iou_identical_partial_disjoint_touching_and_degenerate(self):
        self.assertEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 0, 30, 10)), 0)
        self.assertEqual(box_iou((0, 0, 10, 10), (10, 0, 20, 10)), 0)
        self.assertEqual(box_iou((0, 0, 0, 10), (0, 0, 0, 10)), 0)

    def test_iou_rejects_reversed_missing_and_nonfinite_coordinates(self):
        for box in ((1, 0, 0, 1), (0, 0, 1), (0, 0, math.nan, 1), (0, 0, math.inf, 1)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                box_iou(box, (0, 0, 1, 1))

    def test_correct_class_and_inclusive_confidence_and_iou_boundaries(self):
        result = self.match([Prediction("p", 1, (0, 0, 10, 10), 0.25)],
                            [GroundTruth("g", 1, (0, 0, 20, 10))])
        self.assertEqual(result.metrics, {"tp": 1, "fp": 0, "fn": 0, "precision": 1, "recall": 1, "f1": 1})
        self.assertEqual(result.matches[0].iou, 0.5)

    def test_wrong_class_is_fp_and_fn_but_agnostic_matching_preserves_ids(self):
        prediction = Prediction("p", 2, (0, 0, 10, 10), 0.8)
        target = GroundTruth("g", 1, (0, 0, 10, 10))
        aware = self.match([prediction], [target])
        self.assertEqual((aware.metrics["tp"], aware.metrics["fp"], aware.metrics["fn"]), (0, 1, 1))
        self.assertEqual(self.match([prediction], [target], class_aware=False).metrics["tp"], 1)
        self.assertEqual((prediction.class_id, target.class_id), (2, 1))

    def test_duplicate_predictions_claim_a_target_only_once(self):
        predictions = [Prediction("low", 0, (0, 0, 10, 10), 0.7),
                       Prediction("high", 0, (0, 0, 10, 10), 0.9)]
        result = self.match(predictions, [GroundTruth("g", 0, (0, 0, 10, 10))])
        self.assertEqual(result.matches[0].prediction_id, "high")
        self.assertEqual(result.false_positive_ids, ("low",))
        self.assertEqual((result.metrics["tp"], result.metrics["fp"], result.metrics["fn"]), (1, 1, 0))

    def test_one_prediction_cannot_claim_two_targets(self):
        result = self.match([Prediction("p", 0, (0, 0, 10, 10), 0.9)],
                            [GroundTruth("b", 0, (0, 0, 10, 10)), GroundTruth("a", 0, (0, 0, 10, 10))])
        self.assertEqual(result.matches[0].ground_truth_id, "a")
        self.assertEqual(result.false_negative_ids, ("b",))

    def test_next_prediction_can_match_another_available_target(self):
        result = self.match([Prediction("p1", 0, (0, 0, 10, 10), 0.9),
                             Prediction("p2", 0, (0, 0, 10, 10), 0.8)],
                            [GroundTruth("g1", 0, (0, 0, 10, 10)), GroundTruth("g2", 0, (0, 0, 10, 10))])
        self.assertEqual([match.ground_truth_id for match in result.matches], ["g1", "g2"])

    def test_confidence_filtering_does_not_turn_filtered_predictions_into_fp(self):
        result = self.match([Prediction("p", 0, (0, 0, 10, 10), 0.249)],
                            [GroundTruth("g", 0, (0, 0, 10, 10))])
        self.assertEqual(result.filtered_prediction_ids, ("p",))
        self.assertEqual((result.metrics["tp"], result.metrics["fp"], result.metrics["fn"]), (0, 0, 1))

    def test_empty_predictions_ground_truth_and_empty_label_images(self):
        self.assertEqual(self.match().metrics, detection_metrics(0, 0, 0))
        self.assertEqual(self.match(targets=[GroundTruth("g", 0, (0, 0, 1, 1))]).metrics["fn"], 1)
        empty_label = self.match([Prediction("p", 0, (0, 0, 1, 1), 0.9)])
        self.assertEqual(empty_label.false_positive_ids, ("p",))
        self.assertEqual(empty_label.metrics["recall"], 0)

    def test_matching_is_independent_of_input_order_and_breaks_ties_by_id(self):
        predictions = [Prediction(name, 0, (0, 0, 10, 10), 0.8) for name in ("c", "a", "b")]
        targets = [GroundTruth(name, 0, (0, 0, 10, 10)) for name in ("z", "x")]
        expected = self.match(predictions, targets)
        for ordered_predictions in itertools.permutations(predictions):
            for ordered_targets in itertools.permutations(targets):
                self.assertEqual(self.match(ordered_predictions, ordered_targets), expected)
        self.assertEqual([(match.prediction_id, match.ground_truth_id) for match in expected.matches],
                         [("a", "x"), ("b", "z")])

    def test_invalid_scores_thresholds_classes_and_duplicate_ids_are_rejected(self):
        for threshold in (math.nan, -0.1, 1.1):
            with self.assertRaises(ValueError):
                self.match(confidence_threshold=threshold)
        for threshold in (0, math.nan, -0.1, 1.1):
            with self.assertRaises(ValueError):
                self.match(iou_threshold=threshold)
        for prediction in (Prediction("p", 0, (0, 0, 1, 1), math.nan),
                           Prediction("p", -1, (0, 0, 1, 1), 0.5)):
            with self.assertRaises(ValueError):
                self.match([prediction])
        duplicate = Prediction("p", 0, (0, 0, 1, 1), 0.5)
        with self.assertRaises(ValueError):
            self.match([duplicate, duplicate])

    def test_precision_recall_f1_and_invalid_counts(self):
        metrics = detection_metrics(2, 1, 2)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 4 / 7)
        for counts in ((-1, 0, 0), (1.5, 0, 0), (True, 0, 0)):
            with self.assertRaises(ValueError):
                detection_metrics(*counts)

    def test_ultralytics_iou_and_unambiguous_matching_agree(self):
        import torch
        from ultralytics.engine.validator import BaseValidator
        from ultralytics.utils.metrics import box_iou as framework_iou

        boxes = torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]])
        overlap = framework_iou(boxes, boxes)
        self.assertAlmostEqual(box_iou(boxes[0].tolist(), boxes[0].tolist()), float(overlap[0, 0]), places=6)
        native = BaseValidator.match_predictions(
            SimpleNamespace(iouv=torch.tensor([0.5])), torch.tensor([0., 2.]), torch.tensor([0., 1.]), overlap
        )
        result = self.match([Prediction("p0", 0, (0, 0, 10, 10), 0.9),
                             Prediction("p1", 2, (20, 0, 30, 10), 0.8)],
                            [GroundTruth("g0", 0, (0, 0, 10, 10)), GroundTruth("g1", 1, (20, 0, 30, 10))])
        self.assertEqual(native[:, 0].tolist(), [True, False])
        self.assertEqual(result.metrics["tp"], int(native.sum()))

    def test_conflict_documents_the_difference_from_ultralytics_ap_matching(self):
        import torch
        from ultralytics.engine.validator import BaseValidator

        native = BaseValidator.match_predictions(
            SimpleNamespace(iouv=torch.tensor([0.5])), torch.tensor([0., 0.]), torch.tensor([0., 0.]),
            torch.tensor([[1.0, 0.8], [0.6, 0.75]])
        )
        result = self.match([Prediction("high", 0, (0, 0, 10, 10), 0.9),
                             Prediction("low", 0, (0, 0, 8, 10), 0.8)],
                            [GroundTruth("g0", 0, (0, 0, 10, 10)), GroundTruth("g1", 0, (0, 0, 6, 10))])
        # Native AP deduplicates each prediction's best candidate before deduplicating targets.
        # Our diagnostic matcher can use the second available target instead.
        self.assertEqual(native[:, 0].tolist(), [True, False])
        self.assertEqual(result.metrics["tp"], 2)


class ValidationInfrastructureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "source"
        (self.dataset / "valid/images").mkdir(parents=True)
        (self.dataset / "valid/labels").mkdir()
        (self.dataset / "data.yaml").write_text(yaml.safe_dump({"names": dict(enumerate(CLASS_NAMES))}), encoding="utf-8")
        for name, label in (("positive", "1 0.5 0.5 0.2 0.4\n"), ("empty", "")):
            Image.new("RGB", (100, 200)).save(self.dataset / "valid/images" / f"{name}.png")
            (self.dataset / "valid/labels" / f"{name}.txt").write_text(label, encoding="utf-8")
        # Invalid held-out files expose accidental broad traversal without using real test data.
        (self.dataset / "test/images").mkdir(parents=True)
        (self.dataset / "test/labels").mkdir()
        (self.dataset / "test/images/heldout.png").write_bytes(b"not an image")
        (self.dataset / "test/labels/heldout.txt").write_text("not a label", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def samples(self):
        return load_validation_samples(self.dataset, expected_counts=(2, 1, 1))

    def test_validation_pairing_dimensions_ids_and_original_pixel_ground_truth(self):
        samples = self.samples()
        self.assertEqual([sample.sample_id for sample in samples], ["valid/empty", "valid/positive"])
        self.assertEqual(samples[1].ground_truth[0].class_id, 1)
        self.assertEqual(samples[1].ground_truth[0].xyxy, (40, 60, 60, 140))
        self.assertEqual((samples[1].width, samples[1].height), (100, 200))
        self.assertFalse(samples[0].ground_truth)

    def test_rejects_test_or_train_before_any_filesystem_access(self):
        with patch.object(Path, "resolve", side_effect=AssertionError("must reject before I/O")):
            for split in ("test", "train", "val", "../test"):
                with self.assertRaises(ErrorAnalysisError):
                    load_validation_samples(self.dataset, split=split)

    def test_fingerprinting_and_loading_never_open_or_traverse_held_out_files(self):
        original_open, original_iterdir = Path.open, Path.iterdir

        def guarded_open(path, *args, **kwargs):
            if "test" in path.parts:
                raise AssertionError("Held-out file was opened")
            return original_open(path, *args, **kwargs)

        def guarded_iterdir(path):
            if "test" in path.parts:
                raise AssertionError("Held-out directory was traversed")
            return original_iterdir(path)

        with patch.object(Path, "open", guarded_open), patch.object(Path, "iterdir", guarded_iterdir):
            samples = self.samples()
            first = validation_fingerprint(self.dataset, samples)
            snapshot = write_validation_snapshot(samples, self.root / "snapshot")
            self.assertEqual(first, validation_fingerprint(self.dataset, samples))
            self.assertNotIn("test", yaml.safe_load(snapshot.read_text(encoding="utf-8")))
        (self.dataset / "test/labels/heldout.txt").write_text("different invalid held-out content", encoding="utf-8")
        self.assertEqual(first, validation_fingerprint(self.dataset, self.samples()))

    def test_loader_path_guard_rejects_test_images(self):
        with self.assertRaises(ErrorAnalysisError):
            validate_image_paths([self.dataset / "test/images/heldout.png"], self.dataset / "valid/images")

    def test_fingerprint_and_snapshot_reject_injected_test_paths(self):
        heldout = replace(self.samples()[0], image_path=self.dataset / "test/images/heldout.png",
                          label_path=self.dataset / "test/labels/heldout.txt")
        with patch.object(Path, "open", side_effect=AssertionError("must reject before file reads")):
            with self.assertRaises(ErrorAnalysisError):
                validation_fingerprint(self.dataset, [heldout])
            with self.assertRaises(ErrorAnalysisError):
                write_validation_snapshot([heldout], self.root / "invalid_snapshot")
        self.assertFalse((self.root / "invalid_snapshot").exists())

    def test_unfused_diagnostic_override_survives_framework_deepcopy(self):
        model = SimpleNamespace()
        model.fuse = MethodType(_keep_unfused, model)
        copied = deepcopy(model)
        self.assertIs(copied.fuse(verbose=False), copied)
        self.assertIsNot(copied, model)
        self.assertIs(model.fuse(), model)

    def test_snapshot_is_validation_only_preserves_hashes_and_refuses_overwrite(self):
        samples = self.samples()
        path = write_validation_snapshot(samples, self.root / "snapshot")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual((data["train"], data["val"]), ("valid/images", "valid/images"))
        self.assertNotIn("test", data)
        with self.assertRaises(ErrorAnalysisError):
            write_validation_snapshot(samples, self.root / "snapshot")
        with self.assertRaises(ErrorAnalysisError):
            write_validation_snapshot(samples, self.dataset / "analysis")

    def test_missing_pairs_malformed_labels_and_count_drift_fail(self):
        with self.assertRaises(ErrorAnalysisError):
            load_validation_samples(self.dataset)
        label = self.dataset / "valid/labels/positive.txt"
        label.write_text("1 0.5 nan 0.2 0.4", encoding="utf-8")
        with self.assertRaises(ErrorAnalysisError):
            self.samples()
        label.unlink()
        with self.assertRaises(ErrorAnalysisError):
            self.samples()

    def test_original_preprocessing_manifest_detects_validation_content_drift(self):
        samples = self.samples()
        manifest = self.root / "manifest.csv"
        manifest.write_text("split,processed_filename,processed_image_sha256,processed_label_sha256\n" + "".join(
            f"valid,{sample.image_path.name},{sample.image_sha256},{sample.label_sha256}\n" for sample in samples
        ), encoding="utf-8")
        verify_preprocessing_lineage(manifest, samples)
        manifest.write_text(manifest.read_text(encoding="utf-8").replace(samples[0].image_sha256, "changed"), encoding="utf-8")
        with self.assertRaises(ErrorAnalysisError):
            verify_preprocessing_lineage(manifest, samples)

    def test_official_best_epoch_is_selected_instead_of_final_epoch(self):
        path = self.root / "results.csv"
        path.write_text("epoch,metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(B),metrics/recall(B)\n"
                        "57,0.14861,0.04732,0.21938,0.18667\n100,0.1,0.03,0.1,0.1\n", encoding="utf-8")
        best = best_official_row(path)
        self.assertEqual(best["epoch"], 57)
        observed = {key: value + 0.000005 for key, value in best.items() if key != "epoch"}
        self.assertTrue(compare_official_metrics(best, observed)["passed"])
        observed["metrics/mAP50(B)"] += 0.001
        self.assertFalse(compare_official_metrics(best, observed)["passed"])
        observed["metrics/mAP50(B)"] = math.nan
        with self.assertRaises(ErrorAnalysisError):
            compare_official_metrics(best, observed)

    def test_export_roundtrip_empty_images_and_invalid_numerical_or_split_values(self):
        row = {"sample_id": "valid/example", "split": "valid", "width": 100, "height": 200,
               "predictions": [{"box_id": "p", "class_id": 5, "xyxy": [0, 0, 10, 10], "confidence": 0.0011}],
               "ground_truth": []}
        empty = {**row, "sample_id": "valid/empty", "predictions": []}
        path = self.root / "predictions.jsonl"
        path.write_text(json.dumps(row) + "\n" + json.dumps(empty) + "\n", encoding="utf-8")
        self.assertEqual(read_prediction_export(path), [row, empty])
        self.assertEqual(validate_export_rows([row, empty]), {
            "images": 2, "predictions": 1, "annotations": 0, "empty_label_images": 2, "images_without_predictions": 1,
        })
        for changed in ({**row, "split": "test"}, {**row, "width": 0}):
            with self.assertRaises(ErrorAnalysisError):
                validate_export_rows([changed])
        with self.assertRaises(ErrorAnalysisError):
            validate_export_rows([row, row])
        row["predictions"][0]["confidence"] = math.inf
        with self.assertRaises(ErrorAnalysisError):
            validate_export_rows([row])


if __name__ == "__main__":
    unittest.main()
