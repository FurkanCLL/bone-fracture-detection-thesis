from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

from bone_fracture_pipeline.detection_evaluation import GroundTruth, Prediction
from bone_fracture_pipeline.error_analysis import (
    ErrorAnalysisError, _capture_validation, _diagnostic_fingerprint, _load_diagnostic_samples,
    _validate_diagnostic_rows, _write_diagnostic_snapshot, load_validation_samples,
    validate_export_rows, validation_fingerprint, write_validation_snapshot,
)
from bone_fracture_pipeline.error_analysis_training import read_stage1_validation, verify_sample_export
from bone_fracture_pipeline.error_analysis_quantitative import (
    CATEGORIES, SIZE_GROUPS, aggregate_evaluations, analyze_exports, categorize_false_negatives,
    class_and_size_analysis, class_confusion, confidence_localization, evaluate_rows,
    false_positive_analysis, sensitivity, size_group, spearman_correlation,
    summarize_taxonomy, validate_numerical_tree, run_stage2, verify_protected_files,
)
from bone_fracture_pipeline.prepare_dataset import CLASS_NAMES


class DiagnosticRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "source"
        for split in ("train", "valid"):
            (self.dataset / split / "images").mkdir(parents=True)
            (self.dataset / split / "labels").mkdir()
            Image.new("RGB", (100, 200)).save(self.dataset / split / "images/example.png")
            (self.dataset / split / "labels/example.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
        (self.dataset / "data.yaml").write_text(yaml.safe_dump({"names": dict(enumerate(CLASS_NAMES))}), encoding="utf-8")
        (self.dataset / "test/images").mkdir(parents=True)
        (self.dataset / "test/images/heldout.png").write_bytes(b"invalid held-out fixture")

    def tearDown(self):
        self.temporary.cleanup()

    def samples(self, split):
        return _load_diagnostic_samples(self.dataset, split=split, expected_counts=(1, 1, 0))

    def test_protected_hash_change_is_rejected(self):
        source = self.root / "requirements.txt"
        source.write_text("fixed dependency fixture\n", encoding="utf-8")
        baseline = self.root / "protection.json"
        self.assertTrue(verify_protected_files(self.root, baseline)["all_hashes_unchanged"])
        source.write_text("modified dependency fixture\n", encoding="utf-8")
        with self.assertRaises(ErrorAnalysisError):
            verify_protected_files(self.root, baseline)

    def test_changed_frozen_method_stops_before_prediction_loading(self):
        base = self.root / "outputs/error_analysis/stage2"
        base.mkdir(parents=True)
        (base / "method_pre_execution.json").write_text(json.dumps({"method": {"changed": True}}), encoding="utf-8")
        with patch("bone_fracture_pipeline.error_analysis_quantitative.read_stage1_validation",
                   side_effect=AssertionError("No source loading after method drift")):
            with self.assertRaises(ErrorAnalysisError):
                run_stage2(self.root, base / "unused", publish=False)
        self.assertFalse((base / "unused").exists())

    def test_report_and_figures_reject_heldout_population_before_writing(self):
        from bone_fracture_pipeline.error_analysis_figures import generate_figures
        from bone_fracture_pipeline.error_analysis_report import write_report

        result = {"split_counts": {"train": {}, "valid": {}, "test": {}}}
        with self.assertRaises(ValueError):
            generate_figures(result, self.root / "figures")
        with self.assertRaises(ValueError):
            write_report({"analysis": result}, self.root / "report.md")
        self.assertFalse((self.root / "figures").exists())
        self.assertFalse((self.root / "report.md").exists())

    def row(self, split):
        sample = self.samples(split)[0]
        return {"sample_id": sample.sample_id, "split": split, "image": f"{split}/images/example.png",
                "width": sample.width, "height": sample.height, "image_sha256": sample.image_sha256,
                "label_sha256": sample.label_sha256, "predictions": [],
                "ground_truth": json.loads(json.dumps([asdict(box) for box in sample.ground_truth]))}

    def test_train_snapshot_and_fingerprint_are_explicit_and_isolated(self):
        samples = self.samples("train")
        original_open, original_iterdir = Path.open, Path.iterdir

        def guarded_open(path, *args, **kwargs):
            if "test" in path.parts:
                raise AssertionError("Held-out access")
            return original_open(path, *args, **kwargs)

        def guarded_iterdir(path):
            if "test" in path.parts:
                raise AssertionError("Held-out traversal")
            return original_iterdir(path)

        with patch.object(Path, "open", guarded_open), patch.object(Path, "iterdir", guarded_iterdir):
            before = _diagnostic_fingerprint(self.dataset, samples, split="train")
            yaml_path = _write_diagnostic_snapshot(samples, self.root / "train_copy", split="train")
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            self.assertEqual((data["train"], data["val"]), ("train/images", "train/images"))
            self.assertNotIn("test", data)
            self.assertEqual(before, _diagnostic_fingerprint(self.dataset, self.samples("train"), split="train"))

    def test_test_route_is_rejected_before_filesystem_or_framework_access(self):
        with patch.object(Path, "resolve", side_effect=AssertionError("Must reject before I/O")):
            for split in ("test", "../test", "val", "all"):
                with self.subTest(split=split):
                    with self.assertRaises(ErrorAnalysisError):
                        _load_diagnostic_samples(self.dataset, split=split, expected_counts=(0, 0, 0))
                    with self.assertRaises(ErrorAnalysisError):
                        _diagnostic_fingerprint(self.dataset, [], split=split)
                    with self.assertRaises(ErrorAnalysisError):
                        _write_diagnostic_snapshot([], self.root / "invalid", split=split)
                    with self.assertRaises(ErrorAnalysisError):
                        _capture_validation(self.root / "missing.pt", self.root / "missing.yaml", [], {}, source_split=split)

    def test_existing_stage1_wrappers_still_reject_train(self):
        samples = self.samples("train")
        with self.assertRaises(ErrorAnalysisError):
            load_validation_samples(self.dataset, split="train")
        with self.assertRaises(ErrorAnalysisError):
            validation_fingerprint(self.dataset, samples)
        with self.assertRaises(ErrorAnalysisError):
            write_validation_snapshot(samples, self.root / "invalid")
        with self.assertRaises(ErrorAnalysisError):
            validate_export_rows([self.row("train")])

    def test_training_export_schema_and_source_verification(self):
        row = self.row("train")
        self.assertEqual(_validate_diagnostic_rows([row], split="train")["annotations"], 1)
        verify_sample_export([row], self.samples("train"), split="train")
        row["label_sha256"] = "changed"
        with self.assertRaises(ErrorAnalysisError):
            verify_sample_export([row], self.samples("train"), split="train")

    def test_split_mismatch_and_injected_heldout_paths_fail(self):
        train = self.samples("train")[0]
        heldout = replace(train, image_path=self.dataset / "test/images/heldout.png")
        with self.assertRaises(ErrorAnalysisError):
            _diagnostic_fingerprint(self.dataset, [heldout], split="train")
        with self.assertRaises(ErrorAnalysisError):
            _write_diagnostic_snapshot([heldout], self.root / "invalid", split="train")
        with self.assertRaises(ErrorAnalysisError):
            _validate_diagnostic_rows([self.row("valid")], split="train")

    def test_primary_prediction_hash_mismatch_fails_before_source_loading(self):
        evidence_dir = self.root / "docs/evidence/error_analysis"
        evidence_dir.mkdir(parents=True)
        export = self.root / "outputs/error_analysis/stage1/example/validation_predictions.jsonl"
        export.parent.mkdir(parents=True)
        export.write_text("invalid payload", encoding="utf-8")
        metadata = {"dataset": {"root": "data/prepared/v3_detection_clahe"},
                    "prediction_export": {"path": export.relative_to(self.root).as_posix(), "sha256": "wrong"}}
        (evidence_dir / "stage1_validation.json").write_text(json.dumps(metadata), encoding="utf-8")
        with patch("bone_fracture_pipeline.error_analysis_training.load_validation_samples", side_effect=AssertionError("must stop first")):
            with self.assertRaises(ErrorAnalysisError):
                read_stage1_validation(self.root)


class QuantitativeAnalysisTests(unittest.TestCase):
    def row(self, predictions=(), targets=(), *, name="example", split="valid", width=640, height=640):
        return {"sample_id": f"{split}/{name}", "split": split, "image": f"{split}/images/{name}.png",
                "width": width, "height": height, "image_sha256": f"{split}/{name}", "label_sha256": f"label/{split}/{name}",
                "predictions": [asdict(prediction) for prediction in predictions],
                "ground_truth": [asdict(target) for target in targets]}

    def reference(self, row):
        return evaluate_rows([row], confidence=0.25, iou=0.50)

    def category(self, predictions, targets=None):
        if targets is None:
            targets = [GroundTruth("g", 0, (0, 0, 10, 10))]
        return categorize_false_negatives(self.reference(self.row(predictions, targets)))[0]

    def test_complete_miss_means_no_retained_nearby_candidate(self):
        for predictions in ([], [Prediction("p", 0, (100, 100, 110, 110), 0.9)]):
            record = self.category(predictions)
            self.assertEqual(record["primary_category"], "complete_miss")
            self.assertFalse(record["flags"]["retained_nearby_candidate"])

    def test_localization_failure_requires_correct_class_high_confidence_and_nearby_overlap(self):
        record = self.category([Prediction("p", 0, (0, 0, 3, 10), 0.9)])
        self.assertEqual(record["primary_category"], "localization_failure")
        self.assertTrue(record["candidate_combination_flags"]["correct_class_high_confidence_poor_overlap"])

    def test_low_confidence_requires_suitable_correct_class_overlap(self):
        record = self.category([Prediction("p", 0, (0, 0, 10, 10), 0.249)])
        self.assertEqual(record["primary_category"], "low_confidence_detection")
        self.assertEqual(record["best_nearby_candidate"]["iou"], 1)

    def test_classification_requires_high_confidence_sufficient_overlap(self):
        record = self.category([Prediction("p", 1, (0, 0, 10, 10), 0.9)])
        self.assertEqual(record["primary_category"], "classification_error")
        self.assertTrue(record["flags"]["classification_error"])

    def test_matching_conflict_has_precedence_and_records_other_target(self):
        targets = [GroundTruth("g0", 0, (0, 0, 10, 10)), GroundTruth("g1", 0, (0, 0, 10, 10))]
        record = self.category([Prediction("p", 0, (0, 0, 10, 10), 0.9), Prediction("low", 0, (0, 0, 10, 10), 0.2)], targets)
        self.assertEqual(record["ground_truth_id"], "g1")
        self.assertEqual(record["primary_category"], "matching_conflict")
        self.assertTrue(record["flags"]["low_confidence_detection"])
        self.assertEqual(record["competing_assignments"], [{"prediction_id": "p", "assigned_ground_truth_id": "g0"}])

    def test_low_confidence_wrong_class_poor_overlap_is_mixed(self):
        record = self.category([Prediction("p", 1, (0, 0, 3, 10), 0.2)])
        self.assertEqual(record["primary_category"], "mixed_or_unresolved")
        self.assertTrue(record["candidate_combination_flags"]["wrong_class_low_confidence_poor_overlap"])
        self.assertFalse(record["flags"]["low_confidence_detection"])
        self.assertFalse(record["flags"]["classification_error"])

    def test_low_confidence_wrong_class_sufficient_overlap_still_requires_two_corrections(self):
        self.assertEqual(self.category([Prediction("p", 1, (0, 0, 10, 10), 0.2)])["primary_category"], "mixed_or_unresolved")

    def test_multiple_strong_signals_remain_nonexclusive_and_primary_mixed(self):
        record = self.category([Prediction("low", 0, (0, 0, 10, 10), 0.2), Prediction("wrong", 1, (0, 0, 10, 10), 0.9)])
        self.assertEqual(record["primary_category"], "mixed_or_unresolved")
        self.assertTrue(record["flags"]["low_confidence_detection"])
        self.assertTrue(record["flags"]["classification_error"])
        record = self.category([Prediction("poor", 0, (0, 0, 3, 10), 0.9), Prediction("wrong", 1, (0, 0, 10, 10), 0.8)])
        self.assertEqual(record["primary_category"], "mixed_or_unresolved")
        self.assertTrue(record["flags"]["localization_failure"])

    def test_nearby_and_reference_boundaries_are_inclusive(self):
        self.assertEqual(self.category([Prediction("p", 0, (0, 0, 1, 10), 0.25)])["primary_category"], "localization_failure")
        self.assertEqual(self.category([Prediction("p", 0, (0, 0, 0.99, 10), 0.25)])["primary_category"], "complete_miss")
        row = self.row([Prediction("p", 0, (0, 0, 5, 10), 0.25)], [GroundTruth("g", 0, (0, 0, 10, 10))])
        self.assertFalse(categorize_false_negatives(self.reference(row)))
        self.assertEqual(aggregate_evaluations(self.reference(row))["tp"], 1)

    def test_taxonomy_rejects_other_operating_points(self):
        wrong_reference = evaluate_rows([self.row([], [GroundTruth("g", 0, (0, 0, 10, 10))])], confidence=0.05, iou=0.50)
        with self.assertRaises(ErrorAnalysisError):
            categorize_false_negatives(wrong_reference)

    def test_taxonomy_counts_exhaustive_unique_and_class_percentages(self):
        rows = [self.row([], [GroundTruth("g", class_id, (0, 0, 10, 10))], name=str(index))
                for index, class_id in enumerate((0, 1, 1))]
        records = categorize_false_negatives(evaluate_rows(rows, confidence=0.25, iou=0.50))
        summary = summarize_taxonomy(records)
        self.assertEqual(summary["total_false_negatives"], 3)
        self.assertEqual(summary["primary_categories"]["complete_miss"], {"count": 3, "percentage_of_fn": 100})
        self.assertEqual(sum(item["count"] for item in summary["primary_categories"].values()), 3)
        self.assertEqual(summary["class_rows"][1]["fn"], 2)
        self.assertEqual(summary["class_rows"][1]["categories"]["complete_miss"]["percentage_of_class_fn"], 100)

    def test_size_boundaries_and_class_totals_partition_targets(self):
        rows = []
        for index, size in enumerate((31.99, 32, 63.99, 64)):
            target = GroundTruth("g", index, (0, 0, size, size))
            row = self.row([Prediction("p", index, target.xyxy, 0.9)], [target], name=str(index))
            self.assertEqual(size_group(target, row)[0], SIZE_GROUPS[(0, 1, 1, 2)[index]])
            rows.append(row)
        reference = evaluate_rows(rows, confidence=0.25, iou=0.50)
        classes, sizes = class_and_size_analysis(reference, [])
        self.assertEqual(sum(row["support"] for row in classes), 4)
        self.assertEqual(sum(row["tp"] for row in classes), 4)
        self.assertEqual([row["support"] for row in sizes if row["class_id"] is None], [1, 2, 1])
        self.assertIsNone(next(row["recall"] for row in sizes if row["support"] == 0))

    def test_empty_distributions_and_zero_denominators_are_explicit(self):
        reference = self.reference(self.row())
        self.assertEqual(aggregate_evaluations(reference)["f1"], 0)
        classes, sizes = class_and_size_analysis(reference, [])
        self.assertEqual(len(classes), 6)
        self.assertIsNone(classes[0]["matched_iou"]["median"])
        self.assertTrue(all(row["recall"] is None for row in sizes))
        summary = summarize_taxonomy([])
        self.assertEqual(set(summary["primary_categories"]), set(CATEGORIES))
        self.assertIsNone(summary["primary_categories"]["complete_miss"]["percentage_of_fn"])

    def test_sensitivity_grids_preserve_denominators_and_matching_modes(self):
        row = self.row([Prediction("p", 0, (0, 0, 10, 10), 0.1)], [GroundTruth("g", 0, (0, 0, 10, 10))])
        iou_rows, confidence_rows = sensitivity([row])
        self.assertEqual((len(iou_rows), len(confidence_rows)), (12, 10))
        self.assertTrue(all(row["ground_truth_count"] == 1 for row in iou_rows + confidence_rows))
        aware = [row for row in confidence_rows if row["class_aware"]]
        self.assertEqual([row["tp"] for row in aware], [1, 1, 1, 0, 0])
        self.assertEqual([row["matched_ground_truth_count"] for row in aware], [1, 1, 1, 0, 0])

    def test_spearman_ties_degenerate_samples_and_nonfinite_rejection(self):
        self.assertAlmostEqual(spearman_correlation([1, 1, 2, 3], [5, 5, 7, 9]), 1)
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(spearman_correlation([], []))
        self.assertIsNone(spearman_correlation([1, 1, 1], [1, 2, 3]))
        with self.assertRaises(ErrorAnalysisError):
            spearman_correlation([1, math.nan, 3], [1, 2, 3])

    def test_confidence_bins_include_boundaries_and_exclude_empty_image_geometry(self):
        scores = (0.0011, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
        predictions = [Prediction(str(index), 0, (0, 0, 10, 10), score) for index, score in enumerate(scores)]
        result = confidence_localization([self.row(predictions, [GroundTruth("g", 0, (0, 0, 10, 10))]),
                                          self.row(predictions[:2], [], name="empty")])
        self.assertEqual(result["annotated_image_candidate_count"], 7)
        self.assertEqual(result["excluded_empty_image_candidate_count"], 2)
        self.assertEqual([row["candidates"] for row in result["bins"]], [1, 1, 1, 1, 1, 2])
        self.assertTrue(all(row["best_any_class_iou"]["median"] == 1 for row in result["bins"]))

    def test_fp_groups_empty_image_denominators_and_unique_overlaps(self):
        annotated = self.row([Prediction("tp", 0, (0, 0, 10, 10), 0.9), Prediction("wrong", 1, (0, 0, 10, 10), 0.8),
                              Prediction("duplicate", 0, (0, 0, 6, 10), 0.7)], [GroundTruth("g", 0, (0, 0, 10, 10))])
        empty = self.row([Prediction("p", 3, (0, 0, 10, 10), 0.9)], [], name="empty")
        rows = [annotated, empty, self.row(name="empty_without_prediction")]
        reference = evaluate_rows(rows, confidence=0.25, iou=0.50)
        result, records = false_positive_analysis(reference)
        self.assertEqual((result["total_fp"], len(records)), (3, 3))
        self.assertEqual(result["groups"]["empty_label"]["percentage_images_with_fp"], 50)
        self.assertEqual(result["groups"]["empty_label"]["fp_per_image"], 0.5)
        self.assertEqual(result["duplicate_like_fp_count"], 1)
        self.assertEqual(result["unique_overlapping_pairs_involving_fp"], {"same_class": 1, "different_class": 2})
        self.assertIsNone(next(record["best_ground_truth_iou"] for record in records if record["image_group"] == "empty_label"))

    def test_fp_analysis_rejects_nonreference_operating_point(self):
        reference = evaluate_rows([self.row()], confidence=0.05, iou=0.50)
        with self.assertRaises(ErrorAnalysisError):
            false_positive_analysis(reference)

    def test_confusion_rows_are_gt_columns_are_prediction_and_reconcile(self):
        row = self.row([Prediction("wrong", 1, (0, 0, 10, 10), 0.9), Prediction("correct", 1, (20, 0, 30, 10), 0.8)],
                       [GroundTruth("g0", 0, (0, 0, 10, 10)), GroundTruth("g1", 1, (20, 0, 30, 10))])
        aware = self.reference(row)
        agnostic = evaluate_rows([row], confidence=0.25, iou=0.50, class_aware=False)
        matrix = class_confusion(agnostic, aware)
        self.assertEqual((matrix["matrix"][0][1], matrix["matrix"][1][1]), (1, 1))
        self.assertEqual(matrix["wrong_class_matches"], 1)
        self.assertEqual(matrix["additional_gt_matched_when_agnostic"], 1)
        self.assertEqual(matrix["gt_lost_when_agnostic"], 0)

    def test_numerical_tree_rejects_nonfinite_and_invalid_rates(self):
        validate_numerical_tree({"precision": 0, "recall": None, "nested": [{"iou": 1}]})
        for value in ({"precision": 1.1}, {"nested": [math.nan]}, {"score": math.inf}):
            with self.assertRaises(ErrorAnalysisError):
                validate_numerical_tree(value)

    def test_analysis_is_deterministic_and_rejects_heldout_rows(self):
        validation = [self.row([Prediction("p", 1, (0, 0, 10, 10), 0.9)], [GroundTruth("g", 0, (0, 0, 10, 10))]),
                      self.row(name="empty")]
        training = [self.row([Prediction("p", 0, (0, 0, 10, 10), 0.9)], [GroundTruth("g", 0, (0, 0, 10, 10))], split="train")]
        result, raw = analyze_exports(validation, training)
        repeated, repeated_raw = analyze_exports(list(reversed(validation)), training)
        self.assertEqual(result, repeated)
        self.assertEqual(raw, repeated_raw)
        self.assertEqual(result["validation_reference"]["fn"], result["analysis_c_fn_taxonomy"]["total_false_negatives"])
        self.assertEqual(result["training_reference"]["tp"], 1)
        with self.assertRaises(ErrorAnalysisError):
            analyze_exports([self.row(split="test")], training)
        with self.assertRaises(ErrorAnalysisError):
            evaluate_rows([self.row(split="test")], confidence=0.25, iou=0.50)


if __name__ == "__main__":
    unittest.main()
