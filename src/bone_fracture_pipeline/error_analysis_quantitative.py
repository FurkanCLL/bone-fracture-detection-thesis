"""Deterministic Stage 2 diagnostics over the preserved D prediction candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from bone_fracture_audit.audit import sha256_file
from bone_fracture_pipeline.dataset_difficulty import describe
from bone_fracture_pipeline.detection_evaluation import GroundTruth, Prediction, box_iou, detection_metrics, match_detections
from bone_fracture_pipeline.error_analysis import ErrorAnalysisError, _validate_diagnostic_rows, _write_json
from bone_fracture_pipeline.error_analysis_training import (
    export_training_predictions, read_stage1_validation, read_training_export,
)
from bone_fracture_pipeline.prepare_dataset import CLASS_NAMES


CATEGORIES = ("complete_miss", "localization_failure", "low_confidence_detection", "classification_error",
              "matching_conflict", "mixed_or_unresolved")
CONFIDENCE_THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.50)
IOU_THRESHOLDS = (0.10, 0.30, 0.50)
CONFIDENCE_BINS = (0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0000001)
SIZE_GROUPS = ("short_side_lt32", "short_side_32_to_lt64", "short_side_ge64")
REFERENCE_CONFIDENCE = 0.25
REFERENCE_IOU = 0.50
NEARBY_IOU = 0.10
LOW_SUPPORT_WARNING = 30

# These definitions are fixed before real-data categorization and copied into generated evidence.
METHOD = {
    "reference": {"confidence_at_least": 0.25, "iou_at_least": 0.50, "class_aware": True},
    "iou_sensitivity": {"confidence": [0.05, 0.25], "iou": list(IOU_THRESHOLDS), "matching_modes": ["class_aware", "class_agnostic"]},
    "confidence_sensitivity": {"confidence": list(CONFIDENCE_THRESHOLDS), "iou": 0.50},
    "matching": "Stage 1 confidence-first one-to-one greedy; score/IoU ties use stable box IDs; thresholds are inclusive.",
    "fn_taxonomy": {
        "nearby_iou_at_least": 0.10, "sufficient_iou_at_least": 0.50,
        "candidate_scope": "All retained post-NMS predictions above the original strict 0.001 export floor.",
        "strong_signals": {
            "matching_conflict": "Same-class candidate has conf >= 0.25 and IoU >= 0.50, but is assigned to another GT.",
            "low_confidence_detection": "Same-class candidate has conf < 0.25 and IoU >= 0.50.",
            "classification_error": "Wrong-class candidate has conf >= 0.25 and IoU >= 0.50.",
            "localization_failure": "Same-class candidate has conf >= 0.25 and 0.10 <= IoU < 0.50.",
        },
        "primary_precedence": [
            "Matching conflict takes precedence when an otherwise reference-eligible candidate is already claimed.",
            "Two or more remaining strong mechanisms produce mixed_or_unresolved.",
            "Exactly one remaining strong mechanism determines the primary category.",
            "No candidate at IoU >= 0.10 produces complete_miss within the retained export only.",
            "Otherwise use mixed_or_unresolved: nearby candidates require multiple corrections or have weak evidence.",
        ],
        "nonexclusive_flags": "Preserve all strong mechanisms plus all eight class x confidence x overlap combinations for nearby candidates.",
        "causal_status": "Operational candidate evidence, not a unique causal diagnosis or proof of a recoverable TP.",
    },
    "size_groups": {"measure": "Original box shorter side * min(640/image_width, 640/image_height)",
                    "boundaries_px": [32, 64], "intervals": ["[0,32)", "[32,64)", "[64,infinity)"],
                    "interpretation": "Nominal 640-pixel letterbox scale; descriptive groups, not a tuned cutoff."},
    "class_confusion": "Rows=GT class, columns=predicted class, using one-to-one class-agnostic reference matches only.",
    "confidence_quality": "Per-candidate maximum GT IoU on annotated images; no one-to-one constraint, so these rates are not precision/AP.",
    "fp_overlap": "Each FP once for same-class overlap >= 0.50 with a matched TP; unique unordered retained-prediction pairs for overlap counts.",
    "percentages": "All-FN percentages divide by reference FN; class taxonomy percentages divide by class FN; empty-image rates divide by all empty images.",
    "empty_distributions": "No observations -> count 0 and null descriptive statistics; zero metric denominators follow Stage 1's zero convention.",
    "low_support_warning": "Flag descriptive groups with fewer than 30 observations; this is a caution marker, not a statistical power test.",
    "interpretation_limits": [
        "IoU 0.10 represents permissive approximate overlap, not clinically adequate localization.",
        "Native AP and fixed-threshold diagnostic metrics have different assignment/aggregation semantics.",
        "Class-agnostic matching retains candidates generated with class-aware NMS; it is not class-agnostic AP.",
        "Empty labels mean no annotated region, not clinically confirmed fracture absence.",
        "No deployment threshold is selected and no model or hyperparameter is tuned.",
    ],
}


def percentage(count: int, total: int) -> float | None:
    return 100.0 * count / total if total else None


def detection_objects(row: Mapping[str, object]) -> tuple[tuple[Prediction, ...], tuple[GroundTruth, ...]]:
    predictions = tuple(Prediction(item["box_id"], item["class_id"], tuple(item["xyxy"]), item["confidence"])
                        for item in sorted(row["predictions"], key=lambda item: item["box_id"]))
    targets = tuple(GroundTruth(item["box_id"], item["class_id"], tuple(item["xyxy"]))
                    for item in sorted(row["ground_truth"], key=lambda item: item["box_id"]))
    return predictions, targets


def evaluate_rows(rows: Sequence[Mapping[str, object]], *, confidence: float, iou: float, class_aware: bool = True) -> list[dict[str, object]]:
    evaluations = []
    for row in rows:
        if row["split"] not in ("train", "valid"):
            raise ErrorAnalysisError("Held-out rows cannot enter quantitative evaluation.")
        predictions, targets = detection_objects(row)
        result = match_detections(predictions, targets, confidence_threshold=confidence, iou_threshold=iou, class_aware=class_aware)
        evaluations.append({"row": row, "predictions": predictions, "targets": targets, "result": result,
                            "confidence": confidence, "iou": iou, "class_aware": class_aware})
    return evaluations


def aggregate_evaluations(evaluations: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    tp = sum(len(item["result"].matches) for item in evaluations)
    fp = sum(len(item["result"].false_positive_ids) for item in evaluations)
    fn = sum(len(item["result"].false_negative_ids) for item in evaluations)
    targets = sum(len(item["targets"]) for item in evaluations)
    retained = sum(len(item["predictions"]) - len(item["result"].filtered_prediction_ids) for item in evaluations)
    if tp + fn != targets or tp + fp != retained:
        raise ErrorAnalysisError("Operating-point counts do not reconcile with their source populations.")
    return {**detection_metrics(tp, fp, fn), "matched_ground_truth_count": tp,
            "ground_truth_count": targets, "retained_prediction_count": retained, "images": len(evaluations)}


def sensitivity(rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    iou_rows, confidence_rows = [], []
    for class_aware in (True, False):
        for confidence in (0.05, 0.25):
            for iou in IOU_THRESHOLDS:
                metrics = aggregate_evaluations(evaluate_rows(rows, confidence=confidence, iou=iou, class_aware=class_aware))
                iou_rows.append({"confidence": confidence, "iou": iou, "class_aware": class_aware, **metrics})
        for confidence in CONFIDENCE_THRESHOLDS:
            metrics = aggregate_evaluations(evaluate_rows(rows, confidence=confidence, iou=0.50, class_aware=class_aware))
            confidence_rows.append({"confidence": confidence, "iou": 0.50, "class_aware": class_aware, **metrics})
    return iou_rows, confidence_rows


# Keep ambiguous candidate evidence visible instead of forcing every FN into one simple mechanism.
def categorize_false_negatives(reference: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    records = []
    for image in reference:
        if (image["confidence"], image["iou"], image["class_aware"]) != (REFERENCE_CONFIDENCE, REFERENCE_IOU, True):
            raise ErrorAnalysisError("FN taxonomy requires the fixed class-aware reference operating point.")
        row, result = image["row"], image["result"]
        assigned = {match.prediction_id: match.ground_truth_id for match in result.matches}
        targets = {target.box_id: target for target in image["targets"]}
        for target_id in result.false_negative_ids:
            target = targets[target_id]
            candidates = [(prediction, box_iou(prediction.xyxy, target.xyxy)) for prediction in image["predictions"]]
            nearby = [(prediction, overlap) for prediction, overlap in candidates if overlap >= NEARBY_IOU]
            combinations = {f"{label}_{confidence}_{geometry}": [] for label in ("correct_class", "wrong_class")
                            for confidence in ("high_confidence", "low_confidence") for geometry in ("sufficient_overlap", "poor_overlap")}
            for prediction, overlap in nearby:
                label = "correct_class" if prediction.class_id == target.class_id else "wrong_class"
                confidence = "high_confidence" if prediction.confidence >= REFERENCE_CONFIDENCE else "low_confidence"
                geometry = "sufficient_overlap" if overlap >= REFERENCE_IOU else "poor_overlap"
                combinations[f"{label}_{confidence}_{geometry}"].append(prediction.box_id)
            eligible = combinations["correct_class_high_confidence_sufficient_overlap"]
            if any(identifier not in assigned for identifier in eligible):
                raise ErrorAnalysisError("An unmatched GT has an eligible unclaimed reference prediction.")
            flags = {
                "matching_conflict": bool(eligible),
                "low_confidence_detection": bool(combinations["correct_class_low_confidence_sufficient_overlap"]),
                "classification_error": bool(combinations["wrong_class_high_confidence_sufficient_overlap"]),
                "localization_failure": bool(combinations["correct_class_high_confidence_poor_overlap"]),
                "retained_nearby_candidate": bool(nearby),
                "jointly_imperfect_candidate": any(bool(identifiers) for key, identifiers in combinations.items()
                                                   if key in ("wrong_class_low_confidence_sufficient_overlap", "wrong_class_high_confidence_poor_overlap",
                                                              "wrong_class_low_confidence_poor_overlap", "correct_class_low_confidence_poor_overlap")),
            }
            strong = [name for name in ("low_confidence_detection", "classification_error", "localization_failure") if flags[name]]
            if flags["matching_conflict"]:
                category = "matching_conflict"
            elif len(strong) > 1:
                category = "mixed_or_unresolved"
            elif strong:
                category = strong[0]
            else:
                category = "mixed_or_unresolved" if nearby else "complete_miss"
            best = min(nearby, key=lambda pair: (-pair[1], -pair[0].confidence, pair[0].box_id)) if nearby else None
            records.append({
                "sample_id": row["sample_id"], "class_id": target.class_id, "ground_truth_id": target_id,
                "primary_category": category, "flags": flags,
                "candidate_combination_flags": {key: bool(value) for key, value in combinations.items()},
                "candidate_ids_by_combination": {key: sorted(value) for key, value in combinations.items()},
                "nearby_retained_candidate_count": len(nearby),
                "best_retained_iou": max((overlap for _, overlap in candidates), default=0.0),
                "best_nearby_candidate": {"prediction_id": best[0].box_id, "class_id": best[0].class_id,
                                          "confidence": best[0].confidence, "iou": best[1]} if best else None,
                "competing_assignments": [{"prediction_id": identifier, "assigned_ground_truth_id": assigned[identifier]} for identifier in sorted(eligible)],
            })
    expected = sum(len(image["result"].false_negative_ids) for image in reference)
    identities = {(record["sample_id"], record["ground_truth_id"]) for record in records}
    if len(records) != expected or len(identities) != expected:
        raise ErrorAnalysisError("FN taxonomy is not exhaustive and one-to-one.")
    return sorted(records, key=lambda record: (record["sample_id"], record["ground_truth_id"]))


def summarize_taxonomy(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts = Counter(record["primary_category"] for record in records)
    class_rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        subset = [record for record in records if record["class_id"] == class_id]
        by_category = Counter(record["primary_category"] for record in subset)
        class_rows.append({"class_id": class_id, "class_name": name, "fn": len(subset),
                           "categories": {category: {"count": by_category[category], "percentage_of_class_fn": percentage(by_category[category], len(subset))} for category in CATEGORIES}})
    if set(counts) - set(CATEGORIES) or sum(counts.values()) != len(records):
        raise ErrorAnalysisError("FN primary categories failed reconciliation.")
    return {"total_false_negatives": len(records),
            "primary_categories": {category: {"count": counts[category], "percentage_of_fn": percentage(counts[category], len(records))} for category in CATEGORIES},
            "nonexclusive_flag_counts": dict(sorted(Counter(name for record in records for name, enabled in record["flags"].items() if enabled).items())),
            "candidate_combination_flag_counts": dict(sorted(Counter(name for record in records for name, enabled in record["candidate_combination_flags"].items() if enabled).items())),
            "class_rows": class_rows, "primary_counts_exhaustive_and_nonoverlapping": True}


def size_group(target: GroundTruth, row: Mapping[str, object]) -> tuple[str, float]:
    short_side = min(target.xyxy[2] - target.xyxy[0], target.xyxy[3] - target.xyxy[1]) * min(640 / row["width"], 640 / row["height"])
    group = SIZE_GROUPS[0] if short_side < 32 else SIZE_GROUPS[1] if short_side < 64 else SIZE_GROUPS[2]
    return group, short_side


def class_and_size_analysis(reference: Sequence[Mapping[str, object]], taxonomy: Sequence[Mapping[str, object]] | None = None) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    counts = [{"tp": 0, "fp": 0, "fn": 0} for _ in CLASS_NAMES]
    overlaps = [[] for _ in CLASS_NAMES]
    size_counts = {(class_id, group): Counter() for class_id in range(len(CLASS_NAMES)) for group in SIZE_GROUPS}
    for image in reference:
        predictions = {prediction.box_id: prediction for prediction in image["predictions"]}
        targets = {target.box_id: target for target in image["targets"]}
        matched = {match.ground_truth_id for match in image["result"].matches}
        for match in image["result"].matches:
            target = targets[match.ground_truth_id]
            if target.class_id != predictions[match.prediction_id].class_id:
                raise ErrorAnalysisError("Class-wise precision/recall require class-aware reference matches.")
            counts[target.class_id]["tp"] += 1
            overlaps[target.class_id].append(match.iou)
        for identifier in image["result"].false_positive_ids:
            counts[predictions[identifier].class_id]["fp"] += 1
        for identifier in image["result"].false_negative_ids:
            counts[targets[identifier].class_id]["fn"] += 1
        for target in targets.values():
            group, _ = size_group(target, image["row"])
            size_counts[target.class_id, group]["support"] += 1
            size_counts[target.class_id, group]["tp" if target.box_id in matched else "fn"] += 1
    class_rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        values = counts[class_id]
        support = values["tp"] + values["fn"]
        class_rows.append({"class_id": class_id, "class_name": name, "support": support,
                           "support_percentage": percentage(support, sum(item["tp"] + item["fn"] for item in counts)),
                           **detection_metrics(**values), "matched_iou": describe(overlaps[class_id]),
                           "low_support_warning": support < LOW_SUPPORT_WARNING,
                           "matched_iou_low_support_warning": len(overlaps[class_id]) < LOW_SUPPORT_WARNING,
                           "fn_primary_counts": {category: sum(record["class_id"] == class_id and record["primary_category"] == category for record in taxonomy) for category in CATEGORIES} if taxonomy is not None else None})
    size_rows = []
    for class_id in (None, *range(len(CLASS_NAMES))):
        for group in SIZE_GROUPS:
            values = sum((size_counts[index, group] for index in range(len(CLASS_NAMES))), Counter()) if class_id is None else size_counts[class_id, group]
            support = values["support"]
            size_rows.append({"class_id": class_id, "class_name": "All classes" if class_id is None else CLASS_NAMES[class_id],
                              "size_group": group, "support": support, "tp": values["tp"], "fn": values["fn"],
                              "recall": values["tp"] / support if support else None,
                              "low_support_warning": support < LOW_SUPPORT_WARNING})
    global_counts = aggregate_evaluations(reference)
    if any(sum(row[key] for row in class_rows) != global_counts[key] for key in ("tp", "fp", "fn")):
        raise ErrorAnalysisError("Class-wise totals do not equal global counts.")
    return class_rows, size_rows


def class_confusion(agnostic_reference: Sequence[Mapping[str, object]], aware_reference: Sequence[Mapping[str, object]]) -> dict[str, object]:
    matrix = [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES]
    agnostic_targets = set()
    aware_targets = {(image["row"]["sample_id"], match.ground_truth_id) for image in aware_reference for match in image["result"].matches}
    for image in agnostic_reference:
        predictions = {prediction.box_id: prediction for prediction in image["predictions"]}
        targets = {target.box_id: target for target in image["targets"]}
        for match in image["result"].matches:
            matrix[targets[match.ground_truth_id].class_id][predictions[match.prediction_id].class_id] += 1
            agnostic_targets.add((image["row"]["sample_id"], match.ground_truth_id))
    total = sum(map(sum, matrix))
    correct = sum(matrix[index][index] for index in range(len(CLASS_NAMES)))
    if total != aggregate_evaluations(agnostic_reference)["tp"]:
        raise ErrorAnalysisError("Class confusion does not reconcile with agnostic matched GT counts.")
    return {"rows": "ground_truth_class", "columns": "predicted_class", "matrix": matrix,
            "geometric_matches": total, "correct_class_matches": correct, "wrong_class_matches": total - correct,
            "wrong_class_percentage_of_geometric_matches": percentage(total - correct, total),
            "additional_gt_matched_when_agnostic": len(agnostic_targets - aware_targets),
            "gt_lost_when_agnostic": len(aware_targets - agnostic_targets),
            "gt_matched_in_both_modes": len(aware_targets & agnostic_targets),
            "limitation": "Removing class eligibility can rearrange greedy assignments; the TP difference alone is not a pure classification-error count."}


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or any(not math.isfinite(value) for value in (*first, *second)):
        raise ErrorAnalysisError("Correlation requires paired finite observations.")
    if len(first) < 3 or len(set(first)) < 2 or len(set(second)) < 2:
        return None

    def average_ranks(values):
        ordered = sorted(range(len(values)), key=lambda index: values[index])
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and values[ordered[stop]] == values[ordered[start]]:
                stop += 1
            ranks[ordered[start:stop]] = (start + stop - 1) / 2 + 1
            start = stop
        return ranks

    result = float(np.corrcoef(average_ranks(first), average_ranks(second))[0, 1])
    if not math.isfinite(result):
        raise ErrorAnalysisError("Rank correlation produced a non-finite value.")
    return max(-1.0, min(1.0, result))


# Candidate overlap quality is distinct from one-to-one TP precision and score calibration.
def confidence_localization(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    records = []
    excluded_empty_candidates = 0
    for row in rows:
        predictions, targets = detection_objects(row)
        if not targets:
            excluded_empty_candidates += len(predictions)
            continue
        for prediction in predictions:
            records.append({"confidence": prediction.confidence,
                            "best_any_class_iou": max((box_iou(prediction.xyxy, target.xyxy) for target in targets), default=0.0),
                            "best_correct_class_iou": max((box_iou(prediction.xyxy, target.xyxy) for target in targets if target.class_id == prediction.class_id), default=0.0)})
    bins = []
    for low, high in zip(CONFIDENCE_BINS[:-1], CONFIDENCE_BINS[1:]):
        subset = [record for record in records if low <= record["confidence"] < high]
        bins.append({"confidence_lower_inclusive": low, "confidence_upper": min(high, 1.0),
                     "upper_inclusive": high > 1.0, "candidates": len(subset),
                     "confidence": describe([record["confidence"] for record in subset]),
                     "best_any_class_iou": describe([record["best_any_class_iou"] for record in subset]),
                     "best_correct_class_iou": describe([record["best_correct_class_iou"] for record in subset]),
                     "any_class_iou_at_least_0_5_count": sum(record["best_any_class_iou"] >= 0.5 for record in subset),
                     "correct_class_iou_at_least_0_5_count": sum(record["best_correct_class_iou"] >= 0.5 for record in subset)})
    if sum(row["candidates"] for row in bins) != len(records):
        raise ErrorAnalysisError("Confidence bins do not partition the retained annotated-image candidates.")
    confidence = [record["confidence"] for record in records]
    return {"annotated_image_candidate_count": len(records), "excluded_empty_image_candidate_count": excluded_empty_candidates,
            "bins": bins, "spearman_confidence_vs_best_any_class_iou": spearman_correlation(confidence, [record["best_any_class_iou"] for record in records]),
            "spearman_confidence_vs_best_correct_class_iou": spearman_correlation(confidence, [record["best_correct_class_iou"] for record in records]),
            "interpretation": "Descriptive candidate-level association, not independent samples, probability calibration, significance, one-to-one precision, or causal evidence."}


def reference_distributions(reference: Sequence[Mapping[str, object]]) -> dict[str, object]:
    matched_confidence, matched_iou, fp_confidence = [], [], []
    for image in reference:
        predictions = {prediction.box_id: prediction for prediction in image["predictions"]}
        for match in image["result"].matches:
            matched_confidence.append(predictions[match.prediction_id].confidence)
            matched_iou.append(match.iou)
        fp_confidence.extend(predictions[identifier].confidence for identifier in image["result"].false_positive_ids)
    return {"matched_prediction_confidence": describe(matched_confidence), "matched_iou": describe(matched_iou),
            "false_positive_confidence": describe(fp_confidence),
            "selection_note": "TP IoU and confidence distributions are conditional on meeting the reference thresholds."}


def false_positive_analysis(reference: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    if any((image["confidence"], image["iou"], image["class_aware"]) != (REFERENCE_CONFIDENCE, REFERENCE_IOU, True)
           for image in reference):
        raise ErrorAnalysisError("FP analysis requires the fixed class-aware reference operating point.")
    groups = {name: {"images": 0, "images_with_fp": 0, "fp": 0, "by_class": [0] * len(CLASS_NAMES), "scores": []}
              for name in ("annotated", "empty_label")}
    records = []
    pairs = Counter()
    for image in reference:
        group = "annotated" if image["targets"] else "empty_label"
        values = groups[group]
        values["images"] += 1
        values["images_with_fp"] += bool(image["result"].false_positive_ids)
        values["fp"] += len(image["result"].false_positive_ids)
        fp_ids = set(image["result"].false_positive_ids)
        matched_ids = {match.prediction_id for match in image["result"].matches}
        retained = [prediction for prediction in image["predictions"] if prediction.confidence >= REFERENCE_CONFIDENCE]
        for prediction in retained:
            if prediction.box_id not in fp_ids:
                continue
            values["by_class"][prediction.class_id] += 1
            values["scores"].append(prediction.confidence)
            overlaps = [(other, box_iou(prediction.xyxy, other.xyxy)) for other in retained if other.box_id != prediction.box_id]
            duplicate_like = any(other.box_id in matched_ids and other.class_id == prediction.class_id and overlap >= 0.5 for other, overlap in overlaps)
            records.append({"sample_id": image["row"]["sample_id"], "image_group": group, "prediction_id": prediction.box_id,
                            "predicted_class_id": prediction.class_id, "confidence": prediction.confidence,
                            "best_ground_truth_iou": max((box_iou(prediction.xyxy, target.xyxy) for target in image["targets"]), default=None),
                            "same_class_overlap_with_matched_tp": duplicate_like,
                            "same_class_overlapping_other_prediction": any(other.class_id == prediction.class_id and overlap >= 0.5 for other, overlap in overlaps),
                            "wrong_class_overlapping_other_prediction": any(other.class_id != prediction.class_id and overlap >= 0.5 for other, overlap in overlaps)})
        # Unique unordered pairs prevent counting each overlap twice.
        for index, first in enumerate(retained):
            for second in retained[index + 1:]:
                if first.box_id not in fp_ids and second.box_id not in fp_ids:
                    continue
                if box_iou(first.xyxy, second.xyxy) >= 0.5:
                    pairs["same_class" if first.class_id == second.class_id else "different_class"] += 1
    total = sum(values["fp"] for values in groups.values())
    if total != aggregate_evaluations(reference)["fp"] or len(records) != total:
        raise ErrorAnalysisError("FP image groups do not reconcile with the reference matcher.")
    output_groups = {}
    for name, values in groups.items():
        output_groups[name] = {key: values[key] for key in ("images", "images_with_fp", "fp")}
        output_groups[name].update({"percentage_of_total_fp": percentage(values["fp"], total),
                                   "percentage_images_with_fp": percentage(values["images_with_fp"], values["images"]),
                                   "fp_per_image": values["fp"] / values["images"] if values["images"] else None,
                                   "fp_per_image_with_fp": values["fp"] / values["images_with_fp"] if values["images_with_fp"] else None,
                                   "by_predicted_class": [{"class_id": class_id, "class_name": CLASS_NAMES[class_id],
                                                           "fp": count, "percentage_of_group_fp": percentage(count, values["fp"])} for class_id, count in enumerate(values["by_class"])],
                                   "confidence": describe(values["scores"])})
    return {"total_fp": total, "groups": output_groups,
            "by_predicted_class": [{"class_id": class_id, "class_name": name,
                                    "fp": sum(record["predicted_class_id"] == class_id for record in records),
                                    "percentage_of_total_fp": percentage(sum(record["predicted_class_id"] == class_id for record in records), total),
                                    "confidence": describe([record["confidence"] for record in records if record["predicted_class_id"] == class_id])} for class_id, name in enumerate(CLASS_NAMES)],
            "confidence": describe([record["confidence"] for record in records]),
            "duplicate_like_fp_count": sum(record["same_class_overlap_with_matched_tp"] for record in records),
            "fp_with_same_class_overlap_count": sum(record["same_class_overlapping_other_prediction"] for record in records),
            "fp_with_cross_class_overlap_count": sum(record["wrong_class_overlapping_other_prediction"] for record in records),
            "unique_overlapping_pairs_involving_fp": {key: pairs[key] for key in ("same_class", "different_class")},
            "overlap_iou_at_least": 0.5,
            "interpretation": "Annotation-based FP; empty-label images are not medically verified negative images."}, records


def validate_numerical_tree(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            validate_numerical_tree(child)
            if key in ("precision", "recall", "f1", "iou") and isinstance(child, (int, float)) and not 0 <= child <= 1:
                raise ErrorAnalysisError(f"Diagnostic rate {key} is outside [0,1].")
    elif isinstance(value, (list, tuple)):
        for child in value:
            validate_numerical_tree(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ErrorAnalysisError("A quantitative output is non-finite.")


def analyze_exports(validation: Sequence[Mapping[str, object]], training: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    validation_counts = _validate_diagnostic_rows(validation, split="valid")
    training_counts = _validate_diagnostic_rows(training, split="train")
    validation = sorted(validation, key=lambda row: row["sample_id"])
    training = sorted(training, key=lambda row: row["sample_id"])
    reference = evaluate_rows(validation, confidence=0.25, iou=0.50)
    agnostic = evaluate_rows(validation, confidence=0.25, iou=0.50, class_aware=False)
    train_reference = evaluate_rows(training, confidence=0.25, iou=0.50)
    taxonomy = categorize_false_negatives(reference)
    class_rows, size_rows = class_and_size_analysis(reference, taxonomy)
    train_class_rows, train_size_rows = class_and_size_analysis(train_reference)
    iou_rows, confidence_rows = sensitivity(validation)
    fp_summary, fp_records = false_positive_analysis(reference)
    generalization = []
    for split, rows in (("train", training), ("valid", validation)):
        for confidence in (0.05, 0.25):
            generalization.append({"split": split, "confidence": confidence, "iou": 0.50, "class_aware": True,
                                   **aggregate_evaluations(evaluate_rows(rows, confidence=confidence, iou=0.50))})
    agnostic_comparison = []
    for confidence in CONFIDENCE_THRESHOLDS:
        aware_row = next(row for row in confidence_rows if row["confidence"] == confidence and row["class_aware"])
        agnostic_row = next(row for row in confidence_rows if row["confidence"] == confidence and not row["class_aware"])
        agnostic_comparison.append({"confidence": confidence, "iou": 0.50, "aware": aware_row, "agnostic": agnostic_row,
                                    "tp_difference": agnostic_row["tp"] - aware_row["tp"],
                                    "recall_difference_percentage_points": 100 * (agnostic_row["recall"] - aware_row["recall"])})
    train_names = {Path(row["image"]).name for row in training}
    validation_names = {Path(row["image"]).name for row in validation}
    train_hashes = {row["image_sha256"] for row in training}
    validation_hashes = {row["image_sha256"] for row in validation}
    result = {
        "method": METHOD, "split_counts": {"train": training_counts, "valid": validation_counts},
        "validation_reference": aggregate_evaluations(reference), "training_reference": aggregate_evaluations(train_reference),
        "analysis_a_iou_sensitivity": iou_rows,
        "analysis_b_confidence_sensitivity": {"operating_points": confidence_rows, "candidate_localization": confidence_localization(validation)},
        "analysis_c_fn_taxonomy": summarize_taxonomy(taxonomy),
        "analysis_d_classes": {"class_rows": class_rows, "size_rows": size_rows,
                               "class_confusion": class_confusion(agnostic, reference)},
        "analysis_e_false_positives": fp_summary,
        "analysis_f_generalization": {"operating_points": generalization, "train_class_rows": train_class_rows,
                                      "train_size_rows": train_size_rows,
                                      "train_reference_distributions": reference_distributions(train_reference),
                                      "validation_reference_distributions": reference_distributions(reference),
                                      "train_candidate_localization": confidence_localization(training),
                                      "class_agnostic_comparison": agnostic_comparison},
        "split_identity_checks": {"stable_sample_ids_disjoint": not ({row["sample_id"] for row in training} & {row["sample_id"] for row in validation}),
                                  "shared_image_filenames": len(train_names & validation_names),
                                  "shared_encoded_image_hashes": len(train_hashes & validation_hashes),
                                  "note": "Fixed split membership is preserved; exact encoded-byte intersection is not a patient/study or near-duplicate audit."},
    }
    if result["analysis_c_fn_taxonomy"]["total_false_negatives"] != result["validation_reference"]["fn"]:
        raise ErrorAnalysisError("Categorized FN total differs from the reference matcher.")
    if sum(row["support"] for row in class_rows) != validation_counts["annotations"]:
        raise ErrorAnalysisError("Class support does not equal validation GT count.")
    if sum(row["support"] for row in size_rows if row["class_id"] is None) != validation_counts["annotations"]:
        raise ErrorAnalysisError("Box size groups do not partition validation GT.")
    validate_numerical_tree(result)
    raw = {"false_negatives": taxonomy, "false_positives": fp_records,
           "reference_images": [{"sample_id": image["row"]["sample_id"], "split": image["row"]["split"],
                                  **image["result"].metrics,
                                  "matches": [{"prediction_id": match.prediction_id, "ground_truth_id": match.ground_truth_id, "iou": match.iou} for match in image["result"].matches],
                                  "false_positive_ids": list(image["result"].false_positive_ids),
                                  "false_negative_ids": list(image["result"].false_negative_ids)} for image in reference]}
    return result, raw


def verify_protected_files(project_root: Path, baseline_path: Path) -> dict[str, object]:
    # Persist the full baseline on disk; a long terminal listing can truncate hashes.
    if not baseline_path.exists():
        names = ("docs/evidence/phase2f/cd_pairing_validation.json", "docs/evidence/phase2f/environment_freeze.json",
                 "docs/evidence/phase2f/experiment_matrix_validation.json", "docs/evidence/error_analysis/README.md",
                 "docs/evidence/error_analysis/stage1_validation.json", "requirements.txt", "pyproject.toml")
        paths = [project_root / name for name in names]
        paths += list((project_root / "configs").rglob("*"))
        paths += list((project_root / "outputs/training/official").rglob("*"))
        stage1_path = project_root / "docs/evidence/error_analysis/stage1_validation.json"
        if stage1_path.is_file():
            stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
            export = (project_root / stage1["prediction_export"]["path"]).resolve()
            if not export.is_relative_to(project_root / "outputs/error_analysis/stage1"):
                raise ErrorAnalysisError("Protected Stage 1 export must remain in its own output directory.")
            paths += [export, export.parent / "stage1_summary.json"]
        _write_json(baseline_path, {path.relative_to(project_root).as_posix(): sha256_file(path) for path in paths if path.is_file()})
    expected = json.loads(baseline_path.read_text(encoding="utf-8"))
    changed = [name for name, digest in expected.items() if sha256_file(project_root / name) != digest]
    if changed:
        raise ErrorAnalysisError(f"Protected input/evidence content changed: {changed}")
    return {"protected_file_count": len(expected), "all_hashes_unchanged": True,
            "baseline_path": baseline_path.relative_to(project_root).as_posix(), "baseline_sha256": sha256_file(baseline_path)}


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def run_stage2(project_root: Path, output: Path, *, training_export: Path | None = None, publish: bool = True) -> dict[str, object]:
    project_root, output = project_root.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(project_root / "outputs/error_analysis/stage2"):
        raise ErrorAnalysisError("Use a fresh Stage 2 output directory; existing diagnostics are preserved.")
    base = project_root / "outputs/error_analysis/stage2"
    base.mkdir(parents=True, exist_ok=True)
    protection_path = base / "protection_before.json"
    verify_protected_files(project_root, protection_path)
    method_path = base / "method_pre_execution.json"
    if method_path.exists():
        if json.loads(method_path.read_text(encoding="utf-8"))["method"] != METHOD:
            raise ErrorAnalysisError("The method changed after pre-execution freezing; do not recategorize silently.")
    else:
        _write_json(method_path, {"recorded_at_utc": datetime.now(UTC).isoformat(), "method": METHOD})
    validation, stage1 = read_stage1_validation(project_root)
    output.mkdir(parents=True)
    if training_export is None:
        training_export = output / "train_evaluation"
        export_training_predictions(project_root, training_export, stage1)
    training, train_metadata = read_training_export(project_root, training_export, stage1)
    result, raw = analyze_exports(validation, training)
    for split, expected in (("valid", (348, 204, 175)), ("train", (1211, 698, 607))):
        counts = result["split_counts"][split]
        if (counts["images"], counts["annotations"], counts["empty_label_images"]) != expected:
            raise ErrorAnalysisError("Real-data split counts differ from their approved contracts.")
    _write_json(output / "analysis_results.json", result)
    for name, rows in raw.items():
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    tables = {"iou_sensitivity": result["analysis_a_iou_sensitivity"],
              "confidence_sensitivity": result["analysis_b_confidence_sensitivity"]["operating_points"],
              "validation_classes": result["analysis_d_classes"]["class_rows"],
              "validation_sizes": result["analysis_d_classes"]["size_rows"],
              "fn_categories": [{"category": key, **value} for key, value in result["analysis_c_fn_taxonomy"]["primary_categories"].items()],
              "fp_classes": result["analysis_e_false_positives"]["by_predicted_class"],
              "generalization": result["analysis_f_generalization"]["operating_points"],
              "training_classes": result["analysis_f_generalization"]["train_class_rows"]}
    for name, rows in tables.items():
        write_csv(output / "tables" / f"{name}.csv", rows)
    preservation = verify_protected_files(project_root, protection_path)
    code_paths = [project_root / "src/bone_fracture_pipeline" / name for name in (
        "error_analysis.py", "detection_evaluation.py", "error_analysis_training.py", "error_analysis_quantitative.py",
        "error_analysis_figures.py", "error_analysis_report.py")]
    code_paths += [project_root / "tests/test_error_analysis_quantitative.py"]
    record = {"schema_version": 1, "stage": 2, "experiment": "D", "created_at_utc": datetime.now(UTC).isoformat(),
              "status": "quantitative_checks_passed_with_documented_stage1_discrepancy",
              "stage3_started": False, "experiment_f_started": False,
              "checkpoint": stage1["checkpoint"], "stage1_official_reproduction": stage1["official_validation"],
              "stage1_validation_passed": False, "proceeding_despite_discrepancy_explicitly_authorized": True,
              "validation_export": stage1["prediction_export"],
              "training_export": {key: value for key, value in train_metadata.items() if key != "native_training_split_metrics"},
              "dataset": stage1["dataset"], "method_pre_execution_record": {
                  "path": method_path.relative_to(project_root).as_posix(), "sha256": sha256_file(method_path),
                  "recorded_at_utc": json.loads(method_path.read_text(encoding="utf-8"))["recorded_at_utc"]},
              "analysis": result, "analysis_results_path": (output / "analysis_results.json").relative_to(project_root).as_posix(),
              "analysis_results_sha256": sha256_file(output / "analysis_results.json"), "preservation": preservation,
              "verification_record": "docs/evidence/error_analysis/stage2_verification.json",
              "source_sha256": {path.relative_to(project_root).as_posix(): sha256_file(path) for path in code_paths if path.is_file()},
              "limitations": METHOD["interpretation_limits"] + [
                  "Stage 1 mAP50 reproduction remains failed at the unchanged 0.00005 bound; official and diagnostic results remain separate.",
                  "FP16 serialization is a plausible explanation for the small discrepancy, not a proven cause.",
                  "No response below confidence 0.001 or suppressed by NMS/max_det is observable in these exports.",
                  "Complete miss means no retained candidate at IoU >= 0.10, not true absence of model-head response.",
                  "FN primary categories describe candidate evidence; nonexclusive signals can coexist and low-confidence candidates may compete.",
                  "One seed, limited class support, unadjudicated annotations, and unknown patient/study independence restrict inference.",
                  "Confidence/localization associations and train/validation differences do not isolate a unique causal mechanism.",
                  "The optional D-versus-E comparison is omitted because E requires separate prediction preparation and a different target definition.",
              ]}
    _write_json(output / "stage2_run.json", record)
    if publish:
        from bone_fracture_pipeline.error_analysis_figures import generate_figures
        from bone_fracture_pipeline.error_analysis_report import write_report

        figures = generate_figures(result, project_root / "docs/figures/error_analysis")
        record["figures"] = [{"path": path.relative_to(project_root).as_posix(), "sha256": sha256_file(path)} for path in figures]
        _write_json(project_root / "docs/evidence/error_analysis/stage2_quantitative.json", record)
        write_report(record, project_root / "docs/evidence/error_analysis/STAGE2_REPORT.md")
        _write_json(output / "stage2_run.json", record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run D's six quantitative diagnostic groups over train/validation only.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/error_analysis/stage2/D_seed42"))
    parser.add_argument("--reuse-training", type=Path, help="Reuse a verified Stage 2 train export; no new inference.")
    parser.add_argument("--no-publish", action="store_true", help="Write ignored numeric outputs without updating canonical report/figures.")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    train = args.reuse_training if args.reuse_training is None or args.reuse_training.is_absolute() else root / args.reuse_training
    record = run_stage2(root, output, training_export=train, publish=not args.no_publish)
    print(json.dumps({"status": record["status"], "validation_reference": record["analysis"]["validation_reference"],
                      "training_reference": record["analysis"]["training_reference"],
                      "fn_categories": record["analysis"]["analysis_c_fn_taxonomy"]["primary_categories"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
