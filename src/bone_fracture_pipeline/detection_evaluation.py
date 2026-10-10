from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GroundTruth:
    box_id: str
    class_id: int
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    box_id: str
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class Match:
    prediction_id: str
    ground_truth_id: str
    iou: float


@dataclass(frozen=True)
class MatchResult:
    matches: tuple[Match, ...]
    false_positive_ids: tuple[str, ...]
    false_negative_ids: tuple[str, ...]
    filtered_prediction_ids: tuple[str, ...]

    @property
    def metrics(self) -> dict[str, float | int]:
        return detection_metrics(len(self.matches), len(self.false_positive_ids), len(self.false_negative_ids))


def _validate_box(box: Sequence[float]) -> None:
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError("A box must contain four finite xyxy coordinates.")
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError("Box corners must be ordered left/top to right/bottom.")


# Zero-area and touching boxes have zero overlap; reversed or non-finite boxes are errors.
def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    _validate_box(first)
    _validate_box(second)
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    if not math.isfinite(union) or not math.isfinite(intersection):
        raise ValueError("Box area overflowed; use original-image pixel coordinates.")
    return intersection / union if union > 0 else 0.0


# These are fixed operating-point metrics, not confidence-integrated average precision.
def detection_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    if any(type(value) is not int or value < 0 for value in (tp, fp, fn)):
        raise ValueError("TP, FP, and FN must be non-negative integers.")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _validate_detections(boxes: Sequence[GroundTruth | Prediction]) -> None:
    identifiers = set()
    for box in boxes:
        if not isinstance(box.box_id, str) or not box.box_id or box.box_id in identifiers:
            raise ValueError("Box IDs must be non-empty and unique within each image and box type.")
        identifiers.add(box.box_id)
        if type(box.class_id) is not int or box.class_id < 0:
            raise ValueError("Class IDs must be non-negative integers.")
        _validate_box(box.xyxy)
        if isinstance(box, Prediction) and not (math.isfinite(box.confidence) and 0 <= box.confidence <= 1):
            raise ValueError("Prediction confidence must be finite and between zero and one.")


# Confidence-first greedy assignment gives each prediction and each target at most one match.
def match_detections(
    predictions: Sequence[Prediction],
    ground_truth: Sequence[GroundTruth],
    *,
    confidence_threshold: float,
    iou_threshold: float,
    class_aware: bool = True,
) -> MatchResult:
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be finite and in [0, 1].")
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be finite and in (0, 1].")
    if type(class_aware) is not bool:
        raise ValueError("class_aware must be boolean.")
    _validate_detections(predictions)
    _validate_detections(ground_truth)

    # Stable IDs settle score and IoU ties, independently of input-list ordering.
    ordered = sorted(predictions, key=lambda prediction: (-prediction.confidence, prediction.box_id))
    available = {target.box_id: target for target in ground_truth}
    matches: list[Match] = []
    false_positives: list[str] = []
    filtered: list[str] = []
    for prediction in ordered:
        if prediction.confidence < confidence_threshold:
            filtered.append(prediction.box_id)
            continue
        candidates = [
            (box_iou(prediction.xyxy, target.xyxy), target.box_id)
            for target in available.values()
            if not class_aware or prediction.class_id == target.class_id
        ]
        candidates = [candidate for candidate in candidates if candidate[0] >= iou_threshold]
        if not candidates:
            false_positives.append(prediction.box_id)
            continue
        overlap, target_id = min(candidates, key=lambda candidate: (-candidate[0], candidate[1]))
        matches.append(Match(prediction.box_id, target_id, overlap))
        del available[target_id]

    # A wrong-class prediction is an FP and leaves the unmatched target as an FN.
    result = MatchResult(tuple(matches), tuple(false_positives), tuple(sorted(available)), tuple(filtered))
    assert len(result.matches) + len(result.false_negative_ids) == len(ground_truth)
    assert len(result.matches) + len(result.false_positive_ids) + len(result.filtered_prediction_ids) == len(predictions)
    return result
