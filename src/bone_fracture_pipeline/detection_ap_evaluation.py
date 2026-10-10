"""Shared AP for D and F, using the pinned Ultralytics matcher and integration."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import torch

from .error_analysis import _validate_diagnostic_rows
from .error_analysis_training import verify_sample_export


def shared_ap(rows, *, samples=None, split="valid") -> dict:
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import compute_ap
    from ultralytics.utils.metrics import box_iou

    counts = _validate_diagnostic_rows(rows, split=split)
    if samples is not None:
        verify_sample_export(rows, samples, split=split)
    context = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    correct, scores, pred_classes, target_classes, identities = [], [], [], [], []
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        targets = sorted(row["ground_truth"], key=lambda g: g["box_id"])
        predictions = sorted((p for p in row["predictions"] if p["confidence"] > 0.001),
                             key=lambda p: (-p["confidence"], p["box_id"]))[:300]
        pboxes = torch.tensor([p["xyxy"] for p in predictions], dtype=torch.float32).reshape(-1, 4)
        tboxes = torch.tensor([g["xyxy"] for g in targets], dtype=torch.float32).reshape(-1, 4)
        pc = torch.tensor([p["class_id"] for p in predictions])
        tc = torch.tensor([g["class_id"] for g in targets])
        # Native IoU-first assignment is independent of the diagnostic confidence-first matcher.
        hits = BaseValidator.match_predictions(context, pc, tc, box_iou(tboxes, pboxes)).numpy()
        correct.extend(hits)
        scores.extend(p["confidence"] for p in predictions)
        pred_classes.extend(p["class_id"] for p in predictions)
        target_classes.extend(g["class_id"] for g in targets)
        identities.extend((row["sample_id"], p["box_id"]) for p in predictions)
    correct = np.asarray(correct, dtype=bool).reshape(-1, 10)
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], *identities[i]))
    correct, classes = correct[order], np.asarray(pred_classes)[order]
    per_class = []
    for c in range(6):
        support = target_classes.count(c)
        hits = correct[classes == c]
        ap = np.zeros(10)
        if support and len(hits):
            tp, fp = hits.cumsum(0), (~hits).cumsum(0)
            recall, precision = tp / support, tp / (tp + fp)
            ap = np.array([compute_ap(recall[:, i], precision[:, i])[0] for i in range(10)])
        per_class.append({"class_id": c, "support": support, "predictions": len(hits),
                          "ap50": float(ap[0]), "ap50_95": float(ap.mean()), "ap_by_iou": ap.tolist()})
    present = [c for c in per_class if c["support"]]
    return {"map50": float(np.mean([c["ap50"] for c in present])) if present else 0.0,
            "map50_95": float(np.mean([c["ap50_95"] for c in present])) if present else 0.0,
            "per_class": per_class, "counts": counts,
            "definition": {"implementation": "Ultralytics 8.4.155 BaseValidator.match_predictions + compute_ap",
                           "iou_thresholds": context.iouv.tolist(), "confidence": ">0.001", "max_det": 300,
                           "matching": "native IoU-first, prediction uniqueness then GT uniqueness",
                           "ties": "fixed image/box ID inputs; native NumPy argsort for IoU ties; stable score/image/box order for AP",
                           "interpolation": "precision envelope, 101 recall points, trapezoidal integration; not pycocotools COCOeval",
                           "unsupported_classes": "reported with zero AP, excluded from macro mean",
                           "diagnostic_matcher_used": False}}
