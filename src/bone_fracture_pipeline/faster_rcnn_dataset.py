"""Train/validation-only adapters for the fixed six-class CLAHE dataset."""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .error_analysis import (
    ValidationSample, _load_diagnostic_samples, _diagnostic_fingerprint,
    _verify_diagnostic_lineage, validate_image_paths,
)


def detector_label(class_id: int) -> int:
    if type(class_id) is not int or class_id not in range(6):
        raise ValueError("Canonical foreground class must be an integer in 0..5.")
    return class_id + 1


def canonical_label(label: int) -> int:
    if type(label) is not int or label not in range(1, 7):
        raise ValueError("Background is not an exported detection; foreground labels are 1..6.")
    return label - 1


# Read only the requested split, checking bytes against the original CLAHE manifest.
def load_split(project: Path, split: str) -> tuple[tuple[ValidationSample, ...], str]:
    if split not in ("train", "valid"):
        raise ValueError("Experiment F cannot access held-out data.")
    root = project / "data/prepared/v3_detection_clahe"
    counts = (1211, 698, 607) if split == "train" else (348, 204, 175)
    samples = _load_diagnostic_samples(root, split=split, expected_counts=counts)
    _verify_diagnostic_lineage(project / "outputs/phase2/phase2d/preprocessing_manifest.csv", samples, split=split)
    return samples, _diagnostic_fingerprint(root, samples, split=split)


# Reuse the pinned D primitives rather than reimplement affine clipping or HSV lookup tables.
def prepare_image(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, *, training: bool,
                  seed: int = 42) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict]:
    from ultralytics.data.augment import LetterBox, RandomHSV, RandomPerspective
    from ultralytics.utils.instance import Instances

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Input must be a three-channel uint8 BGR image.")
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if any(isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer)) for c in classes):
        raise ValueError("Foreground classes must be integers before category conversion.")
    labels = np.asarray([detector_label(int(c)) for c in classes], dtype=np.int64)
    if len(labels) != len(boxes) or not np.isfinite(boxes).all():
        raise ValueError("Targets must contain finite boxes and matching classes.")
    height, width = image.shape[:2]
    if len(boxes) and (np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any(boxes < 0)
                       or np.any(boxes[:, [0, 2]] > width) or np.any(boxes[:, [1, 3]] > height)):
        raise ValueError("Targets must have positive area inside the original image.")
    transform = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, center=True)
    record = {"img": image.copy(), "cls": labels.reshape(-1, 1),
              "instances": Instances(boxes.copy(), segments=np.zeros((0, 1000, 2), dtype=np.float32),
                                     bbox_format="xyxy", normalized=False)}
    params = transform.get_params(record)
    record = transform.apply_image(record, params)
    record = transform.apply_instances(record, params)
    metadata = {"width": width, "height": height, "ratio": list(params["ratio"]),
                "left": params["left"], "top": params["top"],
                "resized_wh": list(params["new_unpad"]), "training": training}
    if training:
        # A per-image/epoch seed isolates augmentation from ROI sampling and loader order.
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            affine = RandomPerspective(degrees=10, translate=0.05, scale=0.10, shear=0, perspective=0)
            affine_params = affine.get_params(record)
            metadata["affine_matrix"] = affine_params["M"].tolist()
            metadata["scale"] = float(affine_params["scale"])
            record = affine.apply_image(record, affine_params)
            record = affine.apply_instances(record, affine_params)
            record = RandomHSV(hgain=0, sgain=0, vgain=0.15)(record)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
    tensor = torch.from_numpy(np.ascontiguousarray(record["img"][..., ::-1].transpose(2, 0, 1))).float() / 255
    result_boxes = torch.as_tensor(record["instances"].bboxes.copy(), dtype=torch.float32).reshape(-1, 4)
    target = {"boxes": result_boxes, "labels": torch.as_tensor(record["cls"].reshape(-1), dtype=torch.int64)}
    if not torch.isfinite(result_boxes).all() or (len(result_boxes) and not (result_boxes[:, 2:] > result_boxes[:, :2]).all()):
        raise ValueError("Augmentation produced invalid targets.")
    return tensor, target, metadata


def original_boxes(boxes: torch.Tensor, metadata: dict) -> torch.Tensor:
    result = boxes.detach().cpu().clone().reshape(-1, 4)
    result[:, [0, 2]] = (result[:, [0, 2]] - metadata["left"]) / metadata["ratio"][0]
    result[:, [1, 3]] = (result[:, [1, 3]] - metadata["top"]) / metadata["ratio"][1]
    result[:, [0, 2]] = result[:, [0, 2]].clamp(0, metadata["width"])
    result[:, [1, 3]] = result[:, [1, 3]].clamp(0, metadata["height"])
    return result


class FractureDataset(Dataset):
    def __init__(self, samples: tuple[ValidationSample, ...], *, split: str, seed: int = 42):
        if split not in ("train", "valid") or any(not s.sample_id.startswith(split + "/") for s in samples):
            raise ValueError("Dataset membership must match the allowed split.")
        self.samples, self.split, self.seed, self.epoch = samples, split, seed, 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        validate_image_paths([sample.image_path], sample.image_path.parent.parent.parent / self.split / "images")
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None or image.shape[:2] != (sample.height, sample.width):
            raise ValueError("Image decoding changed the recorded dimensions.")
        return prepare_image(image, np.array([g.xyxy for g in sample.ground_truth]),
                             np.array([g.class_id for g in sample.ground_truth]), training=self.split == "train",
                             seed=self.seed + self.epoch * len(self) + index)


# Rows retain the Phase 4 schema so fixed-threshold, size and class-agnostic diagnostics work unchanged.
def prediction_row(sample: ValidationSample, prediction: dict, metadata: dict) -> dict:
    boxes = original_boxes(prediction["boxes"], metadata)
    scores, labels = prediction["scores"].detach().cpu(), prediction["labels"].detach().cpu()
    predictions = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        class_id = canonical_label(int(label))
        if not torch.isfinite(box).all() or not torch.isfinite(score):
            raise ValueError("Non-finite prediction.")
        if float(score) > 0.001 and bool((box[2:] > box[:2]).all()):
            predictions.append({"box_id": f"pred/{i:04d}", "class_id": class_id,
                                "xyxy": box.tolist(), "confidence": float(score)})
    split = sample.sample_id.split("/")[0]
    return {"sample_id": sample.sample_id, "split": split,
            "image": f"{split}/images/{sample.image_path.name}",
            "width": sample.width, "height": sample.height,
            "image_sha256": sample.image_sha256, "label_sha256": sample.label_sha256,
            "ground_truth": [asdict(g) for g in sample.ground_truth], "predictions": predictions}
