"""FP32 Faster R-CNN construction, accumulation and checkpoint primitives."""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from bone_fracture_audit.audit import sha256_file
from .faster_rcnn_diagnostics import parameter_status, tensor_status, rng_state


class NumericalTrainingError(FloatingPointError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(f"Non-finite loss at training image {evidence['position_zero_based']}; no automatic retuning.")


# Keep non-finite values as explicit strings so failure evidence remains strict JSON.
def check_losses(loss_dict, *, position, sample_id):
    values = {key: float(value.detach()) for key, value in loss_dict.items()}
    total = float(sum(loss_dict.values()).detach())
    if not math.isfinite(total) or not all(math.isfinite(value) for value in values.values()):
        raise NumericalTrainingError({"position_zero_based": position, "sample_id": sample_id,
                                      "total_loss": total if math.isfinite(total) else str(total),
                                      "losses": {key: value if math.isfinite(value) else str(value)
                                                 for key, value in values.items()}})


class LetterboxedTransform(GeneralizedRCNNTransform):
    # Geometry is already applied to both pixels and boxes in the dataset adapter.
    # Keep COCO normalization and batching, but reject unexpected dimensions instead of resizing twice.
    def resize(self, image, target=None):
        if tuple(image.shape) != (3, 640, 640):
            raise ValueError("F requires exactly one external 640-square letterbox.")
        return image, target


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def adapt_head(model):
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 7)
    return model


def build_model():
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    # Load the complete COCO detector before replacing only the category-specific predictor.
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, trainable_backbone_layers=3, progress=True,
        min_size=640, max_size=640, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, rpn_score_thresh=0.05,
        box_score_thresh=0.001, box_nms_thresh=0.5, box_detections_per_img=300,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5, box_batch_size_per_image=512,
        box_positive_fraction=0.25, bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
    )
    adapt_head(model)
    model.transform = LetterboxedTransform(640, 640, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], size_divisible=32)
    cache = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    if not frozen or any(not p.requires_grad for p in model.rpn.parameters()) or any(not p.requires_grad for p in model.roi_heads.parameters()):
        raise ValueError("The required backbone/head freezing was not applied.")
    report = {
        "identifier": "fasterrcnn_mobilenet_v3_large_fpn", "weights_enum": str(weights),
        "weights_url": weights.url, "weights_sha256": sha256_file(cache),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "frozen_names": frozen, "trainable_backbone_layers": 3,
        "trainable_backbone_blocks": sorted({n.split('.')[2] for n,p in model.named_parameters()
                                              if n.startswith('backbone.body.') and p.requires_grad}),
        "normalization_layers": sorted({type(m).__name__ for m in model.modules() if 'Norm' in type(m).__name__}),
        "transform": {"external": "aspect-preserving centered 640-square LetterBox, 114 padding, scaleup=True",
                      "internal_resize": "identity with strict 640-square guard", "size_divisible": 32,
                      "image_mean": model.transform.image_mean, "image_std": model.transform.image_std,
                      "native_default_min_size": 800, "native_default_max_size": 1333},
        "rpn": {"pre_nms": model.rpn._pre_nms_top_n, "post_nms": model.rpn._post_nms_top_n,
                "nms": model.rpn.nms_thresh, "score": model.rpn.score_thresh,
                "fg_iou": 0.7, "bg_iou": 0.3, "sample_count": 256, "positive_fraction": 0.5,
                "anchor_sizes": model.rpn.anchor_generator.sizes,
                "aspect_ratios": model.rpn.anchor_generator.aspect_ratios},
        "roi": {"score": model.roi_heads.score_thresh, "nms": model.roi_heads.nms_thresh,
                "max_det": model.roi_heads.detections_per_img, "fg_iou": 0.5, "bg_iou": 0.5,
                "sample_count": 512, "positive_fraction": 0.25,
                "bbox_reg_weights": model.roi_heads.box_coder.weights,
                "native_default_score": 0.05, "native_default_max_det": 100},
        "mapping": {"background": 0, "canonical_to_detector": {str(c): c + 1 for c in range(6)},
                    "detector_to_canonical": {str(c + 1): c for c in range(6)}}}
    return model.float(), report


def make_optimizer(model, config):
    # Match D's decay exemption for biases and normalization parameters.
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if name.endswith("bias") or parameter.ndim == 1 else decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": config["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0}], lr=config["lr"], betas=tuple(config["betas"]), eps=1e-8,
        foreach=False, fused=False)


class ImageSchedule:
    """Linear three-epoch warmup and 100-epoch decay, independent of smoke budget."""
    def __init__(self, optimizer, config, images_per_epoch):
        self.optimizer, self.config, self.images_per_epoch = optimizer, config, images_per_epoch
        self.images_seen = 0

    def step(self, images_seen, epoch):
        self.images_seen = images_seen
        fraction = min(images_seen / (self.config["warmup_epochs"] * self.images_per_epoch), 1.0)
        decay = max(1 - epoch / self.config["epochs"], 0) * (1 - self.config["final_lr_factor"]) + self.config["final_lr_factor"]
        lr = self.config["lr"] * fraction * decay
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self):
        return {"images_seen": self.images_seen, "images_per_epoch": self.images_per_epoch,
                "config": self.config, "definition": "linear image-based warmup then epoch-based linear decay"}

    def load_state_dict(self, state):
        if state["config"] != self.config or state["images_per_epoch"] != self.images_per_epoch:
            raise ValueError("Resume scheduler configuration changed.")
        self.images_seen = state["images_seen"]


def accumulation_windows(length, accumulation):
    if length < 1 or accumulation < 1:
        raise ValueError("Epoch length and accumulation must be positive.")
    for start in range(0, length, accumulation):
        yield start, min(accumulation, length - start)


# Check unscaled gradients before clipping so NaNs cannot be hidden by an optimizer update.
def finite_step(model, optimizer):
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not parameters or not all(torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("Non-finite or missing gradients; protocol change requires review.")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True, foreach=False)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite parameter after AdamW update.")
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def train_epoch(model, dataset, optimizer, scheduler, *, epoch, config, device, progress_path, max_images=None):
    if max_images is not None and not 1 <= max_images <= len(dataset):
        raise ValueError("Diagnostic image limit is outside the training split.")
    model.train()
    dataset.epoch = epoch
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config["seed"] + epoch)).tolist()
    optimizer.zero_grad(set_to_none=True)
    losses, components, updates = [], [], []
    started = time.perf_counter()
    with progress_path.open("w", encoding="utf-8") as log:
        for start, window_size in accumulation_windows(len(dataset), config["accumulation"]):
            for position in range(start, start + window_size):
                # A bounded diagnostic stops without stepping an unfinished 64-image window.
                if max_images is not None and position >= max_images:
                    return {"diagnostic_limit_reached": max_images, "images": len(losses),
                            "optimizer_updates": updates, "seconds": time.perf_counter() - started}
                image, target, image_metadata = dataset[order[position]]
                image = image.to(device)
                target = {k: v.to(device) for k, v in target.items()}
                if torch.is_autocast_enabled() or image.dtype != torch.float32:
                    raise ValueError("Experiment F must remain FP32 with AMP disabled.")
                model._failure_forward_rng = rng_state()
                loss_dict = model([image], [target])
                loss = sum(loss_dict.values())
                try:
                    check_losses(loss_dict, position=position, sample_id=dataset.samples[order[position]].sample_id)
                except NumericalTrainingError as error:
                    error.evidence.update({"completed_backwards": len(losses), "optimizer_updates": updates,
                                           "current_window_start": start, "current_window_size": window_size,
                                           "input_shape": list(image.shape), "image_metadata": image_metadata,
                                           "input_status": tensor_status(image), "ground_truth_count": len(target["boxes"]),
                                           "current_lrs": [group["lr"] for group in optimizer.param_groups],
                                           "optimizer_update_count": len(updates), "state_status": parameter_status(model),
                                           "target_boxes": target["boxes"].detach().cpu().tolist(),
                                           "target_labels": target["labels"].detach().cpu().tolist(),
                                           "finite_loss_mean_before_failure": float(np.mean(losses)) if losses else None,
                                           "finite_loss_max_before_failure": max(losses) if losses else None,
                                           "seconds_before_failure": time.perf_counter() - started})
                    raise
                # Divide by the actual window size, including the last 59-image window.
                (loss / window_size).backward()
                losses.append(float(loss.detach()))
                components.append({k: float(v.detach()) for k, v in loss_dict.items()})
                del loss, loss_dict, image, target
            images_seen = epoch * len(dataset) + start + window_size
            lr = scheduler.step(images_seen, epoch)
            norm = finite_step(model, optimizer)
            update = {"update": len(updates) + 1, "images": window_size, "lr": lr,
                      "gradient_norm_before_clip": norm, "completed_images": start + window_size}
            updates.append(update)
            import json
            log.write(json.dumps(update) + "\n")
            log.flush()
            print(f"epoch={epoch+1} images={start+window_size}/{len(dataset)} loss={losses[-1]:.6f} lr={lr:.7f}", flush=True)
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - started
    return {"images": len(losses), "seconds": elapsed, "images_per_second": len(losses) / elapsed,
            "loss_mean": float(np.mean(losses)), "loss_max": max(losses), "loss_min": min(losses),
            "loss_quantiles": np.quantile(losses, [0, .25, .5, .75, 1]).tolist(),
            "first_loss": losses[0], "last_loss": losses[-1],
            "component_means": {k: float(np.mean([c[k] for c in components])) for k in components[0]},
            "component_maxima": {k: max(c[k] for c in components) for k in components[0]},
            "optimizer_updates": updates, "all_losses_gradients_parameters_finite": True,
            "amp_enabled": False}


def checkpoint_is_better(score, best):
    if not math.isfinite(score):
        raise ValueError("Checkpoint selection requires finite shared AP.")
    return score > best


def save_checkpoint(path, model, optimizer, scheduler, *, epoch, best_score, metadata):
    record = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
              "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
              "epoch": epoch, "best_score": best_score, "metadata": metadata,
              "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                      "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    temporary = path.with_suffix(".tmp")
    torch.save(record, temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
