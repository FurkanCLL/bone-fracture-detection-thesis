"""Read-only numerical probes and diagnostic state retention for Experiment F."""
from __future__ import annotations

import random

import numpy as np
import torch


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def tensor_status(value):
    value = value.detach()
    return {"shape": list(value.shape), "finite": bool(torch.isfinite(value).all()),
            "elements": value.numel()}


def parameter_status(model):
    # Check the accumulated gradients before attempting a backward on a failed loss.
    bad_parameters, bad_gradients, gradients = [], [], 0
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            bad_parameters.append(name)
        if parameter.grad is not None:
            gradients += 1
            if not torch.isfinite(parameter.grad).all():
                bad_gradients.append(name)
    return {"parameters_finite": not bad_parameters, "gradients_finite": not bad_gradients,
            "gradient_tensor_count": gradients, "nonfinite_parameters": bad_parameters,
            "nonfinite_gradients": bad_gradients}


class NumericalProbe:
    """Observe native tensors without changing sampling, outputs, or installed code."""
    def __init__(self, model):
        self.state = {}
        self.handles = [model.backbone.register_forward_hook(self.backbone),
                        model.rpn.head.register_forward_hook(self.rpn_head),
                        model.rpn.register_forward_hook(self.rpn),
                        model.roi_heads.box_predictor.register_forward_hook(self.predictor)]
        self.rpn_module = model.rpn
        self.original_filter = model.rpn.filter_proposals

        # The filter is a method, so inspect its inputs through a local wrapper.
        def filter_proposals(proposals, objectness, image_shapes, anchors_per_level):
            self.state["decoded_proposals"] = tensor_status(proposals)
            scores = objectness.detach().sigmoid()
            self.state["unfiltered_objectness"] = tensor_status(scores)
            self.state["maximum_objectness_per_image"] = scores.reshape(len(image_shapes), -1).amax(1).cpu().tolist()
            self.state["applied_rpn_threshold"] = model.rpn.score_thresh
            return self.original_filter(proposals, objectness, image_shapes, anchors_per_level)
        model.rpn.filter_proposals = filter_proposals

    def backbone(self, module, inputs, result):
        self.state.clear()
        self.state["backbone"] = {name: tensor_status(tensor) for name, tensor in result.items()}

    def rpn_head(self, module, inputs, result):
        self.state["rpn_logits"] = [tensor_status(t) for t in result[0]]
        self.state["rpn_deltas"] = [tensor_status(t) for t in result[1]]

    def rpn(self, module, inputs, result):
        self.state["rpn_proposals_per_image"] = [len(boxes) for boxes in result[0]]
        self.state["filtered_proposals"] = [tensor_status(boxes) for boxes in result[0]]

    def predictor(self, module, inputs, result):
        self.state["roi_logits"] = tensor_status(result[0])
        self.state["roi_box_regression"] = tensor_status(result[1])

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.rpn_module.filter_proposals = self.original_filter


def save_failure(path, model, optimizer, scheduler, metadata, evidence):
    # This is the exact partial state, never an epoch checkpoint or resumable run.
    torch.save({"status": "partial_numerical_failure", "resumable": False,
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "partial_gradients": {k: p.grad.detach().cpu() for k, p in model.named_parameters() if p.grad is not None},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "metadata": metadata, "failure": evidence,
                "rng": rng_state(), "forward_rng": getattr(model, "_failure_forward_rng", None)}, path)
