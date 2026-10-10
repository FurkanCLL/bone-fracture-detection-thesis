from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torchvision.models.detection.roi_heads import fastrcnn_loss

from bone_fracture_pipeline.experiment_f_runner import load_config
from bone_fracture_pipeline.faster_rcnn_training import (
    train_epoch, make_optimizer, ImageSchedule, check_losses, NumericalTrainingError,
)
from bone_fracture_pipeline.faster_rcnn_diagnostics import (
    parameter_status, rng_state, restore_rng, save_failure,
)


class ToyDataset:
    def __init__(self, length):
        self.samples = [SimpleNamespace(sample_id=f"train/{i}") for i in range(length)]
        self.seen = []
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        self.seen.append(index)
        count = index % 2
        return torch.full((3, 2, 2), float(index)), {
            "boxes": torch.ones(count, 4), "labels": torch.ones(count, dtype=torch.int64)}, {}


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.batches = []
    def forward(self, images, targets):
        self.batches.append([len(t["boxes"]) for t in targets])
        return {"loss": self.weight * torch.stack([x.mean() for x in images]).mean()}


class Stage1BTests(unittest.TestCase):
    def test_native_zero_proposals_are_captured(self):
        classifier, regression = fastrcnn_loss(torch.zeros(0, 7), torch.zeros(0, 28),
                                               [torch.zeros(0, dtype=torch.int64)], [torch.zeros(0, 4)])
        with self.assertRaises(NumericalTrainingError) as context:
            check_losses({"loss_classifier": classifier, "loss_box_reg": regression},
                         position=401, sample_id="train/empty")
        self.assertEqual(context.exception.evidence["total_loss"], "nan")
        json.dumps(context.exception.evidence, allow_nan=False)

    def test_failure_state_retains_gradients_optimizer_and_rng(self):
        model = ToyModel()
        config = load_config(Path.cwd())[0]
        optimizer = make_optimizer(model, config)
        scheduler = ImageSchedule(optimizer, config, 1211)
        model.weight.square().backward()
        expected = model.weight.grad.clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "capture.pt"
            save_failure(path, model, optimizer, scheduler, {"configuration": config}, {"total_loss": "nan"})
            state = torch.load(path, weights_only=False)
        self.assertFalse(state["resumable"])
        self.assertTrue(torch.equal(state["partial_gradients"]["weight"], expected))
        self.assertEqual(state["scheduler"]["images_per_epoch"], 1211)
        self.assertEqual(state["metadata"]["configuration"], config)
        self.assertIn("rng", state)

    def test_rng_restore_repeats_sampling(self):
        state = rng_state()
        first = torch.randperm(100)
        restore_rng(state)
        self.assertTrue(torch.equal(first, torch.randperm(100)))

    def test_parameter_and_partial_gradient_checks(self):
        model = ToyModel()
        model.weight.grad = torch.tensor(float("nan"))
        status = parameter_status(model)
        self.assertTrue(status["parameters_finite"])
        self.assertFalse(status["gradients_finite"])

    def test_diagnostic_does_not_flush_partial_window(self):
        config = load_config(Path.cwd())[0]
        model, dataset = ToyModel(), ToyDataset(1211)
        optimizer = make_optimizer(model, config)
        scheduler = ImageSchedule(optimizer, config, 1211)
        with tempfile.TemporaryDirectory() as temp:
            stats = train_epoch(model, dataset, optimizer, scheduler, epoch=0, config=config,
                                device=torch.device("cpu"), progress_path=Path(temp)/"updates.jsonl", max_images=500)
        self.assertEqual(stats["images"], 500)
        self.assertEqual(len(stats["optimizer_updates"]), 7)
        self.assertEqual(scheduler.images_seen, 448)
        expected = torch.randperm(1211, generator=torch.Generator().manual_seed(42)).tolist()[:500]
        self.assertEqual(dataset.seen, expected)


if __name__ == "__main__":
    unittest.main()
