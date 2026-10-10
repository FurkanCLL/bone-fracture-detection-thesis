from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator

from bone_fracture_pipeline.experiment_f_runner import load_config
from bone_fracture_pipeline.faster_rcnn_training import (
    train_epoch, make_optimizer, ImageSchedule, check_losses, NumericalTrainingError, PhaseAwareRPN,
    reset_training_state,
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
    def test_reset_discards_preflight_weights_moments_gradients_and_rng(self):
        config = load_config(Path.cwd())[0]
        model = ToyModel()
        initial = {k:v.detach().clone() for k,v in model.state_dict().items()}
        optimizer = make_optimizer(model, config)
        model.weight.square().backward()
        optimizer.step()
        self.assertTrue(optimizer.state)
        optimizer, scheduler = reset_training_state(model, initial, config, 1211)
        self.assertTrue(torch.equal(model.weight, initial["weight"]))
        self.assertIsNone(model.weight.grad)
        self.assertFalse(optimizer.state)
        self.assertEqual(scheduler.images_seen, 0)
        expected = torch.rand(3, generator=torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(torch.rand(3), expected))

    def test_config_rejects_unapproved_scientific_change(self):
        import yaml
        config = load_config(Path.cwd())[0]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            path = project / "configs/experiments/F.yaml"
            path.parent.mkdir(parents=True)
            for key, value in (("batch_size", 3), ("accumulation", 32), ("amp", True),
                               ("lr", .0001), ("training_rpn_score_threshold", .05)):
                path.write_text(yaml.safe_dump({**config,key:value}),encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(project)
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
        config = {**config, "batch_size": 1, "accumulation": 64}
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

    def test_minibatch_order_mixed_targets_and_partial_windows(self):
        expected_order = torch.randperm(1211, generator=torch.Generator().manual_seed(42)).tolist()
        for batch in (1, 2, 4):
            with self.subTest(batch=batch):
                config = {**load_config(Path.cwd())[0], "batch_size": batch, "accumulation": 64 // batch}
                model, dataset = ToyModel(), ToyDataset(1211)
                optimizer = make_optimizer(model, config)
                scheduler = ImageSchedule(optimizer, config, 1211)
                with tempfile.TemporaryDirectory() as temp:
                    stats = train_epoch(model, dataset, optimizer, scheduler, epoch=0, config=config,
                                        device=torch.device("cpu"), progress_path=Path(temp)/"updates.jsonl")
                self.assertEqual(dataset.seen, expected_order)
                self.assertEqual(stats["unique_images"], 1211)
                self.assertEqual([u["images"] for u in stats["optimizer_updates"]], [64]*18 + [59])
                self.assertEqual(stats["batch_sizes"][-1], 3 if batch == 4 else 1)
                self.assertEqual(scheduler.images_seen, 1211)
                if batch > 1:
                    self.assertTrue(any(len(set(labels)) > 1 for labels in model.batches))

    def test_image_weighted_gradient_with_partial_microbatch(self):
        config = {**load_config(Path.cwd())[0], "batch_size": 4, "accumulation": 16}
        model, dataset = ToyModel(), ToyDataset(7)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        scheduler = ImageSchedule(optimizer, config, 7)
        with tempfile.TemporaryDirectory() as temp:
            stats = train_epoch(model, dataset, optimizer, scheduler, epoch=0, config=config,
                                device=torch.device("cpu"), progress_path=Path(temp)/"updates.jsonl")
        self.assertEqual(stats["batch_sizes"], [4, 3])
        # Mean index is three; equal weighting of the two batches gives the wrong gradient.
        self.assertAlmostEqual(model.weight.item(), 1 - (.001/3)*3, places=6)

    def test_phase_threshold_switch_uses_loaded_head(self):
        original = RegionProposalNetwork(AnchorGenerator(((32,),), ((1.,),)), RPNHead(8, 1),
                                         .7, .3, 256, .5, {"training":2000,"testing":1000},
                                         {"training":2000,"testing":1000}, .7, score_thresh=.05)
        rpn = PhaseAwareRPN(original, .0, .05)
        self.assertIs(rpn.head, original.head)
        for training, threshold in ((False,.05),(True,.0),(False,.05),(True,.0)):
            rpn.train(training)
            self.assertEqual(rpn.score_thresh, threshold)

    def test_scheduler_rejects_batch_or_threshold_change_on_resume(self):
        config = load_config(Path.cwd())[0]
        optimizer = make_optimizer(ToyModel(), config)
        scheduler = ImageSchedule(optimizer, config, 1211)
        saved = scheduler.state_dict()
        for key, value in (("batch_size", 3), ("training_rpn_score_threshold", .05)):
            different = ImageSchedule(optimizer, {**config, key:value}, 1211)
            with self.assertRaises(ValueError):
                different.load_state_dict(saved)

    def test_old_diagnostic_scientific_config_is_explicit(self):
        original = load_config(Path.cwd(), original_diagnostic=True)[0]
        self.assertEqual(original["protocol_version"], 1)
        self.assertEqual(original["batch_size"], 1)
        self.assertEqual(original["accumulation"], 64)
        self.assertNotIn("training_rpn_score_threshold", original)


if __name__ == "__main__":
    unittest.main()
