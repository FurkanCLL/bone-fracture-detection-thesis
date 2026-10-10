from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / "outputs/training/experiment_f/test_runtime"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

from bone_fracture_pipeline.detection_evaluation import GroundTruth
from bone_fracture_pipeline.error_analysis import ValidationSample, validate_export_rows
from bone_fracture_pipeline.error_analysis_quantitative import evaluate_rows, size_group
from bone_fracture_pipeline.faster_rcnn_dataset import (
    detector_label, canonical_label, load_split, FractureDataset, prepare_image, original_boxes, prediction_row,
)
from bone_fracture_pipeline.faster_rcnn_training import (
    adapt_head, LetterboxedTransform, accumulation_windows, finite_step, ImageSchedule,
    checkpoint_is_better, save_checkpoint, make_optimizer,
)
from bone_fracture_pipeline.detection_ap_evaluation import shared_ap
from bone_fracture_pipeline.experiment_f_runner import held_out_guard, fresh_output, load_config, run


def row(targets=(), predictions=(), sample="valid/fixture"):
    return {"sample_id": sample, "split": "valid", "width": 100, "height": 80,
            "ground_truth": list(targets), "predictions": list(predictions)}


def gt(c=0, box=(10, 10, 30, 30), name="g"):
    return {"box_id": name, "class_id": c, "xyxy": list(box)}


def pred(c=0, score=.9, box=(10, 10, 30, 30), name="p"):
    return {**gt(c, box, name), "confidence": score}


class ExperimentFTests(unittest.TestCase):
    def test_class_roundtrip(self):
        for c in range(6):
            self.assertEqual(canonical_label(detector_label(c)), c)

    def test_bad_foreground(self):
        for value in (-1, 6, True, 1.0, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                detector_label(value)

    def test_background_is_not_detection(self):
        for value in (0, -1, 7, True, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_label(value)

    def test_split_rejected_before_io(self):
        with patch("pathlib.Path.open", side_effect=AssertionError("IO")):
            with self.assertRaises(ValueError):
                load_split(Path("unread"), "test")

    def test_dataset_identity_guard(self):
        sample = SimpleNamespace(sample_id="test/forbidden")
        with self.assertRaises(ValueError):
            FractureDataset((sample,), split="train")

    def test_empty_label_target_shape(self):
        image, target, info = prepare_image(np.zeros((80, 100, 3), np.uint8), [], [], training=True)
        self.assertEqual(tuple(target["boxes"].shape), (0, 4))
        self.assertEqual(target["labels"].dtype, torch.int64)
        self.assertEqual(tuple(image.shape), (3, 640, 640))

    def test_resize_roundtrip_arbitrary_dimensions(self):
        for h, w in ((79, 139), (1200, 300), (320, 320), (641, 1279)):
            boxes = np.array([[w*.1, h*.2, w*.8, h*.9]], dtype=np.float32)
            image, target, info = prepare_image(np.zeros((h, w, 3), np.uint8), boxes, [5], training=False)
            self.assertTrue(torch.allclose(original_boxes(target["boxes"], info), torch.tensor(boxes), atol=.0002))
            self.assertEqual(target["labels"].tolist(), [6])
            self.assertEqual(tuple(image.shape), (3, 640, 640))

    def test_internal_transform_no_second_resize(self):
        transform = LetterboxedTransform(640, 640, [.485,.456,.406], [.229,.224,.225])
        image = torch.zeros(3, 640, 640)
        target = {"boxes": torch.tensor([[10.,20.,30.,40.]])}
        batch, targets = transform([image], [target])
        self.assertEqual(tuple(batch.tensors.shape), (1,3,640,640))
        self.assertTrue(torch.equal(targets[0]["boxes"], target["boxes"]))
        self.assertAlmostEqual(float(batch.tensors[0,0,0,0]), -.485/.229, places=5)
        with self.assertRaises(ValueError):
            transform([torch.zeros(3,800,800)])

    def test_validation_is_seed_independent(self):
        image = np.arange(80*100*3, dtype=np.uint8).reshape(80,100,3)
        a = prepare_image(image, [[10,10,30,30]], [0], training=False, seed=1)
        b = prepare_image(image, [[10,10,30,30]], [0], training=False, seed=2)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertNotIn("affine_matrix", a[2])

    def test_training_is_deterministic_and_clipped(self):
        image = np.full((80,100,3), 120, np.uint8)
        for seed in range(10):
            a = prepare_image(image, [[0,0,100,80],[40,30,41,31]], [0,1], training=True, seed=seed)
            b = prepare_image(image, [[0,0,100,80],[40,30,41,31]], [0,1], training=True, seed=seed)
            self.assertTrue(torch.equal(a[0], b[0]))
            self.assertTrue(torch.equal(a[1]["boxes"], b[1]["boxes"]))
            self.assertTrue((a[1]["boxes"] >= 0).all() and (a[1]["boxes"] <= 640).all())
            self.assertGreaterEqual(a[2]["scale"], .9)
            self.assertLessEqual(a[2]["scale"], 1.1)

    def test_invalid_boxes_rejected(self):
        for box in ([10,10,5,5], [-1,0,20,20], [0,0,float("nan"),20]):
            with self.assertRaises(ValueError):
                prepare_image(np.zeros((80,100,3),np.uint8), [box], [0], training=True)

    def test_adapter_rejects_noninteger_classes(self):
        for c in (1.5, True, "1"):
            with self.assertRaises(ValueError):
                prepare_image(np.zeros((80,100,3),np.uint8), [[10,10,30,30]], [c], training=True)

    def test_head_has_background_and_six_classes(self):
        model = SimpleNamespace(roi_heads=SimpleNamespace(box_predictor=FastRCNNPredictor(16, 91)))
        adapt_head(model)
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 7)
        self.assertEqual(model.roi_heads.box_predictor.bbox_pred.out_features, 28)

    def test_partial_accumulation_window(self):
        windows = list(accumulation_windows(1211,64))
        self.assertEqual(len(windows),19)
        self.assertEqual(windows[-1], (1152,59))
        self.assertEqual(sum(n for _,n in windows),1211)

    def test_averaged_gradient_including_partial_window(self):
        model = nn.Linear(1,1,bias=False)
        model.weight.data.zero_()
        optimizer = torch.optim.SGD(model.parameters(),lr=.1)
        for start, size in accumulation_windows(5,3):
            for i in range(start,start+size):
                (model(torch.tensor([[float(i+1)]]))).sum().div(size).backward()
            finite_step(model, optimizer)
        self.assertAlmostEqual(model.weight.item(), -.1*(2+4.5), places=6)

    def test_nan_gradient_prevents_update(self):
        model = nn.Linear(1,1)
        optimizer = torch.optim.SGD(model.parameters(),lr=.1)
        before = model.weight.detach().clone()
        model.weight.grad = torch.full_like(model.weight,float("nan"))
        with self.assertRaises(FloatingPointError):
            finite_step(model,optimizer)
        self.assertTrue(torch.equal(before,model.weight))

    def test_optimizer_decay_exemptions(self):
        model = nn.Linear(2,1)
        config = load_config(Path.cwd())[0]
        optimizer = make_optimizer(model,config)
        self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [.0005,0])
        self.assertEqual(optimizer.param_groups[0]["betas"],(.937,.999))

    def test_smoke_uses_full_budget_lr_schedule(self):
        config = load_config(Path.cwd())[0]
        optimizer = torch.optim.AdamW(nn.Linear(1,1).parameters(),lr=.001)
        scheduler = ImageSchedule(optimizer,config,1211)
        self.assertAlmostEqual(scheduler.step(1211,0), .001/3)
        self.assertAlmostEqual(scheduler.step(3633,2), .001*(1-2/100*.99))
        copy = ImageSchedule(optimizer,config,1211)
        copy.load_state_dict(scheduler.state_dict())
        self.assertEqual(copy.images_seen,3633)

    def test_checkpoint_selection(self):
        self.assertTrue(checkpoint_is_better(.1,-1))
        self.assertFalse(checkpoint_is_better(.1,.1))
        with self.assertRaises(ValueError):
            checkpoint_is_better(float("nan"),.1)

    def test_checkpoint_serialization(self):
        model = nn.Linear(1,1)
        config = load_config(Path.cwd())[0]
        optimizer = make_optimizer(model,config)
        model(torch.ones(1,1)).sum().backward()
        finite_step(model,optimizer)
        scheduler = ImageSchedule(optimizer,config,5)
        scheduler.step(5,0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"last.pt"
            save_checkpoint(path,model,optimizer,scheduler,epoch=0,best_score=.1,metadata={"seed":42})
            result = torch.load(path,weights_only=False)
            self.assertTrue(torch.equal(result["model"]["weight"],model.weight))
            self.assertTrue(result["optimizer"]["state"])
            self.assertEqual(result["scheduler"]["images_seen"],5)
            self.assertIn("torch",result["rng"])

    def test_prediction_export_coordinates(self):
        sample = ValidationSample("valid/s",Path("s.png"),Path("s.txt"),100,80,
                                  (GroundTruth("g",0,(10,10,30,30)),),"imagehash","labelhash")
        _, target, info = prepare_image(np.zeros((80,100,3),np.uint8), [[10,10,30,30]], [0], training=False)
        p = {"boxes": target["boxes"], "labels": target["labels"], "scores":torch.tensor([.9])}
        result = prediction_row(sample,p,info)
        self.assertEqual(result["predictions"][0]["class_id"],0)
        self.assertTrue(np.allclose(result["predictions"][0]["xyxy"],[10,10,30,30]))
        validate_export_rows([result])
        self.assertEqual(json.loads(json.dumps(result))["ground_truth"], [json.loads(json.dumps(asdict(sample.ground_truth[0])))])

    def test_ap_perfect_multiclass(self):
        result = shared_ap([row([gt()], [pred()]),row([gt(1)], [pred(1)],sample="valid/b")])
        self.assertAlmostEqual(result["map50"], .995)
        self.assertAlmostEqual(result["map50_95"], .995)
        self.assertEqual([c["support"] for c in result["per_class"]], [1,1,0,0,0,0])

    def test_ap_empty_cases(self):
        for rows in ([],[row()],[row([gt()])],[row(predictions=[pred()])]):
            self.assertEqual(shared_ap(rows)["map50"],0)

    def test_ap_wrong_class_duplicate_and_empty_image(self):
        rows = [row([gt()],[pred(1, .99, name="wrong"),pred(0,.8),pred(0,.7,name="dup")]),
                row(predictions=[pred(0,.95)],sample="valid/b")]
        result = shared_ap(rows)
        self.assertLess(result["map50"], .6)
        self.assertGreater(result["map50"],0)

    def test_ap_floor_and_maxdet(self):
        predictions = [pred(0,.9-i*.0001,name=f"{i:04}",box=(50,50,60,60)) for i in range(300)]
        predictions += [pred(0,.1,name="last")]
        self.assertEqual(shared_ap([row([gt()],predictions)])["map50"],0)
        self.assertEqual(shared_ap([row([gt()],[pred(score=.001)])])["map50"],0)

    def test_ap_reference_ap_per_class(self):
        from ultralytics.utils.metrics import ap_per_class
        # These non-conflicting boxes have a known per-score TP sequence, including a high-score FP.
        rows = [row([gt()], [pred(score=.8),pred(score=.9,box=(50,50,60,60),name="fp")]),
                row([gt(1)],[pred(1,.7)],sample="valid/b")]
        hits = np.array([[False]*10,[True]*10,[True]*10])
        reference = ap_per_class(hits,np.array([.9,.8,.7]),np.array([0,0,1]),np.array([0,1]))[5]
        result = shared_ap(rows)
        self.assertAlmostEqual(result["map50"], reference[:,0].mean())
        self.assertAlmostEqual(result["map50_95"], reference.mean())

    def test_ap_deterministic_ties(self):
        a = row([gt()],[pred(name="b"),pred(name="a")])
        b = row([gt()],[pred(name="a"),pred(name="b")])
        self.assertEqual(shared_ap([a]),shared_ap([b]))

    def test_diagnostic_compatibility(self):
        rows = [row([gt()],[pred()])]
        self.assertEqual(evaluate_rows(rows,confidence=.25,iou=.5)[0]["result"].metrics["tp"],1)
        self.assertEqual(evaluate_rows(rows,confidence=.25,iou=.5,class_aware=False)[0]["result"].metrics["fn"],0)
        self.assertTrue(size_group(GroundTruth("g",0,(10,10,30,30)),rows[0]))

    def test_ap_test_split_rejected(self):
        invalid = row()
        invalid["sample_id"], invalid["split"] = "test/x", "test"
        with self.assertRaises(ValueError):
            shared_ap([invalid])

    def test_test_guard_blocks_before_read_or_list(self):
        project = Path.cwd()
        with held_out_guard(project) as state:
            with self.assertRaises(PermissionError):
                (project/"data/prepared/fiction/test/nonexistent.txt").read_bytes()
            with self.assertRaises(PermissionError):
                list((project/"data/prepared/fiction/test").iterdir())
        self.assertEqual(state["blocked_attempts"],2)

    def test_fresh_output_and_official_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            target = project/"outputs/training/experiment_f/run"
            fresh_output(project,target)
            with self.assertRaises(ValueError):
                fresh_output(project,target)
            with self.assertRaises(ValueError):
                fresh_output(project,project/"outside")
            with self.assertRaises(ValueError):
                run(project,target,mode="official")
            with self.assertRaises(ValueError):
                run(project,target,mode="smoke",resume=Path("last.pt"))


if __name__ == "__main__":
    unittest.main()
