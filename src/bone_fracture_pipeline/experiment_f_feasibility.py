"""Captured-state replay and hardware-only batch feasibility for Experiment F."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import time

import torch

from bone_fracture_audit.audit import sha256_file
from .error_analysis import _write_json
from .experiment_f_runner import held_out_guard, load_config, fresh_output, implementation_hashes
from .faster_rcnn_dataset import load_split, FractureDataset
from .faster_rcnn_diagnostics import NumericalProbe, parameter_status, restore_rng
from .faster_rcnn_training import build_model, set_seed, make_optimizer, ImageSchedule, train_epoch, reset_training_state


def confirmed_rpn_failure(evidence):
    # All three authorization conditions must hold before trying the approved correction.
    def finite_tree(value):
        if isinstance(value, dict):
            return value.get("finite", True) and all(finite_tree(v) for v in value.values())
        if isinstance(value, list):
            return all(finite_tree(v) for v in value)
        return True
    return (evidence.get("rpn_proposals_per_image") == [0]
            and evidence.get("ground_truth_count") == 0
            and any(evidence.get("losses", {}).get(k) in ("nan", "inf", "-inf")
                    for k in ("loss_classifier", "loss_box_reg"))
            and evidence.get("input_status", {}).get("finite") is True
            and evidence.get("state_status", {}).get("parameters_finite") is True
            and evidence.get("state_status", {}).get("gradients_finite") is True
            and bool(evidence.get("intermediates")) and finite_tree(evidence["intermediates"]))


def replay(project, output, capture):
    capture = capture.resolve()
    if not capture.is_relative_to(project / "outputs/training/experiment_f"):
        raise ValueError("Replay requires a trusted local F diagnostic capture.")
    capture_hash = sha256_file(capture)
    saved = torch.load(capture, map_location="cpu", weights_only=False)
    if saved.get("resumable") is not False or not confirmed_rpn_failure(saved["failure"]):
        raise ValueError("The captured evidence does not confirm the authorized RPN mechanism.")
    config = load_config(project)[0]
    samples, fingerprint = load_split(project, "train")
    dataset = FractureDataset(samples, split="train")
    index = next(i for i, s in enumerate(samples) if s.sample_id == saved["failure"]["sample_id"])
    image, target, info = dataset[index]
    set_seed(42)
    model, report = build_model(config)
    model.cuda()
    probe = NumericalProbe(model)
    trials = []
    for threshold in (.05, .0):
        model.load_state_dict(saved["model"])
        model.train()
        model.rpn.score_thresh = threshold
        # Restore the 17 pending image gradients and the pre-forward sampling RNG exactly.
        for name, parameter in model.named_parameters():
            grad = saved["partial_gradients"].get(name)
            parameter.grad = grad.cuda() if grad is not None else None
        restore_rng(saved["forward_rng"])
        losses = model([image.cuda()], [{k: v.cuda() for k, v in target.items()}])
        finite = all(bool(torch.isfinite(v)) for v in losses.values())
        record = {"threshold": threshold, "finite_losses": finite,
                  "losses": {k: float(v.detach()) if torch.isfinite(v) else str(float(v.detach()))
                             for k, v in losses.items()}, "intermediates": deepcopy(probe.state)}
        if finite:
            (sum(losses.values()) / 64).backward()
            record["state_after_backward"] = parameter_status(model)
        trials.append(record)
        del losses
    model.eval()
    inference_threshold = model.rpn.score_thresh
    model.train()
    training_threshold = model.rpn.score_thresh
    probe.close()
    if trials[0]["finite_losses"] or not trials[1]["finite_losses"] or not trials[1]["state_after_backward"]["gradients_finite"]:
        raise ValueError("Captured-state correction verification failed.")
    result = {"confirmed": True, "sample_id": samples[index].sample_id, "train_fingerprint": fingerprint,
              "capture_sha256": sha256_file(capture), "trials": trials, "model": report,
              "training_threshold_after_switch": training_threshold,
              "inference_threshold_after_switch": inference_threshold,
              "optimizer_update_executed": False, "capture_unchanged": sha256_file(capture) == capture_hash}
    _write_json(output / "replay.json", result)
    return result


def benchmark(project, output):
    config, config_hash = load_config(project)
    samples, fingerprint = load_split(project, "train")
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(42)).tolist()[:128]
    selected = [samples[i] for i in order]
    coverage = {"images": 128, "empty_labels": sum(not s.ground_truth for s in selected),
                "annotations": sum(len(s.ground_truth) for s in selected),
                "original_width_range": [min(s.width for s in selected), max(s.width for s in selected)],
                "original_height_range": [min(s.height for s in selected), max(s.height for s in selected)],
                "target_counts": sorted({len(s.ground_truth) for s in selected}),
                "classes": sorted({g.class_id for s in selected for g in s.ground_truth})}
    results, initial_hash = [], None
    for batch in (1, 2, 4):
        candidate = deepcopy(config)
        candidate.update(batch_size=batch, accumulation=64 // batch)
        started = time.perf_counter()
        set_seed(42)
        model, model_report = build_model(candidate)
        # Hash the entire adapted initialization, including the random six-class predictor.
        import hashlib
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.cpu().numpy().tobytes())
        state_hash = digest.hexdigest()
        initial_hash = initial_hash or state_hash
        if state_hash != initial_hash:
            raise ValueError("Benchmark candidates did not start from identical weights.")
        model.cuda()
        optimizer = make_optimizer(model, candidate)
        scheduler = ImageSchedule(optimizer, candidate, len(samples))
        # Equal warmup work initializes kernels, then every mutable state is reset.
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        dataset = FractureDataset(samples, split="train")
        items = [dataset[i] for i in order[:4]]
        loss = sum(model([item[0].cuda() for item in items],
                         [{k: v.cuda() for k, v in item[1].items()} for item in items]).values())
        loss.backward()
        if not parameter_status(model)["gradients_finite"]:
            raise FloatingPointError("Benchmark warmup gradient failed.")
        del loss, items
        optimizer, scheduler = reset_training_state(model, initial, candidate, len(samples))
        del initial
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initialization_seconds = time.perf_counter() - started
        stats = train_epoch(model, dataset, optimizer, scheduler, epoch=0, config=candidate,
                            device=torch.device("cuda:0"), progress_path=output / f"batch_{batch}_updates.jsonl", max_images=128)
        capacity = torch.cuda.get_device_properties(0).total_memory
        record = {"physical_batch": batch, "accumulation": 64 // batch, "effective_images": 64,
                  "initialization_seconds_excluded": initialization_seconds, "initial_state_sha256": state_hash,
                  "empty_optimizer_at_start": True, "model_optimizer_rng_reset_after_warmup": True,
                  "training": stats, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "capacity_bytes": capacity,
                  "weights_sha256": model_report["weights_sha256"], "final_state": parameter_status(model)}
        record["reserved_headroom_fraction"] = 1 - record["peak_reserved_bytes"] / capacity
        if stats["images"] != 128 or [u["images"] for u in stats["optimizer_updates"]] != [64, 64]:
            raise ValueError("Benchmark did not complete two actual 64-image optimizer windows.")
        results.append(record)
        _write_json(output / f"batch_{batch}.json", record)
        del model, optimizer, scheduler, dataset
        gc.collect()
        torch.cuda.empty_cache()
    baseline = results[0]["training"]["images_per_second"]
    eligible = [r for r in results if r["reserved_headroom_fraction"] >= .25
                and r["training"]["all_losses_gradients_parameters_finite"]
                and r["training"]["images_per_second"] >= 1.10 * baseline]
    chosen = max(eligible, key=lambda r: r["physical_batch"]) if eligible else results[0]
    result = {"candidates": results, "coverage": coverage, "train_fingerprint": fingerprint,
              "configuration_sha256_before_selection": config_hash,
              "selection": {"batch_size": chosen["physical_batch"], "accumulation": chosen["accumulation"],
                            "effective_images": 64, "policy": "prefer batch 4, then 2; >=25% reserved headroom and >=10% throughput gain over 1",
                            "validation_ap_used": False, "recorded_before_final_smoke": True}}
    _write_json(output / "benchmark.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("replay", "benchmark"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args(argv)
    project = Path.cwd().resolve()
    output = fresh_output(project, args.output)
    (output / "runtime").mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(output / "runtime")
    os.environ["TORCH_HOME"] = str(project / "outputs/training/experiment_f/weights_cache")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with held_out_guard(project) as guard:
        if args.action == "replay":
            if args.capture is None:
                raise ValueError("Replay requires --capture.")
            result = replay(project, output, args.capture)
        else:
            result = benchmark(project, output)
        result["test_io_blocked_attempts"] = guard["blocked_attempts"]
        result["implementation_hashes"] = implementation_hashes(project)
        _write_json(output / "result.json", result)
    print(json.dumps({"action": args.action, "status": "complete"}))


if __name__ == "__main__":
    main()
