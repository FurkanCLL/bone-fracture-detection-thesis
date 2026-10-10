"""Separate, explicitly gated smoke and official Faster R-CNN execution."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

import yaml

from bone_fracture_audit.audit import sha256_file
from .error_analysis import _write_json, _diagnostic_fingerprint, validate_export_rows
from .error_analysis_training import read_stage1_validation, verify_sample_export
from .faster_rcnn_dataset import FractureDataset, load_split, prediction_row


# Deny test-tree IO before it happens, including directory enumeration and symlink destinations.
@contextmanager
def held_out_guard(project):
    state = {"enabled": True, "blocked_attempts": 0}
    roots = [(project / p).resolve() for p in ("data/raw", "data/prepared")]
    def hook(event, args):
        if state["enabled"] and event in ("open", "os.listdir", "os.scandir") and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).absolute()
            for candidate in (path, path.resolve()):
                if any(candidate.is_relative_to(root) for root in roots) and "test" in {part.casefold() for part in candidate.parts}:
                    state["blocked_attempts"] += 1
                    raise PermissionError("Experiment F prohibits held-out split IO.")
    sys.addaudithook(hook)
    try:
        yield state
    finally:
        state["enabled"] = False


def load_config(project, *, original_diagnostic=False):
    path = project / "configs/experiments/F.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {"experiment": "F", "protocol_version": 2, "dataset": "data/prepared/v3_detection_clahe",
                "model": "fasterrcnn_mobilenet_v3_large_fpn", "weights": "COCO_V1", "trainable_backbone_layers": 3,
                "image_size": 640, "epochs": 100, "batch_size": 1, "validation_batch_size": 1,
                "accumulation": 64, "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0005,
                "betas": [0.937, 0.999], "warmup_epochs": 3, "final_lr_factor": 0.01, "seed": 42,
                "amp": False, "deterministic": True, "early_stopping": False, "workers": 0,
                "training_rpn_score_threshold": 0.0,
                "augmentation": {"degrees": 10.0, "translate": 0.05, "scale": 0.10, "hsv_v": 0.15},
                "evaluation": {"confidence_floor": 0.001, "max_detections": 300, "primary": "shared_map50_95",
                               "roi_nms": 0.5, "rpn_score_threshold": 0.05}}
    batch = config.get("batch_size")
    if type(batch) is not int or batch not in (1, 2, 4):
        raise ValueError("F supports only the approved physical batches 1, 2 and 4.")
    expected.update(batch_size=batch, accumulation=64 // batch)
    if config != expected:
        raise ValueError("F configuration differs from the implemented Stage 1B protocol; explicit review is required.")
    if original_diagnostic:
        # Retain a reproducible diagnostic route after the approved protocol revision.
        original = {**expected, "protocol_version": 1, "batch_size": 1, "accumulation": 64}
        original.pop("training_rpn_score_threshold")
        import hashlib
        return original, hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    return config, sha256_file(path)


def fresh_output(project, output):
    output = output.resolve()
    if not output.is_relative_to(project / "outputs/training/experiment_f") or output.exists():
        raise ValueError("Use a fresh directory strictly below outputs/training/experiment_f.")
    output.mkdir(parents=True)
    return output


def implementation_hashes(project):
    return {p.relative_to(project).as_posix(): sha256_file(p) for p in
            (Path(__file__), *(Path(__file__).parent / name for name in
                              ("faster_rcnn_dataset.py", "faster_rcnn_training.py", "faster_rcnn_diagnostics.py",
                               "experiment_f_feasibility.py", "detection_ap_evaluation.py")))}


# An explicit official opt-in cannot bypass a failed or stale full-epoch smoke gate.
def verify_smoke_gate(project, path, config_hash, source_hashes):
    if path is None:
        raise ValueError("Official mode also requires --smoke-evidence from a successful complete smoke epoch.")
    path = path.resolve()
    if not path.is_relative_to(project / "outputs/training/experiment_f"):
        raise ValueError("Smoke evidence must be a local Experiment F run summary.")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (evidence.get("status") != "passed" or evidence.get("mode") != "smoke"
            or evidence.get("completed_epochs") != 1 or evidence.get("configuration_sha256") != config_hash
            or evidence.get("implementation_hashes") != source_hashes):
        raise ValueError("Official mode is blocked by a failed, incomplete or changed-protocol smoke gate.")
    epochs = evidence.get("epochs", [])
    if len(epochs) != 1:
        raise ValueError("A full training/validation record is required.")
    epoch = epochs[0]
    training = epoch.get("training", {})
    checks = epoch.get("checkpoint_checks", {})
    if (training.get("images") != 1211 or training.get("unique_images") != 1211
            or [u.get("images") for u in training.get("optimizer_updates", [])] != [64]*18 + [59]
            or training.get("all_losses_gradients_parameters_finite") is not True
            or epoch.get("validation_images") != 348
            or not all(checks.get(k, {}).get("model_tensors_exact") for k in ("best", "last"))
            or evidence.get("canonical_data_unchanged") is not True
            or evidence.get("protection_after", {}).get("unchanged") is not True
            or evidence.get("test_io_blocked_attempts") != 0):
        raise ValueError("Official mode requires all images, final optimizer window, validation and checkpoint gates.")
    return evidence


def verify_protection(project, protection):
    changed = [p for p, digest in protection.items() if sha256_file(project / p) != digest]
    if changed:
        raise ValueError(f"Preserved artifacts changed: {changed}")
    return {"files": len(protection), "unchanged": True}


def capture_preservation(project):
    # An ordinary checkout can generate its own guard without a private Stage 1 audit file.
    paths = [project / "requirements.txt", project / "pyproject.toml"]
    for directory in ("configs", "src", "tests", "docs/evidence", "docs/figures", "outputs/training/official"):
        base = project / directory
        paths.extend(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                     and p.name not in (".env", ".env.local"))
    return {p.relative_to(project).as_posix(): sha256_file(p) for p in paths}


def audit_d(project):
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.data.augment import LetterBox, RandomPerspective, RandomHSV
    from torchvision.models.detection.transform import GeneralizedRCNNTransform
    manifest_path = project / "outputs/training/official/D_seed42/run_manifest.json"
    args = json.loads(manifest_path.read_text(encoding="utf-8"))["applied_trainer_config"]["actual_args"]
    keys = ("batch", "nbs", "lr0", "lrf", "weight_decay", "warmup_epochs", "warmup_bias_lr", "momentum", "cos_lr", "amp", "epochs", "workers")
    # Simulate the source's actual warmup and step condition, including cross-epoch carry.
    import numpy as np
    batches, last, windows, previous_images = 152, -1, [], 0
    for ni in range(batches * 4):
        accumulate = max(1, int(np.interp(ni, [0, 456], [1, 8]).round())) if ni < 456 else 8
        if ni - last >= accumulate:
            epoch, index = divmod(ni, batches)
            seen = epoch * 1211 + min((index + 1) * 8, 1211)
            windows.append({"iteration": ni, "epoch": epoch + 1, "images": seen - previous_images,
                            "accumulation_batches": accumulate})
            previous_images, last = seen, ni
    return {"actual_args": {k: args[k] for k in keys}, "manifest_sha256": sha256_file(manifest_path),
            "nominal_accumulation_batches": 8, "scaled_weight_decay": args["weight_decay"] * 8 * 8 / 64,
            "warmup_iterations": 456, "batches_per_epoch": 152,
            "simulated_updates_by_epoch": [sum(w["epoch"] == e for w in windows) for e in range(1, 5)],
            "first_epoch_windows": [w for w in windows if w["epoch"] == 1],
            "source_semantics": {"loss": "YOLO loss multiplied by physical batch; accumulated without averaging; norm clipped at 10",
                                 "epoch_boundary": "no explicit final flush; gradients and last_opt_step carry across epochs; final training remainder can be unstepped",
                                 "lr": "epoch linear decay to lrf=.01; bias warmup from .1, other groups from 0; Adam beta1=.937",
                                 "ema": "D evaluates EMA; F evaluates direct FP32 model"},
            "source_files": {inspect.getfile(obj): sha256_file(Path(inspect.getfile(obj))) for obj in
                             (BaseTrainer, v8DetectionLoss, LetterBox, RandomPerspective, RandomHSV, GeneralizedRCNNTransform)}}


def verify_checkpoint(path, model, metadata, optimizer=None):
    import torch
    record = torch.load(path, map_location="cpu", weights_only=False)
    if record["metadata"] != metadata or record["epoch"] < 0:
        raise ValueError("Checkpoint metadata was not preserved.")
    expected = model.state_dict()
    if set(record["model"]) != set(expected) or not all(torch.equal(v, expected[k].detach().cpu()) for k, v in record["model"].items()):
        raise ValueError("Checkpoint tensors differ from the trained model.")
    if not record["optimizer"]["state"] or record["scheduler"]["images_seen"] < 1:
        raise ValueError("Checkpoint has no actual optimizer/scheduler progress.")
    if optimizer is not None:
        optimizer.load_state_dict(record["optimizer"])
    # Exercise loading, rather than only comparing serialized tensors.
    model.load_state_dict(record["model"])
    return {"model_tensors_exact": True, "optimizer_state_present": True, "scheduler_state_present": True,
            "rng_states_present": sorted(record["rng"]), "epoch": record["epoch"] + 1,
            "state_tensor_dtypes": sorted({str(v.dtype) for v in record["model"].values()})}


def run(project, output, *, mode, confirm_official=False, resume=None, smoke_evidence=None):
    if mode not in ("smoke", "official", "diagnostic") or (mode == "official" and not confirm_official):
        raise ValueError("Official training requires --confirm-official-100-epochs.")
    if resume is not None and mode != "official":
        raise ValueError("Smoke mode cannot resume or run a second epoch.")
    project = project.resolve()
    config, config_hash = load_config(project, original_diagnostic=mode == "diagnostic")
    source_hashes = implementation_hashes(project)
    gate = verify_smoke_gate(project, smoke_evidence, config_hash, source_hashes) if mode == "official" else None
    output = fresh_output(project, output)
    (output / "runtime").mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(output / "runtime")
    os.environ["TORCH_HOME"] = str(project / "outputs/training/experiment_f/weights_cache")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    summary = {"status": "running", "mode": mode, "created_at_utc": datetime.now(UTC).isoformat(),
               "configuration": config, "configuration_sha256": config_hash,
               "implementation_hashes": source_hashes,
               "requested_epochs": 1 if mode == "smoke" else 100, "completed_epochs": 0}
    started = time.perf_counter()
    with held_out_guard(project) as guard:
        try:
            import numpy as np
            import torch
            import torchvision
            import ultralytics
            from .faster_rcnn_training import (build_model, set_seed, make_optimizer, ImageSchedule, train_epoch,
                                              checkpoint_is_better, save_checkpoint, finite_step, reset_training_state)
            from .detection_ap_evaluation import shared_ap
            from .faster_rcnn_diagnostics import NumericalProbe, save_failure
            from .error_analysis_quantitative import evaluate_rows, aggregate_evaluations, size_group
            if (torch.__version__, torchvision.__version__, ultralytics.__version__) != ("2.6.0+cu126", "0.21.0+cu126", "8.4.155"):
                raise ValueError("The preserved CUDA dependency versions are required.")
            if not torch.cuda.is_available():
                raise ValueError("CUDA hardware feasibility cannot be measured on CPU.")
            device = torch.device("cuda:0")
            protection_path = project / "outputs/training/experiment_f/audit/protection_before.json"
            protection = capture_preservation(project)
            if protection_path.exists():
                historical = json.loads(protection_path.read_text(encoding="utf-8"))
                verify_protection(project, historical)
                protection.update(historical)
            _write_json(output / "protection_before.json", protection)
            summary["protection_before"] = verify_protection(project, protection)
            summary["d_protocol_audit"] = audit_d(project)
            train_samples, train_fp = load_split(project, "train")
            valid_samples, valid_fp = load_split(project, "valid")
            if {s.image_path.stem for s in train_samples} & {s.image_path.stem for s in valid_samples}:
                raise ValueError("Train/validation image identities overlap.")
            summary["dataset"] = {"train_images": len(train_samples), "valid_images": len(valid_samples),
                                  "train_fingerprint": train_fp, "valid_fingerprint": valid_fp,
                                  "train_annotations": sum(len(s.ground_truth) for s in train_samples),
                                  "valid_annotations": sum(len(s.ground_truth) for s in valid_samples),
                                  "train_empty": sum(not s.ground_truth for s in train_samples),
                                  "valid_empty": sum(not s.ground_truth for s in valid_samples),
                                  "lineage_verified": True, "identities_disjoint": True}
            if gate is not None and gate["dataset"] != summary["dataset"]:
                raise ValueError("Current data differs from the successful smoke gate.")
            d_rows, d_evidence = read_stage1_validation(project)
            summary["d_evaluation"] = {"official": d_evidence["official_validation"]["best_row"],
                                        "official_source": d_evidence["official_validation"]["reference"],
                                        "native": d_evidence["official_validation"]["observed"],
                                        "original_comparison": d_evidence["official_validation"]["comparison"],
                                        "shared": shared_ap(d_rows, samples=valid_samples),
                                        "prediction_sha256": d_evidence["prediction_export"]["sha256"]}
            train = FractureDataset(train_samples, split="train")
            valid = FractureDataset(valid_samples, split="valid")
            set_seed(config["seed"])
            model, model_report = build_model(None if mode == "diagnostic" else config)
            summary["model"] = model_report
            summary["runtime"] = {"torch": torch.__version__, "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
                                  "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0),
                                  "gpu_capacity_bytes": torch.cuda.get_device_properties(0).total_memory,
                                  "deterministic_warn_only": True, "amp": False, "tf32": False,
                                  "device": "cuda:0", "physical_batch": config["batch_size"], "workers": 0}
            model.to(device)
            probe = NumericalProbe(model) if mode == "diagnostic" else None
            last_proposals = {}
            # Capture the actual RPN output before ROI sampling, including a possible empty proposal list.
            def record_proposals(_module, _inputs, result):
                last_proposals["counts"] = [len(boxes) for boxes in result[0]]
            proposal_handle = model.rpn.register_forward_hook(record_proposals)
            optimizer = make_optimizer(model, config)
            scheduler = ImageSchedule(optimizer, config, len(train))
            metadata = {"configuration": config, "configuration_sha256": config_hash,
                        "dataset": summary["dataset"], "model": model_report, "seed": 42,
                        "mode": mode, "implementation_hashes": source_hashes}
            best, first_epoch = -1.0, 0
            if resume is not None:
                resume = resume.resolve()
                if not resume.is_relative_to(project / "outputs/training/experiment_f"):
                    raise ValueError("Resume requires a trusted local F checkpoint.")
                saved = torch.load(resume, map_location=device, weights_only=False)
                if saved.get("resumable") is False:
                    raise ValueError("A partial failure capture is diagnostic only and cannot resume training.")
                if saved["metadata"] != metadata or saved["epoch"] not in range(99):
                    raise ValueError("Resume mode, sources, data or configuration differ from the checkpoint.")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                best, first_epoch = saved["best_score"], saved["epoch"] + 1
                import random
                random.setstate(saved["rng"]["python"])
                np.random.set_state(saved["rng"]["numpy"])
                torch.set_rng_state(saved["rng"]["torch"].cpu())
                torch.cuda.set_rng_state_all([s.cpu() for s in saved["rng"]["cuda"]])
                shutil.copyfile(resume.parent / "best.pt", output / "best.pt")
                summary["resumed_from"] = {"sha256": sha256_file(resume), "completed_epoch": first_epoch}
            else:
                # Preflight covers one annotated and one empty-label image. Restore all state afterward.
                initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.train()
                preflight = []
                for empty in (False, True):
                    index = next(i for i,s in enumerate(train_samples) if (not s.ground_truth) == empty)
                    image, target, _ = train[index]
                    losses = model([image.to(device)], [{k: v.to(device) for k,v in target.items()}])
                    if not all(torch.isfinite(v) for v in losses.values()):
                        raise FloatingPointError("Preflight loss failed.")
                    sum(losses.values()).backward()
                    optimizer.param_groups[0]["lr"] = config["lr"] / 100
                    optimizer.param_groups[1]["lr"] = config["lr"] / 100
                    norm = finite_step(model, optimizer)
                    preflight.append({"empty_label": empty, "losses": {k: float(v.detach()) for k,v in losses.items()},
                                      "gradient_norm": norm, "input_shape": list(image.shape)})
                    del image, target, losses
                optimizer, scheduler = reset_training_state(model, initial_state, config, len(train))
                del initial_state
                summary["preflight"] = {"passed": True, "examples": preflight, "model_optimizer_rng_reset_before_epoch": True}
                summary["preflight_cuda_memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            summary["startup_seconds"] = time.perf_counter() - started
            _write_json(output / "run_summary.json", summary)
            epochs = []
            for epoch in range(first_epoch, config["epochs"] if mode == "official" else 1):
                stats = train_epoch(model, train, optimizer, scheduler, epoch=epoch, config=config, device=device,
                                    progress_path=output / f"epoch_{epoch+1:03d}_updates.jsonl",
                                    max_images=500 if mode == "diagnostic" else None)
                if mode == "diagnostic":
                    summary["diagnostic_statistics"] = stats
                    summary["status"] = "bounded_diagnostic_complete"
                    summary["protection_after"] = verify_protection(project, protection)
                    return summary
                summary["completed_training_epoch"] = epoch + 1
                summary["last_training_statistics"] = stats
                _write_json(output / "run_summary.json", summary)
                model.eval()
                if model.rpn.score_thresh != config["evaluation"]["rpn_score_threshold"]:
                    raise ValueError("Inference RPN threshold was not restored.")
                val_start, rows = time.perf_counter(), []
                with torch.inference_mode():
                    for i, sample in enumerate(valid_samples):
                        image, _, info = valid[i]
                        prediction = model([image.to(device)])[0]
                        rows.append(prediction_row(sample, prediction, info))
                        del image, prediction
                torch.cuda.synchronize()
                validation_seconds = time.perf_counter() - val_start
                verify_sample_export(rows, valid_samples, split="valid")
                ap_start = time.perf_counter()
                ap = shared_ap(rows, samples=valid_samples)
                ap_seconds = time.perf_counter() - ap_start
                export = output / f"epoch_{epoch+1:03d}_validation_predictions.jsonl"
                export.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
                validate_export_rows(rows)
                # Verify direct compatibility; these are smoke diagnostics, not final D-F findings.
                diagnostics = {"class_aware": aggregate_evaluations(evaluate_rows(rows, confidence=.25, iou=.5)),
                               "class_agnostic": aggregate_evaluations(evaluate_rows(rows, confidence=.25, iou=.5, class_aware=False))}
                for row in rows:
                    from .detection_evaluation import GroundTruth
                    for g in row["ground_truth"]:
                        size_group(GroundTruth(g["box_id"], g["class_id"], tuple(g["xyxy"])), row)
                _write_json(output / f"epoch_{epoch+1:03d}_diagnostics.json", diagnostics)
                checkpoint_start = time.perf_counter()
                better = checkpoint_is_better(ap["map50_95"], best)
                best = max(best, ap["map50_95"])
                last = save_checkpoint(output / "last.pt", model, optimizer, scheduler, epoch=epoch, best_score=best, metadata=metadata)
                if better:
                    shutil.copyfile(output / "last.pt", output / "best.pt")
                checkpoint_checks = {"last": verify_checkpoint(output / "last.pt", model, metadata, optimizer)}
                if better:
                    checkpoint_checks["best"] = verify_checkpoint(output / "best.pt", model, metadata)
                checkpoint_seconds = time.perf_counter() - checkpoint_start
                record = {"epoch": epoch + 1, "training": stats, "validation_seconds": validation_seconds,
                          "validation_images": len(rows), "validation_rpn_threshold": model.rpn.score_thresh,
                          "ap_seconds": ap_seconds, "shared_ap": ap, "last_checkpoint": last,
                          "checkpoint_checks": checkpoint_checks, "checkpoint_seconds": checkpoint_seconds,
                          "prediction_sha256": sha256_file(export), "diagnostic_compatibility_passed": True}
                record["best_checkpoint_sha256"] = sha256_file(output / "best.pt")
                epochs.append(record)
                summary["completed_epochs"] = epoch + 1
                summary["epochs"] = epochs
                _write_json(output / "run_summary.json", summary)
            if config_hash != sha256_file(project / "configs/experiments/F.yaml"):
                raise ValueError("F configuration changed during execution.")
            root = project / config["dataset"]
            if train_fp != _diagnostic_fingerprint(root, train_samples, split="train") or valid_fp != _diagnostic_fingerprint(root, valid_samples, split="valid"):
                raise ValueError("Canonical train/validation files changed during execution.")
            summary["protection_after"] = verify_protection(project, protection)
            summary["canonical_data_unchanged"] = True
            summary["status"] = "passed"
        except Exception as error:
            summary["status"] = "failed"
            summary["failure"] = {"type": type(error).__name__, "message": str(error),
                                  "settings_automatically_changed": False}
            if hasattr(error, "evidence"):
                error.evidence["rpn_proposals_per_image"] = last_proposals.get("counts")
                error.evidence["intermediates"] = probe.state if probe is not None else {}
                error.evidence["runtime"] = summary.get("runtime")
                error.evidence["cuda_memory_at_failure"] = {
                    "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                summary["failure"]["numerical_evidence"] = error.evidence
                # Preserve the actual failing model and partial gradients without presenting a resumable epoch.
                failure_path = output / "failure_capture.pt"
                save_failure(failure_path, model, optimizer, scheduler, metadata, error.evidence)
                summary["failure"]["diagnostic_state"] = {"path": str(failure_path), "sha256": sha256_file(failure_path),
                                                            "resumable": False}
            raise
        finally:
            if "probe" in locals() and probe is not None:
                probe.close()
            if "torch" in locals() and torch.cuda.is_available():
                summary["cuda_memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                          "capacity_bytes": torch.cuda.get_device_properties(0).total_memory}
            summary["total_seconds"] = time.perf_counter() - started
            summary["test_io_blocked_attempts"] = guard["blocked_attempts"]
            summary["official_training_executed"] = mode == "official"
            _write_json(output / "run_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("smoke", "official", "diagnostic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-official-100-epochs", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-evidence", type=Path, help="Required successful, matching full-epoch gate for official mode.")
    args = parser.parse_args(argv)
    result = run(args.project_root, args.output, mode=args.mode,
                 confirm_official=args.confirm_official_100_epochs, resume=args.resume, smoke_evidence=args.smoke_evidence)
    print(json.dumps({"status": result["status"], "completed_epochs": result["completed_epochs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
