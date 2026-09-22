from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from bone_fracture_pipeline.augmentation_policy import AUGMENTATION_SETTINGS
from bone_fracture_pipeline.experiment_runner import (
    build_training_arguments,
    resolve_experiment_config,
    summarize_smoke_runs,
    validate_experiment_matrix,
    verify_applied_trainer_config,
    verify_trainer_data_routing,
    write_runtime_dataset_yaml,
)
from bone_fracture_pipeline.training_protocol import AUGMENTATION_OFF_SETTINGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentRunnerTests(unittest.TestCase):
    def test_e_is_a_separate_follow_up_with_d_training_conditions(self) -> None:
        d = resolve_experiment_config(PROJECT_ROOT, "D")
        e = resolve_experiment_config(PROJECT_ROOT, "E")
        matrix, _ = validate_experiment_matrix(PROJECT_ROOT)
        self.assertEqual(set(matrix), {"A", "B", "C", "D"})
        for section in ("model", "training", "augmentation", "evaluation", "seed_policy", "framework"):
            self.assertEqual(e["resolved"][section], d["resolved"][section])
        self.assertEqual(e["resolved"]["dataset"]["expected_classes"], ["fracture"])
        self.assertEqual(e["target_formulation"]["class_mapping"], {str(index): 0 for index in range(6)})
        self.assertEqual(e["target_formulation"]["source_fingerprint"], d["resolved"]["dataset"]["expected_fingerprint"])
        self.assertEqual(e["image_condition"], "clahe")
        self.assertEqual(e["augmentation_condition"], "conservative")

    def test_e_commands_route_only_train_and_validation_to_single_class_data(self) -> None:
        e = resolve_experiment_config(PROJECT_ROOT, "E")
        smoke, smoke_dir, _ = build_training_arguments(
            PROJECT_ROOT, e, smoke=True, require_new_output=False
        )
        official, official_dir, _ = build_training_arguments(
            PROJECT_ROOT, e, smoke=False, require_new_output=False
        )
        self.assertEqual(smoke["epochs"], 1)
        self.assertEqual(official["epochs"], 100)
        self.assertEqual(smoke["batch"], 8)
        self.assertFalse(smoke["amp"])
        self.assertEqual(smoke["degrees"], 10.0)
        self.assertEqual(smoke["translate"], 0.05)
        self.assertEqual(smoke_dir.name, "E_seed42")
        self.assertEqual(official_dir.name, "E_seed42")
        self.assertIn("/smoke/", smoke_dir.as_posix())
        self.assertIn("/official/", official_dir.as_posix())
        self.assertEqual(e["resolved"]["dataset"]["split_policy"]["validation"], "valid")
        self.assertEqual(e["resolved"]["dataset"]["split_policy"]["test_usage"], "final_evaluation_only")
        with tempfile.TemporaryDirectory() as directory:
            runtime = write_runtime_dataset_yaml(PROJECT_ROOT, e, Path(directory))
            data = yaml.safe_load((Path(directory) / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["path"], str((PROJECT_ROOT / "data/prepared/v3_detection_clahe_single_class").resolve()))
        self.assertEqual(data["names"], {0: "fracture"})
        self.assertEqual(data["train"], "train/images")
        self.assertEqual(data["val"], "valid/images")
        self.assertEqual(runtime["source"], "data/prepared/v3_detection_clahe_single_class/data.yaml")

    def test_e_trainer_routing_rejects_a_test_image(self) -> None:
        e = resolve_experiment_config(PROJECT_ROOT, "E")
        root = PROJECT_ROOT / e["resolved"]["dataset"]["root"]
        train_files = [str(root / "train/images" / f"train-{index}.png") for index in range(1211)]
        valid_files = [str(root / "valid/images" / f"valid-{index}.png") for index in range(348)]
        trainer = SimpleNamespace(
            train_loader=SimpleNamespace(dataset=SimpleNamespace(im_files=train_files)),
            validator=SimpleNamespace(dataloader=SimpleNamespace(dataset=SimpleNamespace(im_files=valid_files))),
        )
        result = verify_trainer_data_routing(trainer, PROJECT_ROOT, e["resolved"])
        self.assertEqual(result["training_images_loaded"], 1211)
        self.assertEqual(result["validation_images_loaded"], 348)
        self.assertEqual(result["test_images_loaded"], 0)
        trainer.train_loader.dataset.im_files[0] = str(root / "test/images/heldout.png")
        with self.assertRaisesRegex(Exception, "non-training image"):
            verify_trainer_data_routing(trainer, PROJECT_ROOT, e["resolved"])

    def test_matrix_contains_only_the_two_planned_factors(self) -> None:
        resolutions, result = validate_experiment_matrix(PROJECT_ROOT)

        self.assertTrue(result["only_planned_factors_differ"])
        self.assertEqual(set(resolutions), {"A", "B", "C", "D"})
        self.assertEqual(
            result["comparisons"]["A_vs_B"]["different_paths"],
            ["dataset.data_yaml", "dataset.expected_fingerprint", "dataset.root"],
        )
        self.assertEqual(
            result["comparisons"]["A_vs_C"]["different_paths"],
            [
                "augmentation.degrees",
                "augmentation.hsv_v",
                "augmentation.scale",
                "augmentation.translate",
            ],
        )

    def test_experiments_resolve_to_approved_dataset_and_augmentation_conditions(self) -> None:
        a = resolve_experiment_config(PROJECT_ROOT, "A")
        b = resolve_experiment_config(PROJECT_ROOT, "B")
        c = resolve_experiment_config(PROJECT_ROOT, "C")
        d = resolve_experiment_config(PROJECT_ROOT, "D")

        self.assertEqual(a["resolved"]["augmentation"], AUGMENTATION_OFF_SETTINGS)
        self.assertEqual(b["resolved"]["augmentation"], AUGMENTATION_OFF_SETTINGS)
        self.assertEqual(c["resolved"]["augmentation"], AUGMENTATION_SETTINGS)
        self.assertEqual(d["resolved"]["augmentation"], AUGMENTATION_SETTINGS)
        self.assertEqual(a["resolved"]["dataset"]["root"], c["resolved"]["dataset"]["root"])
        self.assertEqual(b["resolved"]["dataset"]["root"], d["resolved"]["dataset"]["root"])
        self.assertNotEqual(a["resolved"]["dataset"]["root"], b["resolved"]["dataset"]["root"])

    def test_smoke_changes_only_epochs_and_keeps_official_protocol_values(self) -> None:
        resolution = resolve_experiment_config(PROJECT_ROOT, "A")
        smoke, smoke_directory, smoke_kind = build_training_arguments(
            PROJECT_ROOT, resolution, smoke=True, require_new_output=False
        )
        official, official_directory, official_kind = build_training_arguments(
            PROJECT_ROOT, resolution, smoke=False, require_new_output=False
        )

        self.assertEqual(smoke["epochs"], 1)
        self.assertEqual(official["epochs"], 100)
        self.assertEqual(smoke_kind, "smoke")
        self.assertEqual(official_kind, "official")
        self.assertEqual(smoke_directory.name, "A_seed42")
        self.assertEqual(official_directory.name, "A_seed42")
        different = {key for key in smoke if smoke[key] != official[key]}
        self.assertEqual(different, {"data", "epochs", "project"})
        for key, value in {
            "batch": 8,
            "imgsz": 640,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "weight_decay": 0.0005,
            "seed": 42,
            "deterministic": True,
            "workers": 8,
            "amp": False,
            "val": True,
        }.items():
            self.assertEqual(smoke[key], value)

    def test_resolved_split_policy_keeps_test_for_final_evaluation_only(self) -> None:
        for experiment in ("A", "B", "C", "D"):
            resolution = resolve_experiment_config(PROJECT_ROOT, experiment)
            split_policy = resolution["resolved"]["dataset"]["split_policy"]

            self.assertEqual(split_policy["training"], "train")
            self.assertEqual(split_policy["validation"], "valid")
            self.assertEqual(split_policy["checkpoint_selection"], "valid")
            self.assertEqual(split_policy["test_usage"], "final_evaluation_only")

    def test_runtime_dataset_yaml_uses_the_frozen_absolute_dataset_root(self) -> None:
        resolution = resolve_experiment_config(PROJECT_ROOT, "A")
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            result = write_runtime_dataset_yaml(PROJECT_ROOT, resolution, run_directory)
            runtime = yaml.safe_load((run_directory / "dataset.yaml").read_text(encoding="utf-8"))

        self.assertEqual(runtime["path"], str((PROJECT_ROOT / "data/prepared/v3_detection_png").resolve()))
        self.assertEqual(runtime["train"], "train/images")
        self.assertEqual(runtime["val"], "valid/images")
        self.assertEqual(result["source"], "data/prepared/v3_detection_png/data.yaml")

    def test_trainer_device_string_is_equivalent_to_frozen_cuda_device_zero(self) -> None:
        resolution = resolve_experiment_config(PROJECT_ROOT, "A")
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            expected, _, _ = build_training_arguments(
                PROJECT_ROOT,
                resolution,
                smoke=True,
                require_new_output=False,
            )
            actual = dict(expected)
            actual["device"] = "0"
            (run_directory / "args.yaml").write_text(
                yaml.safe_dump(actual, sort_keys=False),
                encoding="utf-8",
            )

            result = verify_applied_trainer_config(run_directory, expected)

        self.assertTrue(result["valid"])
        self.assertEqual(result["differences"], {})

    def test_partial_smoke_summary_records_failure_and_unrun_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_directory = root / "outputs/training/smoke/A_seed42"
            run_directory.mkdir(parents=True)
            manifest = {
                "run_kind": "smoke",
                "status": "failed",
                "configuration": {"image_condition": "png_control", "augmentation_condition": "off"},
                "trainer_arguments": {"batch": 8},
                "failure": {"type": "ExperimentValidationError", "message": "Non-finite validation loss"},
            }
            (run_directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run_directory / "results.csv").write_text(
                "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/mAP50(B),"
                "metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n"
                "1,3.2,27.0,3.0,0,0,nan,nan,nan\n",
                encoding="utf-8",
            )

            summary = summarize_smoke_runs(root)

            self.assertFalse(summary["success"])
            self.assertEqual(summary["phase2f_status"], "blocked")
            self.assertEqual(summary["runs"]["A"]["status"], "failed")
            self.assertEqual(summary["runs"]["B"]["status"], "not_run")
            csv_rows = (root / "docs/evidence/phase2f/smoke_run_summary.csv").read_text(encoding="utf-8")
            self.assertIn("A,png_control,off", csv_rows)
            self.assertIn("B,clahe,off", csv_rows)

    def test_complete_smoke_summary_records_measured_batch_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_directory = root / "docs/evidence/phase2f"
            evidence_directory.mkdir(parents=True)
            (evidence_directory / "environment_freeze.json").write_text(
                json.dumps({"batch_size_status": "planned_not_smoke_tested"}),
                encoding="utf-8",
            )
            for index, experiment in enumerate(("A", "B", "C", "D")):
                run_directory = root / f"outputs/training/smoke/{experiment}_seed42"
                (run_directory / "weights").mkdir(parents=True)
                (run_directory / "weights/last.pt").touch()
                (run_directory / "weights/best.pt").touch()
                manifest = {
                    "run_kind": "smoke",
                    "status": "completed",
                    "configuration": {
                        "image_condition": "png_control" if experiment in {"A", "C"} else "clahe",
                        "augmentation_condition": "conservative" if experiment in {"C", "D"} else "off",
                    },
                    "trainer_arguments": {"batch": 8},
                    "output_verification": {
                        "epochs_completed": 1,
                        "final_values": {
                            "train/box_loss": 1.0,
                            "train/cls_loss": 1.0,
                            "train/dfl_loss": 1.0,
                            "val/box_loss": 1.0,
                            "val/cls_loss": 1.0,
                            "val/dfl_loss": 1.0,
                            "metrics/mAP50(B)": 0.0,
                            "metrics/mAP50-95(B)": 0.0,
                        },
                    },
                    "runtime": {"seconds": 10.0},
                    "gpu_memory": {
                        "batch_size_8_stable": True,
                        "peak_allocated_gib": 2.0 + index / 10,
                        "peak_reserved_gib": 3.0 + index / 10,
                    },
                    "validation_inference": {"valid": True},
                }
                (run_directory / "run_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

            summary = summarize_smoke_runs(root)
            environment = json.loads(
                (evidence_directory / "environment_freeze.json").read_text(encoding="utf-8")
            )

            self.assertTrue(summary["success"])
            self.assertEqual(environment["batch_size_status"], "smoke_validated_all_four_runs")
            self.assertEqual(environment["smoke_observation"]["maximum_peak_allocated_gib"], 2.3)


if __name__ == "__main__":
    unittest.main()
