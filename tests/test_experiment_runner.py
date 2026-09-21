from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from bone_fracture_pipeline.augmentation_policy import AUGMENTATION_SETTINGS
from bone_fracture_pipeline.experiment_runner import (
    build_training_arguments,
    resolve_experiment_config,
    validate_experiment_matrix,
    verify_applied_trainer_config,
    write_runtime_dataset_yaml,
)
from bone_fracture_pipeline.training_protocol import AUGMENTATION_OFF_SETTINGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentRunnerTests(unittest.TestCase):
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
            "amp": True,
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


if __name__ == "__main__":
    unittest.main()
