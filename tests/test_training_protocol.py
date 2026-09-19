from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from bone_fracture_pipeline.prepare_dataset import CLASS_NAMES, fingerprint_dataset
from bone_fracture_pipeline.training_protocol import (
    AUGMENTATION_OFF_SETTINGS,
    ProtocolValidationError,
    collect_environment_metadata,
    load_protocol_config,
    protocol_digest,
    validate_prepared_dataset,
    validate_protocol_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "training" / "baseline.yaml"


class TrainingProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_protocol_config(BASELINE_CONFIG)

    def test_baseline_contains_approved_fixed_values(self) -> None:
        result = validate_protocol_config(self.config)

        self.assertTrue(result["valid"])
        self.assertEqual(self.config["model"]["weights"], "yolov8s.pt")
        self.assertEqual(self.config["training"]["epochs"], 100)
        self.assertEqual(self.config["training"]["batch"], 8)
        self.assertEqual(self.config["training"]["seed"], 42)
        self.assertEqual(self.config["training"]["patience"], 0)

    def test_every_baseline_augmentation_is_explicitly_disabled(self) -> None:
        validate_protocol_config(self.config)

        self.assertEqual(self.config["augmentation"], AUGMENTATION_OFF_SETTINGS)

    def test_test_split_cannot_be_used_for_checkpoint_selection(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["evaluation"]["checkpoint_selection"]["split"] = "test"

        with self.assertRaisesRegex(ProtocolValidationError, "checkpoint_selection.split"):
            validate_protocol_config(changed)

    def test_protocol_digest_is_independent_of_mapping_order(self) -> None:
        reversed_config = dict(reversed(list(self.config.items())))

        self.assertEqual(protocol_digest(self.config), protocol_digest(reversed_config))

    def test_dataset_validation_detects_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "data" / "prepared" / "v3_detection"
            self._create_minimal_dataset(dataset_root)
            changed = copy.deepcopy(self.config)
            changed["dataset"]["expected_fingerprint"] = "0" * 64

            with self.assertRaisesRegex(ProtocolValidationError, "fingerprint"):
                validate_prepared_dataset(root, changed)

    def test_dataset_validation_requires_all_six_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "data" / "prepared" / "v3_detection"
            self._create_minimal_dataset(dataset_root, class_names=CLASS_NAMES[:-1])
            changed = copy.deepcopy(self.config)
            changed["dataset"]["expected_fingerprint"] = fingerprint_dataset(dataset_root).digest

            with self.assertRaisesRegex(ProtocolValidationError, "six approved classes"):
                validate_prepared_dataset(root, changed)

    def test_environment_metadata_hashes_weights_and_records_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "fixture.pt"
            weights.write_bytes(b"test weights")

            metadata = collect_environment_metadata(root, self.config, weights)

            self.assertEqual(metadata["weights"]["filename"], "fixture.pt")
            self.assertEqual(len(metadata["weights"]["sha256"]), 64)
            self.assertEqual(metadata["planned_batch_size"], 8)
            self.assertIn("pytorch", metadata["libraries"])
            self.assertIn("cuda", metadata)

    @staticmethod
    def _create_minimal_dataset(dataset_root: Path, class_names: tuple[str, ...] = CLASS_NAMES) -> None:
        data = {
            "path": ".",
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "names": {index: name for index, name in enumerate(class_names)},
        }
        dataset_root.mkdir(parents=True)
        (dataset_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        for split in ("train", "valid", "test"):
            images = dataset_root / split / "images"
            labels = dataset_root / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / f"{split}.jpg").write_bytes(f"{split} image".encode())
            (labels / f"{split}.txt").write_text("", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
