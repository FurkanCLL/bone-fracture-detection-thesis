from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from bone_fracture_pipeline.augmentation_policy import (
    AUGMENTATION_SETTINGS,
    MatchedSample,
    compare_dataset_identities,
    generate_preview_package,
    load_augmentation_config,
    validate_augmentation_config,
    verify_framework_semantics,
    verify_paired_reproducibility,
)
from bone_fracture_pipeline.prepare_dataset import (
    CLASS_NAMES,
    DatasetExpectations,
    SplitExpectation,
    fingerprint_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "augmentation_conservative.yaml"


class AugmentationPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_augmentation_config(CONFIG_PATH)

    def test_policy_contains_only_approved_conservative_transforms(self) -> None:
        result = validate_augmentation_config(self.config)

        self.assertTrue(result["valid"])
        self.assertEqual(self.config["augmentation"], AUGMENTATION_SETTINGS)
        self.assertEqual(self.config["split_policy"], {"train": True, "valid": False, "test": False})
        self.assertEqual(self.config["augmentation"]["degrees"], 10.0)
        self.assertEqual(self.config["augmentation"]["translate"], 0.05)
        self.assertEqual(self.config["augmentation"]["scale"], 0.10)
        self.assertEqual(self.config["augmentation"]["hsv_v"], 0.15)

    def test_installed_framework_semantics_match_policy(self) -> None:
        result = verify_framework_semantics(PROJECT_ROOT, self.config)

        self.assertTrue(result["valid"])
        self.assertEqual(result["ultralytics_version"], "8.4.155")
        self.assertFalse(result["validation_augmentation"])
        self.assertFalse(result["test_augmentation"])
        self.assertFalse(result["albumentations_installed"])

    def test_stable_identity_and_reproducible_c_d_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control"
            clahe = root / "clahe"
            expectations = _create_dataset_pair(control, clahe)

            samples, result = compare_dataset_identities(
                control,
                clahe,
                expectations=expectations,
                expected_clahe_fingerprint=fingerprint_dataset(clahe).digest,
            )
            paired = verify_paired_reproducibility(samples, self.config)

            self.assertEqual(result["matched_samples"], 8)
            self.assertEqual(result["label_hash_mismatches"], 0)
            self.assertTrue(paired["transformed_geometry_identical"])
            self.assertTrue(paired["python_rng_states_identical"])
            self.assertTrue(paired["numpy_rng_states_identical"])

    def test_preview_is_train_only_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control"
            clahe = root / "clahe"
            expectations = _create_dataset_pair(control, clahe)
            samples, _ = compare_dataset_identities(
                control,
                clahe,
                expectations=expectations,
                expected_clahe_fingerprint=fingerprint_dataset(clahe).digest,
            )

            first = generate_preview_package(samples, root / "first", root / "first", root)
            second = generate_preview_package(samples, root / "second", root / "second", root)

            self.assertNotIn("test", {row["split"] for row in first})
            self.assertEqual([row["preview_case"] for row in first], [row["preview_case"] for row in second])
            self.assertEqual([row["output_boxes"] for row in first], [row["output_boxes"] for row in second])
            self.assertTrue((root / "first" / "contact_sheet.png").is_file())
            first_hashes = [path.read_bytes() for path in sorted((root / "first").glob("*.png"))]
            second_hashes = [path.read_bytes() for path in sorted((root / "second").glob("*.png"))]
            self.assertEqual(first_hashes, second_hashes)


def _create_dataset_pair(control: Path, clahe: Path) -> DatasetExpectations:
    class_counts = [0] * len(CLASS_NAMES)
    split_entries: dict[str, list[tuple[str, str, int]]] = {
        "train": [],
        "valid": [("valid-empty", "", 80)],
        "test": [("test-empty", "", 100)],
    }
    for class_id in range(len(CLASS_NAMES)):
        label = f"{class_id} 0.5000000000 0.5000000000 0.2000000000 0.1800000000\n"
        split_entries["train"].append((f"train-class-{class_id}", label, 30 + class_id * 20))
        class_counts[class_id] += 1
    split_entries["train"][0] = (
        "train-class-0",
        "0 0.5000000000 0.5000000000 0.2000000000 0.1800000000\n"
        "1 0.1200000000 0.1400000000 0.0500000000 0.0600000000\n",
        30,
    )
    class_counts[1] += 1

    yaml_text = (
        "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES))
    )
    for root in (control, clahe):
        root.mkdir(parents=True)
        (root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    for split, entries in split_entries.items():
        for root in (control, clahe):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
        for stem, label, level in entries:
            gradient = np.tile(np.arange(64, dtype=np.uint8), (48, 1))
            control_image = np.dstack(
                (
                    np.clip(level + gradient, 0, 255),
                    np.clip(level + gradient + 2, 0, 255),
                    np.clip(level + gradient + 4, 0, 255),
                )
            ).astype(np.uint8)
            clahe_channel = np.clip(level + gradient + 10, 0, 255).astype(np.uint8)
            clahe_image = np.repeat(clahe_channel[:, :, None], 3, axis=2)
            cv2.imwrite(str(control / split / "images" / f"{stem}.png"), control_image)
            cv2.imwrite(str(clahe / split / "images" / f"{stem}.png"), clahe_image)
            for root in (control, clahe):
                (root / split / "labels" / f"{stem}.txt").write_text(label, encoding="utf-8")

    return DatasetExpectations(
        class_names=CLASS_NAMES,
        splits={
            "train": SplitExpectation(6, 7, 0, tuple(class_counts)),
            "valid": SplitExpectation(1, 0, 1, (0, 0, 0, 0, 0, 0)),
            "test": SplitExpectation(1, 0, 1, (0, 0, 0, 0, 0, 0)),
        },
    )


if __name__ == "__main__":
    unittest.main()
