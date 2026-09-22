# Experiment E: single-class fracture localization preparation

**Status:** Derived dataset, configuration, tests, and one-epoch smoke gate passed on 2026-09-22. The official 100-epoch E run has **not** been launched.

## Scientific question and relationship to D

Experiment E asks whether a single `fracture` output target can improve class-agnostic localization relative to the six anatomical fracture-related targets. The source-semantics case and its uncertainty are documented in [the class investigation](POSITIVE_VS_FRACTURE_CLASS_INVESTIGATION.md). E is a follow-up to the completed A/B/C/D matrix, not a fifth cell in that frozen 2×2 design.

E inherits the same YOLOv8s COCO-pretrained baseline and conservative policy as D. Both use the same CLAHE image condition, train/validation identities, image size 640, batch size 8, 100-epoch official budget, AdamW, learning rate 0.001, weight decay 0.0005, seed 42, deterministic settings, AMP off, and validation mAP50–95 `best.pt` selection. The resolved configuration check permits only the derived dataset path, fingerprint, and expected class list to differ from D. The original A/B/C/D configurations, datasets, runs, and Phase 2 evidence were not changed.

## Dataset lineage and exact change

The new ignored derived dataset is `data/prepared/v3_detection_clahe_single_class`, built from D's `data/prepared/v3_detection_clahe`. Its `data.yaml` declares only `names: {0: fracture}` with the same train, validation, and test path names. Every non-empty source detection row has its one-byte class ID (`0` through `5`) replaced by `0`; all coordinate bytes and line endings are kept. Thus each of the six original labels maps to `fracture` without changing a box, adding or dropping annotations, or changing empty files.

| Split | Images and labels | Boxes | Empty labels |
|---|---:|---:|---:|
| Train | 1,211 | 698 | 607 |
| Validation | 348 | 204 | 175 |
| Test | 169 | 96 | 86 |
| **Total** | **1,728** | **998** | **868** |

The source CLAHE fingerprint remains `ca8286b35d3d31f0b8074aa387a44ca6392178926c23bde61ea6d99be70aba5f`. The derived dataset fingerprint is `f3ad8c1ecb87ab63ecc609d6b2cf853bfadd66fa6ddd698bb681e6d4c8f20908` across 3,457 files. Validator checks found 1,728 matching image hashes, 1,728 exact remapped label files, matched relative paths and split identities, the expected split counts, valid one-class detection rows, and no source-fingerprint drift. Identical image bytes also preserve decoded dimensions and pixels. These checks were repeated after the smoke run. The builder refuses to replace an existing derived dataset.

The test files were included only in automated file integrity and split-membership checks needed to establish dataset lineage. No test image was visually inspected, and no test data was loaded by the trainer, used for validation, or evaluated for metrics.

## Reproducibility and runner safeguards

`configs/experiments/E.yaml` records E's target formulation, explicit six-to-one mapping, D source root/fingerprint, E derived root/fingerprint, and the shared baseline/augmentation-policy paths. The A/B/C/D matrix validator still covers exactly A, B, C, and D. E resolution separately checks that D and E differ only in target dataset fields. Its preflight validates the baseline, installed environment and pretrained-weight hash, data YAML, full dataset fingerprint, per-file lineage, augmentation semantics, and train/validation/test split policy.

The runner retains fresh-output protection, trainer-argument verification, finite recorded metrics and losses, checkpoint/output checks, train/validation loader path checks, fixed validation-only inference, and source-fingerprint restoration after generated cache cleanup. E's run manifest adds its follow-up role, target formulation, class mapping, source lineage, and derived fingerprint. The validation inference check also verifies that `best.pt` exposes only class `0: fracture` and that predicted class IDs, if any, are zero.

To rebuild on a clean checkout after reproducing the approved D CLAHE dataset, run:

```powershell
.venv\Scripts\python.exe -m bone_fracture_pipeline.single_class_dataset
.venv\Scripts\python.exe -m bone_fracture_pipeline.single_class_dataset --validate-only
```

The builder creates `outputs/phase3e/single_class/preparation_validation.json`; its compact checked facts are in [`docs/evidence/phase3e/single_class_validation.json`](evidence/phase3e/single_class_validation.json). The dataset and full generated evidence remain ignored by Git.

## Automated and smoke validation

The full project suite passed: **79 tests** with `.venv\Scripts\python.exe -m unittest discover -s tests -v`. New tests cover class remapping, exact geometry-token and image preservation, empty files and split identities, invalid-source and drift rejection, fingerprint stability, E config resolution, runner data paths, test-path rejection, and unchanged A/B/C/D matrix behavior.

The smoke command was:

```powershell
.venv\Scripts\python.exe -m bone_fracture_pipeline.experiment_runner --experiment E --smoke
```

The one-epoch run completed in `outputs/training/smoke/E_seed42`. It initialized one YOLOv8s output class, used the CLAHE derived dataset and conservative augmentation settings (`degrees=10`, `translate=0.05`, `scale=0.10`, `hsv_v=0.15`), finished training and validation, produced finite recorded losses and metrics, generated `best.pt` and `last.pt`, and passed fixed-sample validation inference. The manifest records 1,211 train images, 348 validation images, zero test images, batch 8, AMP off, no CUDA OOM, peak allocated/reserved memory 3.087/3.686 GiB, and restoration of the derived fingerprint. Saved train and validation batch overlays were inspected for technical label placement; no medical correctness conclusion was drawn.

The one-epoch validation mAP50–95 recorded in `results.csv` is `0.0` at its saved precision. Validation box/class/DFL losses were finite at `4.42867`, `6781.25`, and `308.764`. The high class and DFL losses and near-zero early mAP warrant attention when the official training curves are reviewed. Smoke measurements establish execution and numerical finiteness, not detector quality. Compact gate evidence is in [`docs/evidence/phase3e/smoke_gate.json`](evidence/phase3e/smoke_gate.json); the full ignored manifest and outputs are under `outputs/training/smoke/E_seed42/`.

## Interpretation and next command

Changing six output classes to one changes the prediction task and mAP calculation. E's eventual mAP is not strictly like-for-like with A/B/C/D six-class mAP; any comparison must identify the altered taxonomy and should inspect localization errors, not just a headline score. The source label semantics are supported but not clinically adjudicated, empty-label uncertainty remains, and patient/study independence is unresolved.

After review, the official E command is:

```powershell
.venv\Scripts\python.exe -m bone_fracture_pipeline.experiment_runner --experiment E
```

It is configured for 100 epochs and will require a fresh `outputs/training/official/E_seed42` directory. This report stops at the smoke gate; the official command has not been run.
