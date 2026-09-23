# Deep Learning-Based Object Detection Pipeline for Bone Fracture Identification and Estimation in X-Ray Images

This repository supports a Riga Technical University bachelor thesis on localizing source-annotated suspected fracture regions in X-ray images. It brings together dataset checks, traceable image preparation, controlled YOLOv8 experiments, and validation analysis. “Estimation” refers to estimating a region's location, not fracture severity or prognosis.

## Research purpose and data

The study asks how image preprocessing and conservative training augmentation affect fracture-region detection under a shared protocol. Its starting point is version 3 of the *Bone Fracture Detection: Computer Vision Project* dataset. The local v3 export has 1,728 images across fixed train (1,211), validation (348), and held-out test (169) splits, with six original source classes. The source labels are polygons; the detection pipeline converts each to a minimum enclosing axis-aligned box. An empty label means that no region was annotated in that file, not that the image was clinically verified as fracture-free.

Raw data under `data/raw/` remains unchanged. Derived datasets under `data/prepared/` retain the source split membership and traceable conversions. The held-out test split has not been used for model selection or the official validation results below.

## Pipeline and experimental design

The workflow audits the source exports, prepares and independently validates the v3 detection boxes, builds two controlled image conditions, and fine-tunes a COCO-pretrained YOLOv8s detector. Training uses 640-pixel input, a 100-epoch budget, seed 42, and a shared configuration. The best checkpoint is selected by validation mAP50–95.

The PNG control preserves decoded image pixels. The CLAHE condition converts images to grayscale, applies OpenCV CLAHE (`clipLimit=2.0`, `tileGridSize=8×8`), and saves three identical channels as PNG. The conservative policy for C, D, and E applies only during training: limited rotation, translation, scale, and intensity variation. It excludes flips and composite-image augmentation.

| Experiment | Image condition | Conservative augmentation | Target classes |
| --- | --- | --- | --- |
| A | PNG control | No | Six original classes |
| B | CLAHE | No | Six original classes |
| C | PNG control | Yes | Six original classes |
| D | CLAHE | Yes | Six original classes |
| E | CLAHE | Yes | One merged `fracture` class |

A–D form the controlled 2×2 comparison. E is a separate follow-up that asks whether merging the six source classes changes class-agnostic localization; it changes the prediction task and its mAP is not strictly comparable with the six-class runs.

## Current results

All five official seed-42 runs completed 100 epochs. These are **validation** values from the best epoch selected by mAP50–95, as recorded in each local `outputs/training/official/<experiment>_seed42/results.csv`.

| Experiment | Best epoch | Validation mAP50 | Validation mAP50–95 |
| --- | ---: | ---: | ---: |
| A | 19 | 0.06633 | 0.02414 |
| B | 18 | 0.06414 | 0.02566 |
| C | 40 | 0.12065 | 0.04163 |
| D | 57 | 0.14861 | 0.04732 |
| E | 27 | 0.14895 | 0.05115 |

Conservative augmentation improved validation performance in both image conditions. CLAHE alone brought only a small improvement over A. D was the strongest six-class condition. E's one-class result is only modestly higher than D's aggregate value, with the task-definition caveat above. Baseline training and validation curves show substantial overfitting; none of these scores establishes reliable clinical performance. The runs use one seed each, and no final held-out model evaluation is reported here.

A separate [dataset difficulty characterization](docs/DATASET_DIFFICULTY_CHARACTERIZATION.md) measures train and validation composition, image dimensions, and box scale. It found 698 training boxes across 604 positive images, roughly half of images with empty labels in each split, and a median effective box shorter side near 64 pixels at the 640-pixel training size. The typical enclosing box is therefore not extremely small, and the measured train/validation geometry is broadly similar. These properties may contribute to a difficult learning regime but do not identify a single cause of low validation performance.

## Repository guide and reproducibility

| Path | Purpose |
| --- | --- |
| `configs/training/`, `configs/experiments/` | Shared protocol, augmentation policy, and experiment definitions |
| `src/bone_fracture_audit/` | Dataset audit and provenance checks |
| `src/bone_fracture_pipeline/` | Preparation, validation, experiment runner, and analysis code |
| `tests/` | Focused automated checks |
| `docs/` | Method reports, dataset findings, and compact evidence |
| `data/`, `outputs/` | Local datasets, detailed generated evidence, figures, run logs, and checkpoints; excluded from Git |

The project declares dependencies in `requirements.txt` and `pyproject.toml`. The protocol in [`configs/training/baseline.yaml`](configs/training/baseline.yaml) fixes model, optimizer, image size, seed, split use, checkpoint rule, and numerical settings; each experiment configuration selects its image and augmentation condition. Local official run manifests preserve the resolved settings, environment, data-routing checks, and output verification. The full test suite can be run with `python -m unittest discover -s tests -q` in the configured environment.

For methods and evidence, start with the [v3–v4 comparison](docs/DATASET_V3_V4_COMPARISON_REPORT.md), [conversion validation](docs/PHASE_2B_VALIDATION_REPORT.md), [training protocol](docs/PHASE_2C_BASELINE_PROTOCOL.md), [preprocessing report](docs/PHASE_2D_PREPROCESSING_REPORT.md), [augmentation policy](docs/PHASE_2E_AUGMENTATION_POLICY.md), and [dataset difficulty report](docs/DATASET_DIFFICULTY_CHARACTERIZATION.md). Phase reports describe the state at the time they were written; current run status is in the local official manifests, while best-epoch metrics must be read from `results.csv`.

## Status and limitations

Dataset preparation, protocol checks, A–E official training, and the train/validation difficulty analysis are complete. Further error analysis and the reserved final test evaluation remain separate work. Source-class semantics, empty-label meaning, annotation quality, and patient or study independence are not fully established. This detector is an experimental research prototype, **not a clinically validated diagnostic system**.
