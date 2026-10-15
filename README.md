# Deep Learning-Based Object Detection Pipeline for Bone Fracture Identification and Estimation in X-Ray Images

This repository supports a Riga Technical University bachelor thesis on localizing source-annotated suspected fracture regions in X-ray images. It brings together dataset checks, traceable image preparation, controlled YOLOv8 experiments, a Faster R-CNN comparison, and validation error analysis. “Estimation” refers to estimating a region's location, not fracture severity or prognosis.

## Research purpose and data

The study asks how image preprocessing and conservative training augmentation affect fracture-region detection under a shared YOLOv8 protocol, then examines an alternative detector under the same data condition as Experiment D. Its starting point is version 3 of the *Bone Fracture Detection: Computer Vision Project* dataset. The local v3 export has 1,728 images across fixed train (1,211), validation (348), and held-out test (169) splits, with six original source classes. The source labels are polygons; the detection pipeline converts each to a minimum enclosing axis-aligned box. An empty label means that no region was annotated in that file, not that the image was clinically verified as fracture-free.

Raw data under `data/raw/` remains unchanged. Derived datasets under `data/prepared/` retain the source split membership and traceable conversions. The held-out test split has not been used for model selection or the official validation results below.

## Pipeline and experimental design

The workflow audits the source exports, prepares and independently validates the v3 detection boxes, builds two controlled image conditions, and fine-tunes COCO-pretrained detectors. A–E use YOLOv8s; F uses Faster R-CNN MobileNetV3-Large FPN. All six runs use 640-pixel input, a 100-epoch budget, seed 42, FP32 and checkpoint selection by validation mAP50–95. A–E share the frozen YOLOv8 protocol; F has a separate protocol documenting its framework-specific optimization and evaluation settings.

The PNG control preserves decoded image pixels. The CLAHE condition converts images to grayscale, applies OpenCV CLAHE (`clipLimit=2.0`, `tileGridSize=8×8`), and saves three identical channels as PNG. The conservative policy for C, D, E, and F applies only during training: limited rotation, translation, scale, and intensity variation. It excludes flips and composite-image augmentation.

| Experiment | Detector | Image condition | Conservative augmentation | Target classes |
| --- | --- | --- | --- | --- |
| A | YOLOv8s | PNG control | No | Six original classes |
| B | YOLOv8s | CLAHE | No | Six original classes |
| C | YOLOv8s | PNG control | Yes | Six original classes |
| D | YOLOv8s | CLAHE | Yes | Six original classes |
| E | YOLOv8s | CLAHE | Yes | One merged `fracture` class |
| F | Faster R-CNN MobileNetV3-Large FPN | CLAHE | Yes | Six original classes |

A–D form the controlled 2×2 comparison. E is a separate follow-up that asks whether merging the six source classes changes class-agnostic localization; it changes the prediction task and its mAP is not strictly comparable with the six-class runs. F retains D's six-class CLAHE condition and conservative augmentation to investigate an alternative detector, with the necessary framework differences documented explicitly.

## Current results

All six official seed-42 runs completed 100 epochs. The following A–E values use native Ultralytics validation AP from the best epoch selected by mAP50–95, as recorded in each local `outputs/training/official/<experiment>_seed42/results.csv` and completed `run_manifest.json`. F's shared-evaluator results are reported separately below.

| Experiment | Best epoch | Validation mAP50 | Validation mAP50–95 |
| --- | ---: | ---: | ---: |
| A | 19 | 0.06633 | 0.02414 |
| B | 18 | 0.06414 | 0.02566 |
| C | 40 | 0.12065 | 0.04163 |
| D | 57 | 0.14861 | 0.04732 |
| E | 27 | 0.14895 | 0.05115 |

Conservative augmentation improved validation performance in both image conditions. CLAHE alone brought only a small improvement over A. D was the strongest six-class YOLOv8 condition. E's one-class result is only modestly higher than D's aggregate value, with the task-definition caveat above. Baseline training and validation curves show substantial overfitting; none of these scores establishes reliable clinical performance. The runs use one seed each, and no final held-out model evaluation is reported here.

A separate [dataset difficulty characterization](docs/DATASET_DIFFICULTY_CHARACTERIZATION.md) measures train and validation composition, image dimensions, and box scale. It found 698 training boxes across 604 positive images, roughly half of images with empty labels in each split, and a median effective box shorter side near 64 pixels at the 640-pixel training size. The typical enclosing box is therefore not extremely small, and the measured train/validation geometry is broadly similar. These properties may contribute to a difficult learning regime but do not identify a single cause of low validation performance.

## Experiment F and shared D–F evaluation

The official F run completed 100 epochs with finite training losses, gradients and parameters, all 1,211 training images and 348 validation images per epoch, unchanged dataset fingerprints and zero recorded held-out test IO attempts. Its best checkpoint is epoch **32**, selected by shared validation mAP50–95. The local `outputs/training/experiment_f/official_seed42/run_summary.json` records the complete run, which took approximately **2.46 hours**; `best.pt`, `last.pt`, per-epoch prediction exports and diagnostics remain under that ignored output directory.

For D–F comparison, use the same [shared AP evaluator](src/bone_fracture_pipeline/detection_ap_evaluation.py) on original-image coordinates, six classes, confidence >0.001, a maximum of 300 predictions per image, and IoU thresholds 0.50–0.95. This uses the pinned Ultralytics matching and AP integration, rather than claiming pycocotools COCOeval equivalence.

| Shared evaluation | Selected epoch | Validation mAP50 | Validation mAP50–95 |
| --- | ---: | ---: | ---: |
| D: preserved YOLOv8s best checkpoint | 57 | 0.14923 | 0.04744 |
| F: Faster R-CNN best checkpoint | 32 | 0.15333 | 0.05932 |

F has higher shared aggregate validation AP in this single-seed comparison. This does not establish statistical significance, consistent gains for every class, or clinical generalization. Framework-specific losses, proposal sampling, normalization, EMA, warmup, padding and physical batch size differ, so the runs do not isolate architecture as the only change. The original A–E native AP values above remain unchanged; they must not be mixed with the shared values when assessing D–F differences.

At confidence 0.25 and IoU 0.50, F's selected checkpoint gives **52 TP / 219 FP / 152 FN**, precision 19.2% and recall 25.5% under the existing class-aware diagnostic matcher. Ignoring class equality gives **62 TP / 209 FP / 142 FN**, recall 30.4%. These confidence-first fixed-threshold diagnostics remain separate from AP; increased recall relative to D also comes with substantially more false positives.

The [F protocol](docs/EXPERIMENT_F_PROTOCOL.md) records COCO initialization, six foreground classes plus background, physical batch **4**, accumulation **16**, effective 64-image optimization windows, and RPN score thresholds **0.0 during training / 0.05 during inference**. Historical [Stage 1 failure](docs/evidence/experiment_f/STAGE1_REPORT.md) and [Stage 1B feasibility evidence](docs/evidence/experiment_f/STAGE1B_REPORT.md) preserve the numerical diagnosis, batch benchmarks and export repair. Those reports describe the preparation stage; the completed official run now provides F's research results.

## Repository guide and reproducibility

| Path | Purpose |
| --- | --- |
| `configs/training/`, `configs/experiments/` | Shared protocol, augmentation policy, and experiment definitions |
| `src/bone_fracture_audit/` | Dataset audit and provenance checks |
| `src/bone_fracture_pipeline/` | Preparation, validation, experiment runner, and analysis code |
| `tests/` | Focused automated checks |
| `docs/` | Method reports, dataset findings, and compact evidence |
| `data/`, `outputs/` | Local datasets, detailed generated evidence, figures, run logs, and checkpoints; excluded from Git |

The project declares dependencies in `requirements.txt` and `pyproject.toml`, including Torch 2.6.0, Torchvision 0.21.0 and Ultralytics 8.4.155. The protocol in [`configs/training/baseline.yaml`](configs/training/baseline.yaml) fixes the common A–E model, optimizer, image size, seed, split use, checkpoint rule and numerical settings; each experiment configuration selects its image and augmentation condition. [`configs/experiments/F.yaml`](configs/experiments/F.yaml) defines the separate Faster R-CNN protocol. A–E manifests and F's run summary preserve the resolved settings, environment, data-routing checks and output verification. The full test suite can be run with `python -m unittest discover -s tests -q` in the configured environment.

For methods and evidence, start with the [v3–v4 comparison](docs/DATASET_V3_V4_COMPARISON_REPORT.md), [conversion validation](docs/PHASE_2B_VALIDATION_REPORT.md), [training protocol](docs/PHASE_2C_BASELINE_PROTOCOL.md), [preprocessing report](docs/PHASE_2D_PREPROCESSING_REPORT.md), [augmentation policy](docs/PHASE_2E_AUGMENTATION_POLICY.md), and [dataset difficulty report](docs/DATASET_DIFFICULTY_CHARACTERIZATION.md). Phase reports describe the state at the time they were written; current run status is in the local official manifests, while best-epoch metrics must be read from `results.csv`.

## Status and limitations

Dataset preparation, protocol checks, A–F official training, and the train/validation difficulty analysis are complete. [Error analysis Stage 1](docs/evidence/error_analysis/README.md) adds a reproducible validation-only prediction export and tested diagnostic matching utilities. Its official metric reproduction check fails the fixed mAP50 tolerance, so equivalent reproduction is not claimed.

[Phase 4 Error Analysis](docs/evidence/error_analysis/FINAL_ERROR_ANALYSIS_REPORT.md) is complete, including [Stage 2 quantitative diagnostics](docs/evidence/error_analysis/STAGE2_REPORT.md) and direct Stage 3 review of 36 cases from 36 distinct validation images. At confidence 0.25 and IoU 0.50, class-aware training recall is 94.4% versus 14.7% on validation; ignoring class equality raises validation recall to 15.7%. These fixed-threshold diagnostics are separate from native AP. The qualitative review illustrates missed overlap, weak aligned candidates, confident geometry errors, and interacting class/score/placement behavior, without establishing a single cause. Medical-image overlays remain in ignored local outputs; [public case evidence](docs/evidence/error_analysis/stage3_qualitative.json) contains text, numerical identities, and hashes only.

The Stage 1 D reproduction failure remains unresolved at its original absolute AP tolerance of 0.00005; using shared AP for D–F comparison does not resolve or replace that check. F's official run is complete, while a dedicated post-training F error analysis and integrated thesis reporting remain follow-up work. Final held-out evaluation has not been performed.

Source-class semantics, empty-label meaning, annotation quality, and patient or study independence are not fully established. This detector is an experimental research prototype, **not a clinically validated diagnostic system**.
