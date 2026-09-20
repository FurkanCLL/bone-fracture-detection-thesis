# Bone Fracture Detection Thesis

Implementation workspace for an RTU bachelor thesis on object detection for suspected bone-fracture regions in X-ray images.

## Dataset audit

The Phase 1 audit discovers YOLOv8 exports below `data/raw/`, validates their labels, calculates image and bounding-box statistics, checks exact and cautious perceptual duplicates, and creates review images without changing the source dataset.

```powershell
python -m bone_fracture_audit.cli --raw-root data/raw --output outputs/data_quality/dataset_audit --report docs/DATASET_AUDIT_REPORT.md --thesis-context docs/THESIS.md
```

Create or activate the project virtual environment, install the dependencies, and install the project in editable mode:

```powershell
# Reproduce the Phase 2C NVIDIA/CUDA environment before installing the remaining dependencies.
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install -e .
```

The first command selects the CUDA 12.6 PyTorch wheels used by the recorded training environment. A non-training CPU environment may use the platform-appropriate PyTorch build instead. The editable install makes the `src/` packages and command-line entry points available without setting `PYTHONPATH` manually. In PyCharm, select this environment as the project interpreter. Marking `src` as a Sources Root is optional if the IDE still needs an additional navigation hint; do not commit `.idea` metadata. Generated audit artifacts belong under `outputs/` and are intentionally excluded from Git.

## Dataset follow-up tools

Prepare the fixed 16-train/4-validation empty-label sample for radiologist review:

```powershell
python -m bone_fracture_audit.review
```

Analyze source-name and Phase 1 similarity candidates using feature matching and robust affine alignment:

```powershell
python -m bone_fracture_audit.provenance
```

Compare the independently audited v3 and v4 exports, including held-out files and v3-to-v4 training derivatives:

```powershell
python -m bone_fracture_audit.comparison
```

These commands accept explicit read-only dataset/audit roots and write generated evidence below `outputs/`. See `docs/DATASET_FOLLOWUP_REPORT.md` and `docs/DATASET_V3_V4_COMPARISON_REPORT.md` for the source-backed findings and remaining questions before Phase 2.

## Phase 2A dataset preparation

Version 3 is the canonical Phase 2 source dataset. Prepare its polygon labels as standard YOLO detection boxes with:

```powershell
python -m bone_fracture_pipeline.prepare_dataset
```

The command copies every image byte-for-byte, preserves the original train/validation/test membership and six-class mapping, converts each polygon to its minimum enclosing axis-aligned box, and keeps empty labels as zero-byte files. It builds atomically and refuses to replace an existing prepared dataset unless `--overwrite` is supplied. Use `--source`, `--output`, and `--artifacts` to override the defaults.

The prepared dataset is written to `data/prepared/v3_detection`. Traceability outputs are written to `outputs/phase2/phase2a/v3_detection`, including per-file hashes, one-to-one annotation conversions, the source fingerprint, and the preparation summary. Both locations are intentionally ignored by Git. The raw v3 export is read-only and remains unchanged.

Phase 2A provides the deterministic format conversion. Phase 2B independently validates its geometry, visual overlays, and reproducibility as described below.

## Phase 2B conversion validation

Run the independent conversion and reproducibility checks with:

```powershell
python -m bone_fracture_pipeline.validate_conversion
```

The command re-checks every polygon and prepared box, calculates occupancy statistics, generates ranked review candidates and train/validation-only overlays, and rebuilds Phase 2A in temporary locations for hash comparison. It writes generated evidence to `outputs/phase2/phase2b/v3_detection` and does not modify either dataset. Existing Phase 2B output is protected unless `--overwrite` is supplied.

See `docs/PHASE_2B_VALIDATION_REPORT.md` for the validated results and remaining limitations. Phase 2B technically passed; preprocessing, augmentation, and training remain out of scope until their later planned phases.

## Phase 2C baseline protocol

Validate the frozen training protocol, prepared-dataset fingerprint, installed framework semantics, pretrained YOLOv8s weights, and local environment with:

```powershell
python -m bone_fracture_pipeline.training_protocol
```

This command performs no training. It writes full local evidence to `outputs/phase2/phase2c` and compact tracked evidence to `docs/evidence/phase2c`. The baseline itself is defined in `configs/training/baseline.yaml`; later controlled experiments must derive their common training values from it. See `docs/PHASE_2C_BASELINE_PROTOCOL.md` for the frozen settings, checkpoint rule, test-isolation policy, and remaining Phase 2F checks.

## Phase 2D custom preprocessing

Build and validate the fixed CLAHE image condition with:

```powershell
python -m bone_fracture_pipeline.preprocess_dataset
```

The command reads `data/prepared/v3_detection` without modifying it and creates `data/prepared/v3_detection_clahe`. It converts stored pixels to 8-bit grayscale, applies OpenCV CLAHE with `clipLimit = 2.0` and `tileGridSize = (8, 8)`, replicates the result to three identical channels, and saves lossless PNG files. Splits, dimensions, class mapping, and label bytes are preserved.

The builder validates every output, checks the source fingerprint before and after processing, produces train/validation-only visual QA, and confirms determinism with an independent temporary rebuild. Generated artifacts remain under `outputs/phase2/phase2d`; compact evidence is tracked under `docs/evidence/phase2d`. See `docs/PHASE_2D_PREPROCESSING_REPORT.md` for the validated results and limitations. The command performs no training.
