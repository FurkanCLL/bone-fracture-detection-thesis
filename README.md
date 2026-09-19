# Bone Fracture Detection Thesis

Implementation workspace for an RTU bachelor thesis on object detection for suspected bone-fracture regions in X-ray images.

## Dataset audit

The Phase 1 audit discovers YOLOv8 exports below `data/raw/`, validates their labels, calculates image and bounding-box statistics, checks exact and cautious perceptual duplicates, and creates review images without changing the source dataset.

```powershell
python -m bone_fracture_audit.cli --raw-root data/raw --output outputs/dataset_audit --report docs/DATASET_AUDIT_REPORT.md --thesis-context docs/THESIS.md
```

Install the dependency from `requirements.txt` first. If the package has not been installed in editable mode, set `PYTHONPATH=src` for the command. Generated audit artifacts belong under `outputs/` and are intentionally excluded from Git.

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

The prepared dataset is written to `data/prepared/v3_detection`. Traceability outputs are written to `outputs/phase2a/v3_detection`, including per-file hashes, one-to-one annotation conversions, the source fingerprint, and the preparation summary. Both locations are intentionally ignored by Git. The raw v3 export is read-only and remains unchanged.

Phase 2A provides a technically validated format conversion. Phase 2B visual QA and independent conversion analysis are still required before official model training.
