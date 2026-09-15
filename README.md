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
