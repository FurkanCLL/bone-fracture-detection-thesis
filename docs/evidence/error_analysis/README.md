# Error analysis: Stage 1 infrastructure

Implemented on 2026-10-10. The export, routing, numerical checks, deterministic matching, and tests pass. **Official metric reproduction fails** at the preselected absolute AP tolerance of `0.00005`. This is not an approved equivalent reproduction of the epoch-57 evaluation. Stage 2 has not started.

The machine-readable record is [stage1_validation.json](stage1_validation.json). Detailed logs, validation-only dataset copies, framework caches, run summaries, and prediction JSONL files remain under ignored `outputs/error_analysis/stage1/`.

## Implementation and reproduction

`src/bone_fracture_pipeline/error_analysis.py` resolves the frozen D configuration, checks its official manifest and checkpoint metadata, loads validation only, checks CLAHE build hashes, copies validation into a fresh output directory, and captures native validation predictions without changing AP inputs. `detection_evaluation.py` provides independent diagnostic matching and basic metrics. No dependency changes were needed.

| File | Change |
| --- | --- |
| `src/bone_fracture_pipeline/error_analysis.py` | New validation/export entry point and provenance guards |
| `src/bone_fracture_pipeline/detection_evaluation.py` | New independent IoU, matching, and metric utilities |
| `tests/test_error_analysis.py` | New focused synthetic test suite |
| `docs/evidence/error_analysis/README.md` | New Stage 1 method and verification record |
| `docs/evidence/error_analysis/stage1_validation.json` | New machine-readable compact evidence |
| `README.md` | Current Stage 1 status and evidence link |
| `docs/ROADMAP.md`, `docs/DECISIONS.md` | Local phase status and methodological decision; already ignored by Git |

From the repository root, in the existing environment:

```powershell
.venv/Scripts/python.exe -B -m bone_fracture_pipeline.error_analysis --output outputs/error_analysis/stage1/D_seed42_new
```

Use a fresh output directory. Existing evidence is never overwritten. A metric-reproduction failure preserves the summary/export and exits with code 1. The default uses native standalone layer fusion. `--unfused` is a recorded diagnostic that retains batch normalization, as in epoch validation; it does not fix the discrepancy. Installed framework files are not patched.

The primary evidence run is `D_seed42_repeat_verified`. Its export and AP values exactly match the first `D_seed42` run. `D_seed42_unfused_repeat` exactly matches the earlier unfused diagnostic. The failed metadata-only `D_seed42_repeat` attempt is retained with its log; it stopped before snapshot creation or inference. The checkpoint's older model definition has explicit depth/width multipliers, rather than a YAML filename or scale key; the implementation now records the actual fields.

## Provenance and routing

- Checkpoint: `outputs/training/official/D_seed42/weights/best.pt`, SHA-256 `498e265f90bbf07fc5112b7153a7a11ac09ac819f8b7689e9e832ac302ef2e8a`.
- Model: YOLOv8s, six original classes, depth multiplier `0.33`, width multiplier `0.5`. Saved training metrics corroborate the best CSV row, epoch 57. Checkpoint stripping sets the stored epoch to `-1`.
- Dataset: `data/prepared/v3_detection_clahe`. The historical full fingerprint is retained as provenance, not recalculated because that would read held-out files.
- Current restricted fingerprint: SHA-256 `d43b38126e40e060b79b130c6475db2243f797f191d95038eee1e9d2491d7a78`, covering `data.yaml` and 348 validation image/label pairs. It is unchanged before/after evaluation. Every validation pair matches the original CLAHE preparation manifest.
- Native caches are written into validation copies under `outputs/`. The snapshot YAML has no test key. Its required train key aliases the validation copies; training is never invoked. Loader paths and image identities are checked explicitly.
- Raw data is not opened or changed. No real test image or label is loaded. Tests use synthetic fixtures, including deliberately invalid synthetic held-out files.

## Official comparison

Reference values come from epoch 57 of official `results.csv`, selected by maximum validation mAP50-95, not final-epoch manifest metrics. The bound was fixed before evaluation: `5e-5` allows five-decimal CSV rounding (at most `5e-6`) and a small additional numerical margin. It is not a statistical equivalence claim and was not widened after failure.

| Native standalone metric | Official epoch 57 | Observed | Signed difference | Pass |
| --- | ---: | ---: | ---: | --- |
| mAP50 | 0.14861 | 0.1485241076082241 | -0.00008589239177589358 | No |
| mAP50-95 | 0.04732 | 0.04736909409864185 | +0.00004909409864185116 | Yes |

The unfused diagnostic gives mAP50 `0.14852369940929525` (difference `-0.0000863005907047465`) and mAP50-95 `0.047369839028952886` (difference `+0.00004983902895288467`). It also fails the combined check. Actual backend checks confirm FP32 parameters, zero BatchNorm layers for the fused run, and retained BatchNorm layers for the unfused run.

Checkpoint identity, validation content, original target geometry, native validation confidence/NMS defaults, and trainer batching/preprocessing were checked. The pinned trainer validates the in-memory FP32 EMA, then serializes checkpoint weights in FP16. The saved artifact indeed contains FP16 parameters. Loading it as FP32 cannot recover discarded bits. This is a plausible residual cause, **not a proven complete explanation**; the original epoch-57 FP32 EMA is unavailable in this checkpoint. Layer fusion alone does not explain the mismatch. Official results and checkpoints remain unchanged.

## Prediction export contract

The primary `validation_predictions.jsonl` contains one record per validation image: stable `valid/<stem>` ID, original dimensions, source image/label hashes, predictions, and ground truth. Each box has a stable per-image ID, original class ID, and original-image pixel `xyxy`; each prediction also has its confidence. Empty lists are retained. Full floating-point JSON values are preserved and read back for validation.

Verified counts: **348 images, 204 ground truth boxes, 175 empty-label images, 1,755 predictions, 75 images with no predictions**. No image reaches the maximum-detection limit. All values are finite and boxes lie inside their original image bounds. Maximum framework target roundtrip error is `0.00008365943756416527` pixels.

Recorded inference settings: image size 640; validation batch 16 (the trainer doubles training batch 8); rectangular validation batches; FP32; confidence floor 0.001; NMS IoU 0.7; maximum 300 detections; class-aware, multi-label NMS; six original classes; prediction augmentation off; workers 0; seed 42 and deterministic algorithms. Runtime: Ultralytics 8.4.155, PyTorch 2.6.0+cu126, RTX 4050 Laptop GPU.

The export is captured from the same native AP pass, after NMS, then converted to original-image coordinates using cloned tensors. This establishes common prediction provenance, not equivalence between native AP and the diagnostic matcher. The unfused diagnostic contains 1,757 predictions and is kept separate from the primary export.

## Diagnostic matching specification

`match_detections` filters at confidence `>= threshold`, sorts by descending confidence then box ID, and matches each retained prediction to the highest-IoU available target at IoU `>= threshold`. Target ID breaks equal-IoU ties. Class-aware matching requires equal class IDs; class-agnostic matching ignores that condition without changing IDs. Each prediction and target can participate in at most one match.

Unmatched retained predictions are FP; unmatched targets are FN. A wrong-class prediction can therefore produce both FP and FN. Duplicate predictions cannot reuse a claimed target, but may match another eligible unmatched target. Below-threshold candidates are excluded, not counted as FP. With empty labels every retained prediction is FP; with no predictions every target is FN. Precision, recall, and F1 use zero when their denominator is zero.

Native Ultralytics AP uses its own IoU-based assignment and confidence-integrated metric calculation. Synthetic checks establish agreement on unambiguous matches and demonstrate a conflict where assignments differ. Diagnostic fixed-threshold counts must not be presented as native AP. No real-data sensitivity sweep, error taxonomy, qualitative review, or figure generation is implemented here.

## Verification and limits

The focused suite passes 25 tests; the complete repository suite passes 113. Coverage includes IoU edge cases, classes, one-to-one assignment, duplicates, threshold boundaries, empties, stable tie-breaking, finite metrics, native matching semantics, export roundtrips, validation-only routing, held-out path rejection, provenance drift, and snapshot preservation. Commands and log hashes are recorded in the JSON evidence.

The three pre-existing Phase 2F evidence changes and frozen configurations are preserved. All 75 complete hashes retained from the initial protection capture still match; that initial tool output was truncated, so it is not a complete 151-file before/after hash proof. Current modification times for all 151 protected configuration/evidence/official-output files predate the first evaluation. D's checkpoint is additionally hashed before/after every evaluation. No A-E artifact is written by this extension.

Before Stage 2, review the failed reproduction gate and explicitly decide how to handle the small discrepancy; do not silently claim official equivalence. Candidates below 0.001 or suppressed by NMS are unavailable. Class-agnostic matching still uses candidates produced by class-aware NMS. Repeatability is established on this runtime only. Empty labels do not establish clinical fracture absence. No test evaluation, new training, Experiment F, commit, or push was performed.
