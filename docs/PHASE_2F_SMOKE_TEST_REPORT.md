# Phase 2F Experiment Freeze and Smoke-Test Report

## Status

Phase 2F is complete. The non-finite validation-loss failure was traced to FP16 overflow in the trained Experiment A detection head, the shared protocol was amended to disable AMP, and clean A/B/C/D one-epoch smoke runs passed with batch size 8. No official 100-epoch training was performed and the test split was not used.

## Frozen experiment matrix

| Experiment | Image condition | Controlled training augmentation |
|---|---|---|
| A | Pixel-identical PNG control | Off |
| B | CLAHE PNG | Off |
| C | Pixel-identical PNG control | Conservative policy |
| D | CLAHE PNG | Conservative policy |

All experiments inherit `configs/training/baseline.yaml`. The resolver proves that A/B and C/D differ only in dataset path/fingerprint fields, while A/C and B/D differ only in `degrees`, `translate`, `scale`, and `hsv_v`. Smoke mode changes only epochs from 100 to 1 and routes output to `outputs/training/smoke/`.

## Validation-loss diagnosis

The original COCO-pretrained `yolov8s.pt` completed all 44 validation batches with finite AMP predictions and normalized losses. Both `last.pt` and `best.pt` from the failed A run first became non-finite in validation batch 0 and had non-finite predictions and losses in 40 of 44 batches. Model parameters, input images, and targets were finite.

The first non-finite operation was `Detect.cv3.2.1.bn`. Its FP16 input from the preceding convolution was finite (range -15,800 to 7,932), but the batch-normalization output contained 84 negative infinities. The following activation and classification convolution propagated these into NaN class scores before the loss function ran. Those values then propagated through per-batch loss, accumulation, normalization, trainer metrics, and `results.csv`.

The same failed checkpoint completed all 44 validation batches with finite losses in CUDA FP32 and CPU FP32. CUDA and CPU results were close, while AMP alone reproduced the overflow:

| Checkpoint / mode | First non-finite batch | Non-finite batches | Normalized box / cls / DFL loss |
|---|---:|---:|---:|
| Original YOLOv8s, CUDA AMP | none | 0 / 44 | 3.34482 / 10.40430 / 3.36587 |
| Failed A `last.pt`, CUDA AMP | 0 | 40 / 44 | NaN / NaN / NaN |
| Failed A `best.pt`, CUDA AMP | 0 | 40 / 44 | NaN / NaN / NaN |
| Failed A `best.pt`, CUDA FP32 | none | 0 / 44 | 3.62244 / 91555.50646 / 256.64661 |
| Failed A `best.pt`, CPU FP32 | none | 0 / 44 | 3.62253 / 91562.76605 / 256.76740 |

Batch 0 contained seven finite targets and two empty-label images. Replaying the same images with every target removed still produced non-finite predictions, so empty labels are not the trigger. Standalone `model.val()` completed for the original, failed `last.pt`, and failed `best.pt` checkpoints with finite detection metrics, but that API path does not compute trainer-integrated validation losses and therefore did not contradict the diagnosis.

No installed Ultralytics file was patched. The exact installed validation, trainer, detection-validator, loss, and metrics source hashes are recorded in `docs/evidence/phase2f/installed_validation_source_hashes.json`.

## Protocol amendment

AMP is disabled in the shared baseline and enforced by protocol validation. This is a single global numerical-stability change applied equally to A/B/C/D; no image condition, augmentation policy, split, seed, optimizer, learning rate, batch size, or checkpoint-selection rule changed. The failed AMP run and its original `results.csv` remain archived at `outputs/training/smoke/_failed_attempts/A_seed42_amp_validation_overflow/`.

## Clean smoke execution

| Exp. | Runtime (s) | Peak allocated / reserved GiB | Train box / cls / DFL | Val box / cls / DFL | mAP50 / mAP50-95 |
|---|---:|---:|---:|---:|---:|
| A | 121.911 | 3.088 / 3.686 | 3.03400 / 10.52090 / 2.74980 | 3.20553 / 5.29495 / 2.82431 | 0.00005 / 0.00001 |
| B | 110.098 | 3.088 / 3.686 | 3.04676 / 13.97400 / 2.64497 | 3.09234 / 6.11174 / 2.90359 | 0.00013 / 0.00004 |
| C | 100.066 | 3.088 / 3.686 | 2.99962 / 11.27460 / 2.62248 | 3.30187 / 5.81715 / 2.84232 | 0.00016 / 0.00004 |
| D | 91.493 | 3.088 / 3.686 | 2.99383 / 13.61790 / 2.54058 | 2.98935 / 5.28479 / 2.52963 | 0.00573 / 0.00119 |

Every run completed one epoch, produced finite training and validation values, wrote `last.pt` and `best.pt`, generated expected plots and trainer batch images, and passed fixed-sample inference on three validation images with finite predictions. These smoke metrics are pipeline evidence, not thesis performance results.

## Dataset, augmentation, and visual QA

All manifests recorded 1,211 training images, 348 validation images, zero test images loaded, and restored dataset fingerprints after generated cache cleanup. A/B had every controlled transform disabled. C/D recorded the same conservative train-only transform chain and retained augmentation-free validation.

All three saved training batches from every experiment were manually inspected. A/B showed consistent unaugmented geometry; B showed the expected CLAHE contrast difference. C/D showed plausible conservative rotation/intensity variation with corresponding boxes. No invalid, displaced, or out-of-frame boxes were observed. This is technical pipeline QA, not medical-correctness review.

## Gate decision

The complete Phase 2F gate passes. Batch size 8 was stable across all four smoke runs, the numerical failure has a documented causal fix, configuration and pairing checks pass, fixed-sample validation inference passes, and test isolation is preserved. Phase 2 is technically complete and Phase 3 official controlled training may begin in a separate step. Official training has not started in this work.

## Evidence

Compact evidence is stored in `docs/evidence/phase2f/`, including the matrix/environment checks, smoke summary, manual batch review, diagnostic comparison, per-batch summary, and installed-source hashes. Full traces, checkpoints, and generated visualizations remain under ignored `outputs/` directories.
