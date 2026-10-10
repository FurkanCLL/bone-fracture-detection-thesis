# Experiment F: Faster R-CNN protocol

Stage 1 authorizes implementation, synthetic tests, a two-image training-only forward/backward preflight, and **one full training epoch plus complete validation**. It does not authorize official training. The measured feasibility outcome is recorded separately in `docs/evidence/experiment_f/STAGE1_REPORT.md`.

**Current gate: failed.** The first attempt stopped at zero-based training position 401 with a non-finite loss, after six optimizer updates. No complete epoch, F validation AP, best checkpoint or last checkpoint was produced. Post-failure observability and official-launch guards were added and tested without another GPU training run. The protocol is not ready to freeze or launch.

## Fixed inputs and scientific scope

- Canonical `data/prepared/v3_detection_clahe`: train 1,211 images / 698 boxes / 607 empty labels; valid 348 / 204 / 175.
- The existing six classes and polygon-derived boxes remain unchanged. Empty labels mean no supplied annotations, not verified health.
- Every train/valid image and label is checked against the original CLAHE manifest. Restricted fingerprints include `data.yaml` and only the selected split. No full-dataset fingerprint is recomputed because that would read test files.
- Explicit split routing rejects anything other than `train` and `valid` before IO. Real-path checks reject escaped paths. A runtime Python audit guard rejects test-tree opens and enumeration. The loader uses only validated, explicit image paths; it does not invoke a framework dataset scanner or write a canonical cache.
- No raw/prepared dataset edits, class merging, hyperparameter selection, architecture sweep or validation-driven tuning is permitted. The same validation identities and original-pixel GT are used for common D/F AP.

## Model and input contract

Use Torchvision **0.21.0** `fasterrcnn_mobilenet_v3_large_fpn`, the high-resolution variant, with `FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1`. Load the complete COCO detector before replacing only `FastRCNNPredictor` with seven outputs (28 regression values). Canonical class 0–5 maps to target 1–6; background 0 cannot be exported; predictions map back to 0–5.

`trainable_backbone_layers=3` uses Torchvision's stage boundaries, rather than interpreting "three" as three individual convolutions. Backbone blocks before block 7 are frozen; blocks 7–16, FPN, RPN and ROI components train. Pretrained normalization uses `FrozenBatchNorm2d`; its statistics do not update. Adapted counts: **18,955,854 total; 18,896,958 trainable; 58,896 frozen**. Runtime evidence records all frozen names and trainable block indices.

Weight source: `https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth`; full SHA-256 `fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533`. Torchvision verifies the filename hash prefix during download; evidence verifies the full digest. Weights remain in ignored `outputs/training/experiment_f/weights_cache`.

Input geometry is applied once in the dataset adapter using the installed Ultralytics `LetterBox`: longest side 640, aspect ratio retained, rounded resized dimensions, centered constant-114 padding to **640×640**, no stretching, scale-up allowed. Boxes use the same ratio and integer top/left padding. RGB FP32 pixels enter in [0,1]. The detector retains COCO normalization (mean .485/.456/.406, std .229/.224/.225) and divisibility-32 batching. A small `GeneralizedRCNNTransform` subclass rejects non-640-square input and returns geometry unchanged. It does not perform another interpolation. Arbitrary aspect ratios, odd dimensions, boxes, tensor shape and coordinate inversion are tested.

Torchvision normally uses minimum side 800 and maximum side 1333. Simply setting both to 640 would not reproduce D preprocessing. Here external geometry plus a strict internal identity operation establishes the tensor contract. Prediction boxes are inverse-mapped from padded coordinates, clipped to original dimensions, and exported with original GT/identities/hashes. Zero-area boxes produced solely by clipping predictions outside image content are discarded.

D training also targets 640-square geometry, but its initial loader resize uses ceiling-rounded dimensions before its affine transform. F uses one round-based letterbox. D validation uses rectangular, stride-padded batches of 16; F uses square batches of one. Thus rounding, padding context, interpolation history and model normalization differ. Neither exact pixel parity nor identical detector proposals is claimed.

Explicit model settings:

| Component | Setting |
|---|---|
| RPN pre/post NMS proposals | train 2,000 / 2,000; valid 1,000 / 1,000 |
| RPN NMS / objectness floor | .7 / .05 (high-resolution native settings) |
| RPN positive/background IoU | .7 / .3 |
| RPN sampler | 256 anchors/image, .5 positive fraction |
| Anchor sizes at three levels | 32,64,128,256,512; ratios .5,1,2 |
| ROI positive/background IoU | .5 / .5 |
| ROI sampler | 512/image, .25 positive fraction |
| ROI box coder | 10,10,5,5 |
| ROI NMS | .5, class aware (D uses .7) |
| ROI confidence / max detections | **>.001 / 300**, replacing native .05 / 100 |

The lower ROI floor and larger detection cap expose comparable retained candidates for the common evaluator. Architecture-specific proposal filtering and NMS remain documented differences.

## Train-only augmentation

Call the pinned Ultralytics `RandomPerspective` affine image/box methods and `RandomHSV` directly, with degrees 10, translation .05, scale .10, shear/perspective 0, HSV hue/saturation 0 and value .15. There are no other transforms. Validation executes only deterministic letterboxing.

Affine transformation encloses all four rotated box corners, clips to image bounds and uses D's native box-candidate conditions: width/height >2 px, retained scaled area ratio >.10 and aspect ratio <100. Removed targets and completely empty targets are supported. Intensity uses the actual pinned HSV value LUT, rather than an approximate brightness formula. Seven training-only previews cover all six classes plus an empty-label image; preview pixels remain ignored locally.

Per-sample augmentation seed is `42 + epoch*1211 + canonical_sorted_index`. Python/NumPy RNG state is restored afterward, separating augmentation from stochastic RPN/ROI sampling. Epoch order is a seeded permutation. This makes transformations reproducible but does not reproduce D's worker-8 random-number stream, disabled-transform RNG consumption or ordering. Strengths and meaning match the approved policy; exact stochastic parity is not claimed.

## Optimization and D source audit

D's preserved applied arguments and installed Ultralytics 8.4.155 trainer establish:

- Physical batch 8, nominal batch `nbs=64`, nominal accumulation **8 batches**. With 1,211 images there are 152 physical batches per epoch, including a final batch of 3.
- Official warmup is 3×152 = 456 iterations. Accumulation interpolates from 1 to 8 batches. Update condition is `ni-last_opt_step >= accumulate`; it is not simple unaccumulated batch-8 training. A source-based simulation is retained separately from measured training logs.
- Weight decay is scaled by `batch*accumulate/nbs`, giving .0005 at the configured nominal batch. Biases and normalization weights have no decay.
- AdamW beta1 is .937, beta2 .999; initial LR .001. Bias warmup starts at .1; other groups start at 0. The non-cosine scheduler linearly approaches `lrf=.01` across the 100-epoch budget.
- YOLO multiplies its returned loss by physical batch size and accumulates without averaging by the window. It clips gradient norm to 10. Gradients and step bookkeeping carry over epoch boundaries, with no explicit end-of-training partial flush. D selects the EMA checkpoint.

F uses physical train/valid batch 1, **constant accumulation 64**, AdamW LR target .001, beta1 .937, beta2 .999, epsilon 1e-8, weight decay .0005, bias/one-dimensional parameter decay exemption, and norm clip 10. Each image loss is divided by its actual window size. The epoch has **18×64 + 59** images in **19 optimizer updates**. The final window is averaged by 59 and stepped. No gradients are discarded. All losses, gradients and updated parameters must be finite before continuing.

F warms all parameter groups uniformly from zero toward .001 over the first three epochs, according to images seen at each update. It does not copy D's bias-only .1 transient. For zero-based epoch `e`, update LR is `.001 * min(images_seen/(3*1211),1) * ((1-e/100)*.99+.01)`. Thus the first full smoke epoch ends at LR .000333333, while retaining the **100-epoch** scheduler budget. No one-epoch-specific LR schedule is substituted. Warmup, averaged losses, fixed accumulation, Adam implementation (unfused), FrozenBN, lack of EMA and RPN/ROI sampling are explicit optimization differences. Numerical settings alone do not establish equal dynamics.

AMP and TF32 are disabled. Seed 42 initializes Python, NumPy, Torch and CUDA; deterministic algorithms run with warnings where a deterministic CUDA implementation is unavailable. No cross-hardware bitwise guarantee is made. Early stopping is disabled. Instability or OOM stops execution with failure evidence; there is no automatic reduction of resolution/batch/backbone, enabling AMP or LR tuning.

## Common AP and separate diagnostics

`detection_ap_evaluation.shared_ap` validates the Phase 4 schema and, for real comparisons, exact sample/GT/hash equality. Both models use canonical classes, strict confidence >.001, a cap of 300 detections/image, and IoU .50:.05:.95 in FP32. The shared evaluator calls the **installed Ultralytics 8.4.155** `BaseValidator.match_predictions` and `compute_ap`.

Native matching masks wrong classes, orders eligible pairs by descending IoU, removes duplicate predictions, then duplicate targets independently at each threshold. Image, GT and prediction IDs establish fixed input ordering; native NumPy `argsort` behavior settles IoU ties on the pinned runtime. The AP curve uses descending score with stable image/box-ID ties. Precision envelopes are interpolated at 101 recall points and integrated by trapezoids with the pinned sentinels. This is **not an assertion of pycocotools COCOeval equivalence**; e.g. a single perfect detection gives .995 here. Classes with no GT are reported as zero AP and excluded from the macro mean; all six real validation classes have support. FP on empty-label images affects AP. Completely empty inputs use zero, not NaN.

Deterministic fixtures test empty/multiclass, wrong class, duplicates, confidence floor, detection cap, ties and agreement with native `ap_per_class` on a known TP/FP sequence. Original official D AP, preserved checkpoint native reproduction and shared D AP are reported separately. The original .00005 reproduction tolerance and failed mAP50 status remain unchanged. Common AP can differ due to original-pixel GT geometry, tie ordering and reconstruction precision; no cause is asserted without a controlled replay. Future primary D/F comparison uses this same shared evaluator.

Phase 4 confidence-first diagnostic matching stays unchanged. F exports feed its class-aware/agnostic TP/FP/FN, P/R/F1, per-class and size utilities. AP matching must never be replaced with that diagnostic matcher. Smoke results establish execution feasibility; they do not establish architecture superiority or final validation performance.

## Execution, checkpoints and resume

Run from the repository root in the pinned project environment:

```powershell
# Exactly one training epoch and one complete validation; choose a fresh path.
.venv/Scripts/python.exe -B -m bone_fracture_pipeline.experiment_f_runner --mode smoke --output outputs/training/experiment_f/smoke_stage1
```

**Future command only; Stage 1 does not execute it:**

```powershell
.venv/Scripts/python.exe -B -m bone_fracture_pipeline.experiment_f_runner --mode official --confirm-official-100-epochs --smoke-evidence outputs/training/experiment_f/approved_successful_smoke/run_summary.json --output outputs/training/experiment_f/official_seed42
```

Official mode refuses to start without its explicit opt-in flag **and a successful complete smoke summary matching the current config, source hashes and dataset fingerprints**. The example path represents a later approved successful run; the current failed attempt is rejected. Output paths must be fresh and inside the ignored F tree. The runner validates the entire applied YAML against the implemented contract and records its SHA-256. It captures preservation hashes before execution and verifies them afterward. When present, the Stage 1 historical protection map is also checked. External pretrained download is the only additional network requirement.

Best selection maximizes shared validation mAP50-95; strict improvement preserves the earlier checkpoint on a tie. `last.pt` is always preserved. Atomic checkpoint writes contain FP32 model state, optimizer and image-scheduler state, epoch, best score, dataset fingerprints, model/category/input metadata, YAML/implementation hashes, seed and Python/NumPy/Torch/CUDA RNG state. Each completed checkpoint is reloaded and compared exactly to the in-memory model; optimizer/scheduler progress is checked. Prediction exports retain original coordinates and Phase 4 compatibility.

Resume is allowed only in official mode, into a fresh directory, from a trusted local F checkpoint under the ignored F tree. `--resume path/to/last.pt` additionally requires the official flag. Config, implementation, data, model metadata and mode must match; RNG, model, optimizer and schedule are restored. Smoke checkpoints cannot become official runs through resume. Resume uses whole completed epoch boundaries; a mid-epoch technical failure restarts that epoch from the last completed checkpoint. Wall-clock or cross-machine equivalence is not guaranteed.

After the failed attempt, non-finite-loss exceptions were extended to record per-component values, actual RPN proposal counts, image/target metadata, completed backward passes, prior optimizer updates, elapsed time and prior finite-loss summaries. The runner now saves the failing model, partial gradients and optimizer/scheduler state as a diagnostic-only `failure_capture.pt`, explicitly rejected for resume. This safeguard is covered by synthetic failure tests. It did **not** exist during the first attempt and cannot recover that discarded in-memory state retroactively. An unchanged-config diagnostic retry is needed to establish the exact failing operation before proposing a scientific correction.

Preflight checks one annotated and one empty-label training image with actual forward/backward/update operations, then restores model, optimizer and RNG before the full epoch. Per-update logs record effective window size, LR and gradient norm. The run summary records maximum/representative losses, training/validation/checkpoint timings, throughput, peak allocated/reserved CUDA memory, capacity, provenance and failure status. Exactly one complete smoke epoch is permitted; no AP-driven rerun.

## Primary references

- [Torchvision 0.21 model documentation](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html).
- [Torchvision 0.21 Faster R-CNN implementation](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/detection/faster_rcnn.py), transform and backbone utilities in the installed package.
- [Pinned Ultralytics metric implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.155/ultralytics/utils/metrics.py), installed trainer/validator/data sources and preserved D applied manifest.

These are implementation/protocol notes, not final student-authored thesis prose or clinical validation.
