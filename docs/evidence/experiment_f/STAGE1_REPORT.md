# Experiment F Stage 1: failed full-epoch feasibility gate

**Decision: do not freeze or launch official training.** Infrastructure is implemented and tested, but the only smoke-training attempt stopped on a non-finite loss at zero-based position **401** (the 402nd attempted image). It completed 401 finite-loss backward passes and six optimizer updates, rather than the required full epoch. No F validation inference, AP, best checkpoint or last checkpoint was produced. No scientific setting was changed and no second GPU training run occurred.

The smallest justified next step is an **unchanged-config instrumented diagnostic retry**, with the user's approval, to retain the actual failing component and proposal/model state. Current evidence does not justify lowering LR, changing accumulation, enabling AMP, removing empty-label images, reducing resolution or changing architecture/backbone. A potential empty-proposal mechanism is described below as unconfirmed.

## Implementation and verification

The separately scoped infrastructure commit is `169394d` (`Add controlled Faster R-CNN experiment pipeline`). It adds:

- `configs/experiments/F.yaml` with an explicit six-class comparison protocol.
- `faster_rcnn_dataset.py`: guarded split loading, class conversion, train-only augmentation, letterboxing and original-coordinate prediction export.
- `faster_rcnn_training.py`: COCO model/head construction, freezing audit, seed setup, averaged accumulation, LR schedule, finite checks and checkpoint primitives.
- `detection_ap_evaluation.py`: a common D/F AP definition using the installed pinned Ultralytics matcher/integrator.
- `experiment_f_runner.py`: separate smoke/official modes, provenance, preflight, full-epoch training/validation, diagnostic compatibility, checkpoint selection, fresh outputs and resume safeguards.
- `tests/test_experiment_f.py` and `docs/EXPERIMENT_F_PROTOCOL.md`.

Before execution, **31 focused tests and 191 total tests passed**. After failure-observability and launch-guard improvements, **35 focused tests and all 195 regression tests passed**; the original 160 tests remain unchanged. Synthetic tests cover class/background edges, empty targets, test rejection before IO, arbitrary-dimension coordinate roundtrips, normalization/shape, no double resize, augmentation validity/repeatability, predictor outputs, actual averaged gradients and partial windows, optimizer decay exemptions, loss/gradient failure checks, checkpoint serialization/selection, AP/reference fixtures, diagnostic compatibility, official opt-in and successful-smoke prerequisites.

These tests do not substitute for a completed real F epoch. Real GPU checkpoint/validation completion remains unverified. Epoch-boundary resume is implemented but has not been exercised through a completed official F run.

## Model, initialization and category mapping

| Property | Actual configuration |
|---|---|
| Model | `torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn`, high-resolution variant |
| Runtime | Torch 2.6.0+cu126; Torchvision 0.21.0+cu126; Ultralytics 8.4.155 |
| COCO weights | `FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1` |
| Total / trainable / frozen parameters | 18,955,854 / 18,896,958 / 58,896 |
| Backbone | `trainable_backbone_layers=3`; blocks 0–6 frozen, 7–16 trainable |
| Other components | FPN, RPN and ROI components trainable; pretrained FrozenBatchNorm2d |
| Predictor | 7 classification outputs, 28 regression outputs |
| Category conversion | Canonical 0–5 → targets 1–6; background 0 rejected from exports; predictions return to 0–5 |
| Precision | FP32; AMP and TF32 disabled |

The full COCO detector is loaded before replacing the category-specific predictor. Weight URL is `https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth`; verified full SHA-256 is `fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533`. The sandbox initially blocked the network download with WinError 10013; the authorized download then succeeded. This was an input-setup recovery before training, not an additional training attempt.

The model choice and foreground/background contract follow the [Torchvision 0.21 documentation](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html); exact defaults and loaded parameter states were inspected locally. Native and applied RPN/ROI/normalization/transform settings are explicit in the protocol and numerical evidence.

## Dataset, preprocessing and augmentation fidelity

| Split | Images | Boxes | Empty labels |
|---|---:|---:|---:|
| Train | 1,211 | 698 | 607 |
| Validation | 348 | 204 | 175 |

Canonical CLAHE inputs, fixed membership, six source classes and polygon-derived boxes are preserved. Train/validation identities are disjoint. Every selected image/label hash matches the original CLAHE manifest. Before/after restricted fingerprints agree:

- Train: `d658084e25bb3d7f19634f5328bdd6ae465dcd51f1c770df42f2589c87d2a05e`.
- Validation: `d43b38126e40e060b79b130c6475db2243f797f191d95038eee1e9d2491d7a78`.

No full-dataset fingerprint was recomputed and no held-out test image, label or directory was opened/enumerated. Actual runtime/probe/final-verification test-IO blocked-attempt counters are zero. Separate synthetic tests deliberately request nonexistent test paths and verify rejection before IO. No raw data was used or modified. Canonical caches are not written.

F applies an aspect-preserving centered 640-square letterbox once, constant-114 padding, then RGB conversion and COCO normalization. Its internal transform rejects other dimensions and does no geometric resize. Automated checks verify arbitrary aspect ratios, odd sizes, original-pixel inversion, unchanged transformed boxes and actual 1×3×640×640 batching.

Conservative training-only augmentation directly uses D's pinned affine and HSV primitives: rotation ±10°, translation ±5%, scale .90–1.10, `hsv_v=.15`, with all other policies absent. Four-corner box enclosure, clipping and D's candidate filtering remain active. Seven directly inspected before/after training previews cover each class and an empty-label image; boxes follow the transformed regions. Images remain ignored locally; public evidence contains identities, coordinates, hashes and review notes only. Validation has no stochastic augmentation.

Exact pixel-level parity is not claimed: D's initial resize uses ceiling rounding and its training pipeline/worker RNG sequence differs. D validation uses rectangular stride-padded batches of 16, whereas F uses square batches of one. COCO channel normalization, proposal filtering, ROI NMS .5 versus D NMS .7, trainable normalization behavior, model sampling and loss functions differ. The protocol records these differences rather than treating matching numbers as equal training dynamics.

## Optimizer and accumulation audit

The installed D trainer and preserved applied settings confirm nominal batch size 64, physical batch 8, nominal accumulation 8 batches, scaled weight decay .0005, beta1 .937 and beta2 .999. Official warmup spans 456 physical iterations: accumulation interpolates from 1 to 8, bias LR starts at .1, other group LRs start at 0, and the 100-epoch linear scheduler uses final factor .01. The actual step condition depends on the previous step; there is no explicit last-window flush. YOLO multiplies loss by physical batch size, accumulates without window averaging and clips gradient norm to 10. D validates EMA weights.

A source-based simulation yields **83, 35, 22 and 19** optimizer updates in epochs 1–4 under these conditions. These are reconstructed counts, not newly measured D training logs. D is not a simple unaccumulated batch-8 baseline.

F uses physical batch 1 and constant accumulation 64, with every image loss divided by its actual window size. A successful epoch would have **18 windows of 64 plus one of 59**, giving **19 updates**. AdamW target LR is .001, weight decay .0005 with bias/one-dimensional-parameter exemptions, beta1 .937, beta2 .999 and norm clipping 10. Unfused AdamW and direct FP32 model evaluation are explicit differences from D.

Uniform image-based three-epoch warmup reaches the approved LR target, followed by linear 100-epoch decay to factor .01. At zero-based epoch `e`, LR is `.001 * min(images_seen/(3*1211),1) * ((1-e/100)*.99+.01)`. The smoke uses this full-budget schedule. It does not inherit a shortened one-epoch schedule or D's bias-only .1 transient.

Six completed 64-image updates reached LR **0.0001056977704**. Pre-clip gradient norms were 17.2943, 15.7161, 13.5265, 11.8393, 8.0388 and 1.9666. Losses, gradients and parameters passed finite checks at those updates. Seventeen further finite-loss backward passes accumulated before the abort; they were not stepped and their gradients were lost with the failed process. This is a failed partial epoch, not successful final-window handling on real data. Final-window averaging is verified synthetically.

## Shared AP validation and D references

| D evaluation source | mAP50 | mAP50–95 |
|---|---:|---:|
| Original official best epoch 57 | 0.1486100000 | 0.0473200000 |
| Preserved checkpoint native reproduction | 0.1485241076 | 0.0473690941 |
| New common evaluator, original-pixel export/GT | 0.1492271831 | 0.0474394016 |

The common evaluator verifies all 348 validation identities/GT/hashes, six classes and the retained D prediction export. Class supports remain 29,48,43,36,20,28. It uses class-aware IoU .50:.05:.95, strict confidence >.001 and maximum 300 detections/image; empty-label images and missing predictions are included. It calls pinned native IoU-first matching and precision-envelope AP integration, with fixed ID ordering and explicit tie behavior. AP uses 101-point interpolation and trapezoidal integration, **not a claim of pycocotools COCOeval equivalence**. Synthetic perfect/empty/multiclass/wrong-class/duplicate/cap/tie fixtures and a native `ap_per_class` reference check pass.

Shared/native D differences are retained without forced matching. Original-coordinate GT, reconstruction precision and tie behavior can differ; their separate contribution has not been established by controlled replay. The old native reproduction remains **failed at absolute tolerance .00005**: AP50 difference .0000858924 fails; AP50–95 difference .0000490941 passes. Original evidence and tolerance are unchanged.

Future primary D/F comparisons must use the common evaluator for both. Phase 4 confidence-first diagnostic matching remains separate and unchanged. Export compatibility with its class-aware/agnostic counts and size utilities is verified by tests; full real F validation diagnostic execution was not reached. No D-F superiority or final generalization conclusion can follow from this failed smoke gate.

## Smoke outcome and hardware measurements

| Measurement | Result |
|---|---|
| Preflight | Passed: one annotated and one empty-label image; forward/backward/update; full model/optimizer/RNG reset afterward |
| Full smoke attempts | Exactly one |
| Full epochs completed | **0** |
| Successful finite-loss backwards | 401 of 1,211 intended images |
| Failing image | Zero-based position 401, 402nd attempted forward |
| Completed optimizer updates | 6; 384 images represented |
| F validation images evaluated | 0 of 348 |
| F validation mAP50 / mAP50–95 | **Unavailable**, not zero |
| Trained best / last checkpoint | **Neither created** |
| Checkpoint primitives | Synthetic roundtrip passed; real completed-epoch checkpoint gate unverified |
| Actual GPU capacity | 6,438,780,928 bytes (6,141 MiB), RTX 4050 Laptop GPU |
| Smoke peak allocated / reserved, after preflight reset | 595,149,312 / 620,756,992 bytes (**567.58 / 592 MiB**) |
| Preflight peak allocated / reserved | 480,068,096 / 641,728,512 bytes (457.83 / 612 MiB) |
| Startup including preflight | 43.8344 s |
| Total failed-attempt elapsed time | 72.3135 s |
| Approximate partial training interval | 28.4791 s, inferred by subtraction |
| Complete training/validation duration and throughput | Unavailable |
| AMP / TF32 / automatic setting changes | Disabled / disabled / none |

The six logged window-end total losses were **2.199704, 2.019659, 1.626854, .896208, .958604 and .320340**. These are individual window-end observations, not window means or the global maximum/minimum of all finite samples. Full-prefix/epoch mean, maximum and the non-finite component/value were not persisted by the original implementation; they cannot be reconstructed honestly.

There was **no OOM**. Every preflight input was 3×640×640 and the strict transform rejects a second resize. The measurements show memory headroom for this prefix, not full-epoch numerical feasibility. CUDA allocation/reservation counters exclude some driver/display usage. PyTorch warns that `roi_align_backward_kernel` lacks a deterministic CUDA implementation; the approved warn-only policy permits it. Cross-run bitwise training reproducibility is therefore not guaranteed.

There is no defensible end-to-end 100-epoch estimate because neither full training nor validation completed. Pure arithmetic from the failed prefix gives roughly 86 s of training per epoch, or **2.39 hours for 100 training prefixes scaled to full length**, excluding unmeasured validation/checkpoint overhead and adding startup separately. This is explicitly an unreliable prefix projection, not a safe planning estimate or a promise that the run can complete.

## Failure investigation: facts and remaining uncertainty

The failing identity is `train/image1_518_png.rf.317b927d25a648e43ad0f90efc74b147`. Its source annotation is empty and remains empty after approved augmentation. The decoded/transformed input is finite, 3×640×640, with valid pixel range and no invalid target coordinates. Data hashes match canonical lineage.

A **forward-only**, no-gradient/no-optimizer probe using freshly initialized COCO weights and the same seeded augmentation produces **101 proposals** and finite losses: classifier 2.4128621, objectness .0295272, both regression terms 0. This is a valid input check, not a replay of the failed trained state. It does not establish that empty labels are the cause or justify removing them.

The installed [ROI loss implementation](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/detection/roi_heads.py) produces NaN classifier and regression losses for zero sampled proposals/labels; a deterministic synthetic call confirms that behavior. The [RPN filter](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/detection/rpn.py) applies its score floor during training as well as inference. A learned-background model could consequently supply zero proposals for an empty-target image. That is a **plausible mechanism, not a confirmed diagnosis**: the actual failed model, failing loss components and proposal counts were not retained. An unstable internal activation is not ruled out, and lowering LR is not supported by current evidence.

The smallest next step is a separately approved diagnostic retry at **unchanged settings**, retaining component values, proposal counts and failing state before any scientific correction. If that capture confirms empty proposals, a narrowly scoped candidate correction is training-only RPN score floor 0 while retaining inference .05, subject to explicit protocol review and approval. No such change was implemented or tested in training here. No LR/architecture/augmentation/configuration sweep occurred.

## Post-failure safeguards and preservation

The exact executed implementation is archived locally with hashes and identified by infrastructure commit `169394d`. After the failure:

1. Non-finite-loss exceptions now retain component values, actual RPN proposal counts, sample/box/input metadata, progress, previous updates and finite-loss summaries.
2. The runner saves a diagnostic-only failing model, partial gradients and optimizer/scheduler state. This is marked non-resumable and cannot masquerade as a completed best/last checkpoint.
3. Official mode additionally requires a successful full-epoch smoke summary matching current configuration, implementation hashes and dataset fingerprints. The current failed summary cannot authorize official training.

These changes have synthetic test coverage and **were not followed by GPU training**. They do not retroactively capture the original failure. Three F source/test files intentionally changed after execution; 269 of the 272 execution-baseline files remain byte-identical. **All 234 pre-existing protected files remain unchanged**, including A–E configurations/checkpoints/results/manifests, original tests and Phase 4 source/evidence, pinned dependency declarations and the three unrelated Phase 2F edits. Train/validation lineage/fingerprints were independently rechecked after failure.

Compact public evidence: `stage1_smoke_validation.json`. Full run summary, update logs, execution-source snapshot, model/input audit, seven local previews and the forward-only probe remain under ignored `outputs/training/experiment_f/`. No dataset, weight, checkpoint or X-ray image is staged for Git. Local ignored `docs/ROADMAP.md` and `docs/DECISIONS.md` record the blocked stage; they are not force-added.

**Readiness: blocked.** A complete numerically stable epoch and full validation/checkpoint gate remain required. Do not execute the official command until the blocker is diagnosed, any necessary protocol change is approved and a matching full smoke gate succeeds. This report describes engineering evidence, not final thesis prose or clinical validation.
