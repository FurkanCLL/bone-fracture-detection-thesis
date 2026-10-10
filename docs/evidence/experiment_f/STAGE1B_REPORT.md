# Experiment F Stage 1B: numerical diagnosis and complete smoke gate

**Official training readiness: PASS. Official training was not started.** The final fresh smoke completed all **1,211 training images**, **19 optimizer updates**, **348 validation images**, shared AP, compatible diagnostics and both checkpoint reloads. This is engineering feasibility evidence, not an official Experiment F performance result or a D–F superiority claim. The user will launch the 100-epoch run separately after reviewing this record.

## Root cause and captured-state verification

Historical Stage 1 remains a failed attempt; its report and JSON were not overwritten. The unchanged-scientific-config Stage 1B diagnostic used physical batch 1, accumulation 64, RPN threshold .05, seed 42, the original LR schedule, COCO initialization, FP32 and 640-square geometry. It was bounded to 500 images if no failure occurred. It reproduced the exact failure at zero-based position **401**, sample `train/image1_518_png.rf.317b927d25a648e43ad0f90efc74b147`, after 401 finite backwards and six successful updates. Diagnostic timing was 59.54 s overall, including 25.41 s startup; the failing training interval was 33.34 s.

The target was empty. All 13,500 decoded proposal boxes, backbone features, RPN logits and deltas were finite. Maximum objectness was **0.0305564292**, below .05. Therefore **zero proposals reached ROI Heads**. The ROI predictor received shapes 0×7 and 0×28; native classifier and box regression losses were NaN. RPN objectness remained 0.0002009235 and RPN regression 0. Input shape was 3×640×640, pixels were finite, all parameters were finite, and all 70 pending gradient tensors were finite. LR was .0001056977704. This confirms the score-filter/empty-ROI mechanism for the actual trained failing state; it does not suggest defective empty-label images or an independent activation overflow.

The exact model, optimizer, scheduler, 17 pending image gradients, current RNG and pre-forward RNG were saved in ignored `stage1b_diagnostic/failure_capture.pt`. Its SHA-256 is `74c9a4a9688715d6d8a0f3e6baad26483cf2b07552a8b92c0bd3d0ae694aaa03`. Replay restored the same trained weights, gradients, sample augmentation and pre-forward sampling RNG for each trial. At .05 it reproduced zero proposals and both NaNs. At 0.0 it retained **2,000 proposals**, classifier loss .0036825179, objectness .0002009235 and both regression losses 0. Backward normalized by the original 64-image window produced finite accumulated gradients and parameters. No optimizer update was executed during replay. The original capture hash was independently rechecked afterward.

Read-only hooks and an instance-local proposal-filter wrapper observed native operations; installed Torchvision was not modified. The source audit confirms that .05 filtering also applies during native training, ROI cross-entropy averages an empty target vector, and ROI regression divides by zero sampled labels in this case.

## Protocol revision and minibatch semantics

Protocol **version 2** applies only the conditionally authorized numerical correction: **training RPN score threshold 0.0; validation/inference 0.05**. The captured evidence satisfied all three authorization conditions before implementation. This revision was documented before official training. Model architecture, COCO weights, backbone freezing, six foreground classes, canonical CLAHE data, conservative augmentation, FP32, resolution, LR target .001, AdamW betas .937/.999, decay .0005, clipping 10 and the full 100-epoch scheduler budget remain fixed.

The RPN subclass reuses the loaded head and anchor objects. `train()` and `eval()` switch thresholds immediately, with repeated-switch tests. Both values are explicit in config/model/checkpoint metadata; exact metadata and scheduler guards reject changed settings on resume. Official epoch-boundary resume is implemented and tested at the state/guard level, but a complete official resume was not executed.

Images and target dictionaries are passed as matching lists. The same seed-42 permutation and per-image augmentation seeds apply at batches 1/2/4; no image is skipped or duplicated. Optimization windows remain 64 **images**. A native microbatch objective for actual batch size `b` contributes `(b/W)*loss` to a `W`-image window. For the final 59-image window at batch 4, 14 batches of four contribute 4/59 each and one of three contributes 3/59. All 19 windows are stepped, ending at LR .0003333333333.

This is image-count weighting of **native microbatch objectives**, not a claim of identical independent per-image gradients. Native RPN BCE averages sampled anchors; RPN regression sums positives divided by all sampled anchors. Native ROI cross-entropy averages sampled labels; ROI regression sums positives divided by all labels. Available proposals, sampling denominators and RNG draws can differ across physical batches. Native losses are unchanged. Reported means/quantiles weight each microbatch value by its contributing image count; they are not individually measured per-image losses.

## Hardware-only batch selection

Each candidate began with byte-identical complete adapted model tensors, the same COCO weights, empty optimizer state and seed 42. Equal discarded four-image kernel warmup preceded full model/gradient/optimizer/RNG reset. Initialization/download/kernel warmup time is excluded and recorded separately. Each measured candidate consumed the same 128-image prefix, including all six classes, **65 empty-label images**, **75 targets**, target counts 0–3 and original widths 147–512/heights 148–512. Actual inputs are uniformly 640-square. This is representative prefix coverage, not a worst-case original-dimension or target-count stress test; the final smoke covers the entire training split.

| Batch | Accumulation | Allocated MiB | Reserved MiB | Images/s | Average batch s | Reserved headroom |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 568.88 | 606 | 10.03 | 0.0997 | 90.1% |
| 2 | 32 | 732.88 | 794 | 13.01 | 0.1537 | 87.1% |
| 4 | 16 | 1050.43 | 1140 | 15.68 | 0.2551 | 81.4% |

All candidates completed two optimizer windows of exactly 64 images, with finite losses, gradients and parameters. Batch **4 / accumulation 16** was selected **before validation AP**: it exceeded the predeclared 10% throughput-gain criterion and retained more than the required 25% reserved-capacity margin. Its gain over batch 1 was 56.3%. No architecture or hyperparameter sweep occurred. CUDA allocation counters omit driver/display usage; laptop thermals and other applications can affect future measurements.

## Full smoke attempts and implementation recovery

The first Stage 1B full attempt (`stage1b_smoke`, implementation `9d29b28`) successfully trained all 1,211 images in **76.88 s**, all 19 updates, including the final 59-image window. Validation inference reached the schema check after the complete 348-image loop, but the adapter represented GT `xyxy` as Python tuples while the canonical checker requires lists. This was a serialization bug, not a new numerical instability or a changed annotation. AP and checkpoint saving were not reached. The rows and trained state were not retained by that version; no missing metrics or checkpoint are reconstructed or claimed.

The adapter now emits coordinate lists directly. A regression test exercises `prediction_row → verify_sample_export` in memory, without a JSON roundtrip masking the mismatch. Predictions are retained before downstream checks, and future nonnumerical execution errors retain an explicitly non-resumable model/optimizer/RNG capture. The user **explicitly authorized one additional fresh smoke attempt** after this tested fix. Both attempts remain recorded; the successful attempt is not presented as the only Stage 1B training attempt.

The additional attempt (`stage1b_smoke_retry`, implementation **`db8982d`**) restarted from original COCO initialization with the same selected scientific settings. It completed 303 physical batches, 1,211 distinct image identities in the independently verified order, 19 updates and 348 validation images. No further correction or AP-driven tuning occurred.

| Final smoke measurement | Value |
|---|---:|
| Loss minimum / mean / maximum | 0.00236979 / 0.49676918 / 2.42636871 |
| Shared validation mAP50 | 0.0010088163 |
| Shared validation mAP50–95 | 0.0001812778 |
| Training / validation | 61.46 s / 11.65 s |
| AP / checkpoint saving and reload | 0.585 s / 0.951 s |
| Startup / total run | 14.46 s / 94.38 s |
| Peak allocated / reserved | 1049.25 / 1136 MiB |
| GPU | RTX 4050 Laptop, 6,141 MiB |

Compatible diagnostics at confidence >=.25, IoU >=.50 use the unchanged Phase 4 confidence-first matcher. The complete class-aware and class-agnostic counts, P/R/F1, per-class supports and size-utility compatibility are in `stage1b_smoke_validation.json`. Both modes give **0 TP, 0 FP, 204 FN**, with precision/recall/F1 0 under the existing zero-denominator convention: no predictions reach confidence .25. There are 86,857 candidates above the AP floor, so these fixed-threshold zeros are distinct from the small nonzero shared AP. These describe a one-epoch feasibility model, not official research performance.

Best and last checkpoints were both saved, reloaded and compared tensor-for-tensor with the trained model. Both include FP32 weights, actual optimizer/scheduler progress, RNG, config/source hashes, model mappings and dataset provenance. They are byte-identical in this first completed epoch, as expected. SHA-256: `ef7d549bc08b829746455d59d854a3a0e990bb8fcf1f83ff1800f10995ed1785`. Original-coordinate predictions passed exact sample/GT/hash and coordinate validation; their SHA-256 is `b728f69d2d81aa751a0e5e862f47c2396de106bd7c3428d1d650425c475ebe87`.

A single-epoch extrapolation of training + validation + AP + checkpoint work gives **2.07 hours** for 100 epochs, with a rough **2.07–3.11 hour** planning range before extra startup/integrity/diagnostic overhead. This is not a measured 100-epoch duration or stability guarantee. Proposal availability, checkpoint frequency, thermal throttling and OS activity may change costs.

## Shared evaluation, preservation and remaining differences

The shared AP module is unchanged. D remains **.1492271831 / .0474394016** under this definition. Official D remains .14861/.04732 and native checkpoint reproduction .1485241076/.0473690941. Its original absolute .00005 AP50 reproduction discrepancy remains failed and unaltered. Eventual primary D–F comparison must use shared AP for both; the independent Phase 4 matcher is not substituted for AP.

Relevant differences remain explicit: YOLOv8s versus Faster R-CNN; distinct COCO initialization; D EMA versus F direct FP32 weights; D trainable normalization versus F FrozenBatchNorm and frozen backbone blocks 0–6; RPN/ROI proposal counts, NMS and stochastic sampling; framework loss reductions; D summed losses/dynamic accumulation and bias-specific warmup versus F fixed image windows/averaged native minibatch losses/uniform warmup; D physical batch 8 versus F 4; initial resize rounding and validation rectangular padding versus F square letterbox; and the training-only RPN correction. Matching selected constants does not make optimization numerically identical.

**47 focused tests, 207 complete regression tests and 54 compilation checks passed.** Tests cover batches 1/2/4, exact order/counts, mixed targets, image-weighted gradients, final partial batches, zero-ROI NaNs, capture/RNG retention, phase switching, initialization reset, config/resume rejection, checkpoints, official gates, test isolation, validation augmentation and existing AP fixtures. Independent read-only verification recalculated AP exactly, checked both checkpoint hashes and the failing-state hash, verified the epoch permutation and partial window, and accepted the successful official gate without launching it.

All **236 Stage 1B protected files** remained byte-identical, including the original 234 A–E/Phase 4/dependency/Phase2F artifacts and both historical public F records. The successful run also verified its 277-file execution preservation map. Restricted train/validation fingerprints and original CLAHE lineage match before/after:

- Train: `d658084e25bb3d7f19634f5328bdd6ae465dcd51f1c770df42f2589c87d2a05e`.
- Validation: `d43b38126e40e060b79b130c6475db2243f797f191d95038eee1e9d2491d7a78`.

Held-out test IO counters are zero for diagnostic, replay, benchmark, both smoke attempts and independent verification. No raw/canonical dataset, original A–E checkpoint/result/manifest, shared AP definition, Phase 4 source/evidence or unrelated Phase 2F edit changed. No dependency change occurred. X-rays, weights, caches, failed-state tensors and checkpoints remain ignored locally. Local ignored ROADMAP/DECISIONS are updated without force-adding them.

Complete numerical evidence and source/config/weight hashes are in `stage1b_smoke_validation.json`. Diagnostic source is versioned at `37dbf90`; measured replay/benchmark source snapshots were independently recovered and checked against their execution hashes. Final smoke source is versioned at `db8982d`. Scoped commits preserve the investigation, correction/minibatches, schema repair and final evidence separately. This report is implementation documentation, not final student-authored thesis prose or clinical validation.

**PASS means the current protocol passed the full engineering smoke gate. Official 100-epoch training remains unstarted and requires the user's separate launch.**
