# Phase 2C Baseline Training Protocol

## Status

Phase 2C froze the common YOLOv8 training protocol. Phase 2F later amended its numerical-precision setting after a causal validation-overflow diagnosis; no official training or test-set evaluation has been performed.

The version-controlled source of truth is `configs/training/baseline.yaml`. Compact validation evidence is stored under `docs/evidence/phase2c/`; generated framework settings and other local artifacts remain under the ignored `outputs/phase2/phase2c/` directory.

## Fixed baseline

| Setting | Frozen value |
|---|---:|
| Model | YOLOv8s detection |
| Initialization | COCO-pretrained `yolov8s.pt` |
| Dataset | `data/prepared/v3_detection` |
| Image size | 640 |
| Maximum epochs | 100 |
| Batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0005 |
| Primary seed | 42 |
| Deterministic mode | Enabled |
| AMP | Disabled |
| Early stopping | Disabled with `patience: 0` |
| Training device | CUDA device 0 |
| Generated run root | `outputs/training` |
| Primary metric | Validation mAP50-95 |

YOLOv8s is retained as the approved small-model baseline: it is substantially more capable than the nano variant while remaining practical for the available laptop GPU. An input size of 640 follows the approved protocol and avoids adding a resolution comparison to the controlled preprocessing/augmentation study. The 100-epoch maximum is identical for all future A/B/C/D conditions.

Batch size 8 remains fixed. Phase 2F confirmed it across all four one-epoch smoke runs with a maximum observed 3.088 GiB allocated and 3.686 GiB reserved CUDA memory.

AMP was originally enabled. Phase 2F proved that the trained A checkpoint overflowed in FP16 validation inference while the same batches, targets, weights, and loss path stayed finite in CUDA and CPU FP32. AMP is therefore disabled globally for A/B/C/D as a documented numerical-stability correction, not as an experiment-specific tuning choice.

## Zero-augmentation baseline

Ultralytics 8.4.155 enables several detection transforms by default, including HSV changes, translation, scale, horizontal flip, and mosaic. The baseline explicitly provides the following overrides:

```yaml
augment: false
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0
degrees: 0.0
translate: 0.0
scale: 0.0
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.0
bgr: 0.0
mosaic: 0.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
close_mosaic: 0
auto_augment: null
erasing: 0.0
```

The validation tool checks every key against the installed Ultralytics configuration parser. `auto_augment` and `erasing` are primarily classification settings, but they are neutralized as an extra guard. `copy_paste_mode` is intentionally not set because `copy_paste: 0.0` disables that transform. Resize/letterbox, tensor conversion, and numerical normalization remain normal model input preparation and are not the experimental augmentation variable.

## Model selection and metrics

The selected checkpoint is the checkpoint with the highest validation `metrics/mAP50-95(B)`. For pinned Ultralytics 8.4.155, the protocol tool verifies at runtime that detection fitness weights are `[0, 0, 0, 1]` for Precision, Recall, mAP50, and mAP50-95. Therefore the framework's `best.pt` exactly follows the primary validation metric in this environment.

This equivalence is version-sensitive and is checked on every protocol-validation run. If it fails after a framework change, official training must pause until checkpoint selection is implemented explicitly from validation mAP50-95. The test split must never be used for checkpoint selection.

In addition to the primary metric, official runs will retain validation mAP50, Precision, Recall, and per-class AP50-95. Final test-set evaluation belongs to the later fixed-protocol evaluation stage.

## Seed and split policy

Seed 42 is the primary seed for every controlled comparison. Seeds 43 and 44 are recorded only as conditional final repetitions if time and compute permit; they are not run in Phase 2C.

Training uses `train`, iterative model assessment and checkpoint selection use `valid`, and `test` is reserved for final evaluation. The prepared dataset fingerprint is fixed at `c5031d9937e2a1d917a2369f327b38a59a4b5e8bd40ae2da659451840c7d41b1`, matching the Phase 2B approval.

## Validated environment

The successful validation recorded:

- Windows 11, CPython 3.12.14;
- PyTorch 2.6.0+cu126 and torchvision 0.21.0+cu126;
- Ultralytics 8.4.155;
- CUDA runtime 12.6 and cuDNN 9.5.1;
- NVIDIA GeForce RTX 4050 Laptop GPU, 5.997 GiB detected VRAM, driver 560.94;
- OpenCV 4.14.0, NumPy 2.5.2, Pillow 12.3.0, and PyYAML 6.0.3;
- `yolov8s.pt`, 22,588,772 bytes, SHA-256 `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`.

The environment validator records current deterministic backend flags separately from the requested training configuration. They are not expected to be globally active before a trainer is initialized. Ultralytics must apply the configured seed and deterministic mode when official training starts.

## Validation performed

The protocol tool confirmed that:

- the baseline YAML parses and contains the fixed values;
- the prepared dataset and its six-class `data.yaml` exist;
- all expected split directories exist;
- the current 3,457-file prepared-dataset fingerprint equals the Phase 2B fingerprint;
- all augmentation-off keys are supported and accepted by the installed framework;
- `patience: 0` disables early stopping in the installed version;
- detection fitness exactly equals mAP50-95;
- the COCO-pretrained YOLOv8s detection model and weights load successfully;
- the test split is excluded from fitting and checkpoint selection;
- no official training was performed.

## Phase 2F verification

Phase 2F confirmed batch-8 VRAM stability, launcher argument fidelity, finite training and validation results, checkpoint creation, fixed-sample validation inference, and the intended A/B/C/D transform behavior. Exact repeatability is not guaranteed across all PyTorch releases, platforms, or devices even when deterministic settings are requested.

This remains an experimental dataset-level localization pipeline. Protocol validation and later dataset metrics do not establish clinical safety, clinical reliability, or generalization to hospital populations.
