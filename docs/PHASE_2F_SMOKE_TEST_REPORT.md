# Phase 2F Experiment Freeze and Smoke-Test Report

## Status

The executable A/B/C/D experiment matrix is frozen and its configuration, environment, dataset, and C/D pairing checks pass. Phase 2F is **not complete or approved**, however, because Experiment A reproducibly produced non-finite validation losses during the real one-epoch smoke test.

The stability gate stopped Experiments B, C, and D. No official 100-epoch training was started, and the test split was not used.

## Frozen experiment matrix

| Experiment | Image condition | Controlled training augmentation |
|---|---|---|
| A | Pixel-identical PNG control | Off |
| B | CLAHE PNG | Off |
| C | Pixel-identical PNG control | Conservative policy |
| D | CLAHE PNG | Conservative policy |

All experiments inherit `configs/training/baseline.yaml`. The resolver proved that A/B and C/D differ only in the three dataset path/fingerprint fields, while A/C and B/D differ only in `degrees`, `translate`, `scale`, and `hsv_v`. All model, optimizer, learning-rate, weight-decay, seed, deterministic, AMP, worker, validation, checkpoint, and test-isolation settings remain fixed.

The shared launcher is `bone_fracture_pipeline.experiment_runner`. Smoke mode changes only the epoch count from 100 to 1 and routes output from `outputs/training/official/` to `outputs/training/smoke/`. The official path was never invoked.

## Preflight results

The following checks passed before Experiment A trained:

- PNG control fingerprint `5dd43c8e40eda542a3d77ad65aad29e6964fbba6bb9d4ac77006413a7b8bb1ce`;
- 1,728 images, 1,728 labels, 998 annotations, and 868 empty labels;
- approved 1,211/348/169 train/validation/test split counts and six-class order;
- byte-level pretrained-weight hash `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`;
- Phase 2C Python, PyTorch, torchvision, Ultralytics, CUDA, cuDNN, driver, GPU, VRAM, OpenCV, NumPy, Pillow, and PyYAML environment match;
- CUDA device 0 availability and absence of Albumentations;
- validation-only checkpoint selection and final-evaluation-only test policy;
- final C/D dataset identity and stochastic pairing through the resolved launcher configurations.

The C/D check matched all 1,728 sample identities and reproduced the same representative transformed box geometry, Python/NumPy RNG progression, shuffle order, and worker seeding under the frozen conditions.

## Smoke execution

The first A attempt stopped before training because Ultralytics interpreted `path: .` relative to the process directory. The launcher was corrected to write a run-local dataset YAML containing the absolute, fingerprint-validated dataset root. The failed attempt was preserved below `outputs/training/smoke/_failed_attempts/`.

A subsequent real one-epoch training completed but was rejected by an overly strict post-run comparison because Ultralytics serializes CUDA device `0` as the string `"0"`. The comparison was corrected without changing the requested device. That completed attempt was also preserved.

The final A retry used committed launcher revision `90d86908f7d3afa36cb58921452e7195055e0847` and completed all 152 training batches plus validation with batch size 8, image size 640, eight workers, AMP, seed 42, and deterministic mode. It did not OOM. The trainer displayed a maximum of approximately 1.95 GiB GPU memory during the epoch. `last.pt`, `best.pt`, `results.csv`, plots, and three trainer batch images were produced.

Training losses were finite:

| Value | Result |
|---|---:|
| train box loss | 3.24173 |
| train classification loss | 26.9853 |
| train DFL loss | 3.01948 |

All three validation losses were `nan`. The prior completed A attempt produced the same three finite training losses and the same three non-finite validation losses, so the failure is reproducible under seed 42. The saved checkpoint parameters were independently checked and were finite. The one-epoch metrics were zero, but they are technical smoke outputs and must not be interpreted as thesis performance results.

The optional Ultralytics AMP auxiliary self-check could not download `yolo26n.pt` in the restricted network environment. Ultralytics explicitly retained `amp=True`. This is recorded as an environment observation, not asserted as the cause of the non-finite validation losses.

## Trainer batch QA

`train_batch0.jpg`, `train_batch1.jpg`, and `train_batch2.jpg` from Experiment A were manually inspected. The visible boxes were technically aligned with the displayed anatomy, no invalid or out-of-frame boxes were observed, and no controlled rotation, translation, scaling, or intensity transform was apparent. Standard letterbox padding remained present as expected.

This review is technical pipeline QA only. It is not a medical-correctness assessment.

## Gate decision

Experiment A failed the explicit requirement that validation losses be finite. The post-run pipeline therefore stopped before fixed-sample inference and before launching B, C, or D. Continuing the matrix, changing batch size, disabling AMP, or accepting the `nan` values would have bypassed the frozen smoke-test criteria.

Phase 2 is not fully complete, the official A/B/C/D experiment matrix is not approved to start, and Phase 3 must remain paused. The next work is a focused diagnosis of the non-finite validation-loss path under the pinned environment, followed by clean successful A/B/C/D one-epoch smoke runs using the same frozen scientific settings.

## Evidence

Compact evidence is stored in `docs/evidence/phase2f/`:

- `experiment_matrix_validation.json`;
- `environment_freeze.json`;
- `cd_pairing_validation.json`;
- `smoke_test_summary.json`;
- `smoke_run_summary.csv`;
- `smoke_failure_diagnostic.json`;
- `manual_batch_review.json`.

Full local outputs and checkpoints remain under ignored `outputs/training/smoke/`. These smoke artifacts are not official thesis results.
