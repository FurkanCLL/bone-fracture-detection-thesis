# Phase 2E PNG Control and Controlled Augmentation Policy

## Status

Phase 2E is complete. A pixel-preserving PNG control dataset was created for Experiments A and C, and the conservative train-only augmentation policy for Experiments C and D was validated against Ultralytics 8.4.155. No model training, mAP-based tuning, or test-set visual review was performed.

The implementation and technical QA are approved for Phase 2F smoke testing. This approval does not show that augmentation or CLAHE improves detector performance.

## PNG control normalization

The base prepared dataset contains JPEG images, while the CLAHE condition contains PNG images. Experiments A and C therefore use `data/prepared/v3_detection_png` so image-container format is not an uncontrolled difference in the A/B/C/D comparison.

The conversion performs only:

```text
decode approved JPEG pixels -> encode the identical matrix as lossless PNG
```

It performs no grayscale conversion, intensity change, normalization, resize, crop, rotation, denoising, sharpening, CLAHE, or augmentation. The decoder rejects non-neutral EXIF orientation or a source profile other than the approved three-channel `uint8` profile.

Full validation confirmed:

| Check | Result |
|---|---:|
| PNG control images | 1,728 |
| Labels | 1,728 |
| Annotations | 998 |
| Empty labels | 868 |
| Exact decoded pixel matches | 1,728 |
| Pixel mismatches | 0 |
| Dimension/channel/dtype mismatches | 0 |
| Label hash mismatches | 0 |
| Decode or PNG-format failures | 0 |

The approved source fingerprint remained unchanged at `c5031d9937e2a1d917a2369f327b38a59a4b5e8bd40ae2da659451840c7d41b1`. The PNG control fingerprint is `5dd43c8e40eda542a3d77ad65aad29e6964fbba6bb9d4ac77006413a7b8bb1ce`. An independent temporary rebuild reproduced the same fingerprint across all 3,457 files.

## Frozen augmentation condition

The version-controlled policy is `configs/training/augmentation_conservative.yaml`. It inherits the Phase 2C baseline rather than duplicating unrelated model and optimizer settings.

| Setting | Frozen value | Installed behavior |
|---|---:|---|
| `degrees` | `10.0` | uniform angle from -10° to +10° |
| `translate` | `0.05` | independent x/y center translation within ±5% of output size |
| `scale` | `0.10` | uniform multiplicative scale from 0.90 to 1.10 |
| `hsv_v` | `0.15` | uniform value multiplier from 0.85 to 1.15 with `uint8` truncation/clipping |
| `hsv_h` | `0.0` | disabled |
| `hsv_s` | `0.0` | disabled |

`augment` remains `false` because the installed configuration defines that option as prediction-time augmentation. Training augmentation is activated by Ultralytics when it constructs the dataset in `train` mode. Behavioral validation confirmed that `val` and `test` modes take the non-augmentation path.

The installed package does not include Albumentations. This matters because Ultralytics would otherwise construct a separate set of low-probability default transforms. The validator fails if Albumentations appears later without an explicit approved policy.

## Explicitly disabled transforms

The policy sets the following to zero, null, or otherwise off:

- horizontal and vertical flip;
- shear and perspective;
- Mosaic, MixUp, CutMix, and copy-paste;
- BGR/channel swapping;
- AutoAugment and random erasing;
- third-party default Albumentations transforms.

Validation and test augmentation remain disabled. The policy creates no offline augmented image copies.

## C/D sample identity and stochastic pairing

The PNG control and CLAHE datasets contain the same 1,728 identities using `<split>/<source stem>`. All split counts, dimensions, dtypes, and label hashes match. Their fingerprints are:

- PNG control: `5dd43c8e40eda542a3d77ad65aad29e6964fbba6bb9d4ac77006413a7b8bb1ce`;
- CLAHE: `ca8286b35d3d31f0b8074aa387a44ca6392178926c23bde61ea6d99be70aba5f`.

Ultralytics 8.4.155 seeds Python, NumPy, and PyTorch, supplies a seeded DataLoader generator, and derives Python/NumPy worker seeds from PyTorch worker seeds. A deterministic harness reset seed 42 for matched control and CLAHE sequences. Across eight representative train identities covering the six classes, transformed box geometry and the Python/NumPy RNG-state progression were identical. Shuffle and worker-seeding checks were also reproducible.

This supports paired C/D augmentation under the frozen conditions, but it is conditional rather than universal. C and D must start as fresh runs with the same seed, sample ordering, cardinality, labels, dimensions, batch size, eight-worker policy, cache setting, rectangular-training setting, dependency versions, and Ultralytics version. Resuming from different RNG states or changing any of those conditions can break exact per-sample pairing. Phase 2F must repeat the pairing check against the final run launcher.

## Technical preview QA

Nine deterministic train-only comparisons exercise positive and negative rotation, translation, scale 0.90/1.10, value ±15%, a small box, a boundary-adjacent box, multiple annotations, and all six classes. The endpoint cases intentionally show the maximum approved transform strengths.

Automated checks found no invalid boxes. Manual review confirmed aligned transformed boxes, valid clipping for the boundary case, conservative appearance, and no unexpected transform. No validation or test image was used. The review is recorded in `docs/evidence/phase2e/manual_preview_review.json`; the generated contact sheet remains under `outputs/phase2/phase2e/augmentation/preview/`.

## A/B/C/D input matrix

| Experiment | Image dataset | Controlled training augmentation |
|---|---|---|
| A | `v3_detection_png` | Off |
| B | `v3_detection_clahe` | Off |
| C | `v3_detection_png` | Conservative policy on |
| D | `v3_detection_clahe` | Conservative policy on |

The model, pretrained weights, image size, batch size, epoch budget, optimizer, learning rate, weight decay, seed policy, checkpoint rule, and metrics remain inherited from the frozen Phase 2C protocol.

## Reproduction and evidence

Run the two Phase 2E validators with:

```powershell
python -m bone_fracture_pipeline.png_control
python -m bone_fracture_pipeline.augmentation_policy
```

Large manifests and preview images are ignored below `outputs/phase2/phase2e/`. Compact summaries, the preview index, and manual review decision are tracked under `docs/evidence/phase2e/`.

The final project suite passed all 56 tests. An additional independent full-dataset check re-decoded every JPEG/PNG pair and independently confirmed 1,728 exact pixel matches, 1,728 byte-identical labels, equal control/CLAHE identity sets, and the recorded source/control fingerprints.

Phase 2F must still create/freeze the four final experiment configurations, run the short training smoke checks, confirm batch-8 VRAM stability, verify configuration application, and repeat the C/D pairing check through the final launcher. Until then, no official experiment should begin.
