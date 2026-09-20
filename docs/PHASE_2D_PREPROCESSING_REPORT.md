# Phase 2D Custom X-ray Preprocessing Report

## Status

Phase 2D is complete. The deterministic CLAHE dataset was created at `data/prepared/v3_detection_clahe` from the approved `data/prepared/v3_detection` source. The source dataset and all labels remain unchanged. No model training, hyperparameter tuning, augmentation definition, or test-set visual review was performed.

The derived dataset passes technical preprocessing validation and is approved for Phase 2E planning and the later controlled preprocessing conditions. This approval does not establish that CLAHE improves object-detection performance or medical correctness.

## Frozen preprocessing pipeline

Every image follows the same fixed operations:

1. decode the stored pixels without applying an orientation transform;
2. standardize the image to one 8-bit grayscale channel;
3. apply OpenCV CLAHE with `clipLimit = 2.0` and `tileGridSize = (8, 8)`;
4. replicate the processed grayscale channel three times;
5. save the derived image as lossless PNG.

OpenCV 4.14.0 was used with one processing thread and PNG compression level 3. The source profile was uniform: all 1,728 images were three-channel, 8-bit JPEG files with EXIF orientation value 1. The decoder checks encoded dimensions and rejects non-neutral EXIF orientation rather than silently rotating an image. No resize, crop, rotation, denoising, sharpening, gamma correction, or augmentation is included.

## Dataset and label integrity

| Check | Result |
|---|---:|
| Images | 1,728 |
| Label files | 1,728 |
| Annotations | 998 |
| Empty label files | 868 |
| Byte-identical label files | 1,728 |
| Label hash mismatches | 0 |
| Dimension mismatches | 0 |
| Decode failures | 0 |
| Non-PNG outputs | 0 |
| Non-three-channel outputs | 0 |
| Outputs with unequal channels | 0 |

The train, validation, and test membership and the six-class mapping are unchanged. Every output image has the source width and height, and every copied label is byte-identical to its source.

The approved source fingerprint was unchanged before and after preprocessing:

```text
c5031d9937e2a1d917a2369f327b38a59a4b5e8bd40ae2da659451840c7d41b1
```

The derived 3,457-file dataset fingerprint is:

```text
ca8286b35d3d31f0b8074aa387a44ca6392178926c23bde61ea6d99be70aba5f
```

An independent temporary rebuild produced the same fingerprint across all 3,457 files. The temporary rebuild was then removed. Existing destination protection and transactional replacement prevent a failed build from leaving a partial approved dataset.

## Intensity results

The following statistics describe pixel changes; they are not detector-performance results.

| Scope | Images | Original mean | CLAHE mean | Original standard deviation | CLAHE standard deviation | Images with increased standard deviation |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 1,728 | 53.970 | 65.079 | 50.100 | 52.410 | 1,653 |
| Train | 1,211 | 53.039 | 64.141 | 49.660 | 52.252 | 1,161 |
| Validation | 348 | 54.748 | 66.309 | 49.562 | 52.110 | 328 |
| Test | 169 | 58.717 | 69.026 | 53.605 | 53.775 | 164 |

Across the complete dataset, the pooled mean increased by 11.110 intensity levels and the pooled standard deviation increased by 2.310. The median per-image standard deviation increased from 29.872 to 38.347. These measurements are consistent with a local-contrast transformation, but they do not prove that fracture regions became easier for a model to detect. Test statistics were generated automatically for integrity documentation only; the test images were not selected or interpreted visually.

## Visual quality assurance

The deterministic visual package contains 18 side-by-side source/CLAHE comparisons: 16 train images and 2 validation images. No test image is included. Selection covers every class and examples ranked for dark intensity, bright intensity, low contrast, small annotated regions, multiple annotations, and unusual aspect ratio.

Manual inspection found:

- matching orientation, framing, dimensions, and box placement in each pair;
- no severe clipping or processing artifact in the reviewed samples;
- clearer local separation in dark and low-contrast examples;
- additional visibility of some background texture and noise, an expected CLAHE limitation;
- no visual basis for a medical-correctness or accuracy claim.

The review decision is recorded in `docs/evidence/phase2d/manual_visual_review.json`. Full review images and the contact sheet remain under `outputs/phase2/phase2d/visual_review/`.

## Reproducibility and evidence

Run the fixed builder with:

```powershell
python -m bone_fracture_pipeline.preprocess_dataset
```

The command refuses to overwrite existing output unless `--overwrite` is supplied. Full generated manifests, review images, and local evidence are written below `outputs/phase2/phase2d/`; compact canonical evidence is stored under `docs/evidence/phase2d/`.

The final automated suite contains 49 passing tests, including focused preprocessing tests for deterministic pixels, 8-bit conversion, orientation/geometry guards, train/validation-only review selection, atomic overwrite protection, source preservation, label identity, output encoding, and independent rebuild reproducibility.

## Interpretation and next boundary

Phase 2D establishes one controlled, reproducible preprocessing condition. It does not establish superiority over the original images. Phase 2E may now define augmentation independently without changing this preprocessing contract. Phase 2F must still verify dataset loading and the frozen batch-size/training smoke protocol before official A/B/C/D experiments begin.

This remains an experimental dataset-level localization pipeline. The technical QA result is not evidence of clinical safety, reliability, or generalization to hospital populations.
