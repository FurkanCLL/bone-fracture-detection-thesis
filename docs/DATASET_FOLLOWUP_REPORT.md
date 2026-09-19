# Dataset Follow-up Report

**Project:** RTU bachelor thesis — bone fracture localization in X-ray images

**Date:** 2026-09-14

**Status:** Pre–Phase 2 follow-up complete; Phase 2 has not started

## 1. Outcome

The downloaded version 4 dataset **does contain offline augmented training images**. This is confirmed by two independent forms of evidence:

1. The exact [Roboflow Universe version 4 page](https://universe.roboflow.com/veda/bone-fracture-detection-daoon/dataset/4) linked in the local `data.yaml` states that the version has three outputs per training example, rotation from -15° to +15°, exposure changes from -25% to +25%, and no preprocessing.
2. Local feature matching identifies 2,895 high-confidence and 348 probable transformed-image pairs. The high-confidence pairs form 1,107 groups and show the expected small rotations and exposure differences.

Roboflow documents that its dataset-version augmentations are applied only to training images and are materialized as [offline augmentation](https://docs.roboflow.com/datasets/dataset-versions/image-augmentation). This agrees with the local evidence: all confirmed derivative groups are confined to `train`.

No high-confidence or probable derivative relationship was found across train, validation, or test in the candidate set. This is reassuring for direct augmentation leakage, but it does **not** prove patient- or study-level independence.

A separate label-consistency problem was found. Four confirmed training derivative groups contain both empty and annotated label files. They contain 12 exported images in total: five empty-label images and seven annotated images. The raw files were not changed.

## 2. Evidence boundaries

This report uses three evidence levels:

- **Verified local fact:** directly reproduced from raw files or generated audit tables.
- **Confirmed external fact:** stated on the exact source/version page connected to the local export metadata.
- **Interpretation:** the most plausible explanation supported by the evidence, but not a source-defined semantic rule.

The image matching is a screening and confirmation workflow, not patient identification. It cannot establish that two visually similar radiographs came from different patients, and it may miss transformations outside its candidate-generation or alignment assumptions.

## 3. Exact dataset provenance

The canonical local export is:

```text
data/raw/bone-fracture-detection/bone fracture detection.v4-v4.yolov8
```

Its `data.yaml` identifies:

- Roboflow workspace: `veda`
- project: `bone-fracture-detection-daoon`
- version: `4`
- license: `CC BY 4.0`
- source URL: `https://universe.roboflow.com/veda/bone-fracture-detection-daoon/dataset/4`

The [Roboflow version page](https://universe.roboflow.com/veda/bone-fracture-detection-daoon/dataset/4) reports exactly 4,148 exported images: 3,631 train, 348 validation, and 169 test. These counts match the local Phase 1 audit exactly.

The selected dataset was downloaded through the [Kaggle listing by Parisa Karimi Darabi](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project). The Kaggle description names six fracture-region classes and does not include the local export's additional bare `humerus` class. It also does not define what an empty label means.

## 4. Offline augmentation investigation

### 4.1 Method

Candidate pairs were formed from the union of:

- repeated filenames after removing the Roboflow `.rf.<hash>` suffix; and
- the Phase 1 difference-hash candidates with Hamming distance at most 4.

This produced 6,719 non-identical candidate pairs over the 4,148 canonical images.

Each candidate was checked with:

- ORB keypoints and binary descriptor matching;
- a ratio test to reject ambiguous feature matches;
- RANSAC partial-affine alignment for rotation, scale, and translation;
- aligned-pixel correlation;
- overlap ratio; and
- a linear intensity fit and residual error to distinguish exposure changes from structural differences.

The complete measurements are stored in `outputs/data_quality/dataset_followup/provenance/pair_analysis.csv`. The classifications intentionally use conservative thresholds:

| Classification | Meaning |
|---|---|
| `high_confidence_derivative` | Strong feature geometry and aligned-pixel agreement support the same underlying image. |
| `probable_derivative` | The evidence supports a derivative relationship but is below the strongest threshold. |
| `inconclusive` | Some similarity exists, but the evidence is not strong enough for either conclusion. |
| `unrelated_false_positive` | The filename/hash screening signal is contradicted by weak alignment or visibly different content. |

The completed run used OpenCV 4.14.0 and NumPy 2.3.5. These versions and the main matching settings are recorded in `summary.json`; compatible dependency ranges are declared in `pyproject.toml` and `requirements.txt`.

### 4.2 Results

| Classification | All candidate pairs | Cross-split pairs |
|---|---:|---:|
| High-confidence derivative | 2,895 | 0 |
| Probable derivative | 348 | 0 |
| Inconclusive | 159 | 24 |
| Unrelated false positive | 3,317 | 732 |
| **Total** | **6,719** | **756** |

The 2,895 high-confidence pairs form 1,107 connected groups: 964 groups of three images and 143 groups of two images, covering 3,178 exported training images. This does not recover every original image because weak-feature X-rays can fail matching and generic source keys can collide.

All 3,631 training files belong to repeated source-name groups. Their multiplicities are:

| Images sharing a stripped source name | Groups |
|---:|---:|
| 2 | 2 |
| 3 | 818 |
| 6 | 158 |
| 9 | 22 |
| 12 | 1 |
| 15 | 1 |

The dominant triplet pattern matches the source page's three outputs per training example. Larger groups are not automatically larger derivative families: manual inspection and feature results show that generic names such as `image1_244_png` can collide across unrelated source records.

Among the high-confidence pairs:

- 2,690 have an estimated relative rotation of at least 1°;
- rotation between two exported variants ranges approximately from -29.0° to +28.1°, which is compatible with comparing two independently rotated outputs from the documented -15° to +15° range; and
- 1,987 have a fitted relative exposure slope outside 0.9–1.1.

The fitted exposure slope is a pairwise diagnostic, not a reconstruction of Roboflow's exact percentage setting.

### 4.3 Interpretation

**High confidence:** the training export contains stored rotation and exposure derivatives created when Roboflow version 4 was generated. They are not merely on-the-fly transformations that would first be applied during future model training.

**High confidence:** validation and test were not subjected to the same three-output augmentation process. This follows both the source documentation and the absence of confirmed derivative groups outside training.

## 5. Cross-split leakage review

Phase 1 found no byte-identical images crossing splits. It produced 97 cross-split difference-hash candidates and 148 source-name groups that crossed splits. Neither signal alone was reliable.

The stronger follow-up evaluated 756 cross-split candidate pairs:

- zero were high-confidence derivatives;
- zero were probable derivatives;
- 24 were machine-inconclusive; and
- 732 were rejected as unrelated false positives.

All 24 machine-inconclusive pairs were placed first in the generated side-by-side review set. Visual review found different anatomy, different projections, different implants, or clearly different radiographic studies. Repeated stripped filenames were especially misleading. The gallery and metrics are under `outputs/data_quality/dataset_followup/provenance/cross_split_review/`.

**Conclusion:** there is no credible evidence of exact or offline-augmentation derivative leakage across the current splits within the methods used.

**Remaining limitation:** there are no patient or study identifiers in the YOLO export, filenames, or local metadata. Different views or visits from the same patient could therefore remain undetected. The current split cannot be called patient-independent from this export alone.

## 6. Empty-label findings

The Phase 1 counts remain unchanged: 2,088 of 4,148 label files are empty.

The available source descriptions do not explicitly say that an empty file is a radiologist-verified negative. The Kaggle description discusses fracture-region annotations but does not document a negative-image policy. Empty labels must therefore not be described as confirmed healthy images.

The derivative analysis provides direct evidence that at least a small subset is inconsistent:

| Stripped source name | Derivative files | Empty | Annotated | Annotation class in annotated variants |
|---|---:|---:|---:|---|
| `image1_244_png` | 3 | 1 | 2 | `fingers positive` |
| `image2_821_png` | 3 | 2 | 1 | `forearm fracture` |
| `image3_1128_png` | 3 | 1 | 2 | `forearm fracture` |
| `image3_278_png` | 3 | 1 | 2 | `forearm fracture` |

The images within each row are confirmed transformed versions of the same X-ray. In three groups, the fracture is also visually conspicuous. An empty label beside annotated transformed copies is therefore strong evidence of missing or inconsistently propagated annotations in those five exported files.

Sixty-two stripped source-name groups mix empty and annotated files, but only the four groups above are confirmed derivative groups. The remaining mixed groups must not be counted as label defects because many repeated generic names refer to unrelated images.

### 6.1 Radiologist review set

A reproducible review package was created at:

```text
outputs/data_quality/radiologist_empty_label_review/
```

It contains exactly 20 source-faithful X-ray copies:

- 16 from `train`;
- 4 from `valid`;
- 0 from `test`;
- fixed seed `20260914`;
- 20 unique source keys;
- 20 unique SHA-256 hashes; and
- no selected exact duplicate or obvious near-duplicate under the documented dHash and normalized fingerprint check.

The files are named `empty_01.jpg` through `empty_20.jpg`. `review_index.csv` records the original relative path, split, filename, dimensions, seed, source key, source SHA-256, and copied-file SHA-256. Every copied-file hash matches its source. `README.md` asks the reviewer to choose `no visible fracture`, `visible/suspected fracture`, or `uncertain` without assuming that an empty label is healthy.

The test split was deliberately excluded because the review is a pre-training data-quality check, not final test evaluation.

## 7. Rare `humerus fracture` investigation

The three `humerus fracture` annotations are not three independent source examples.

- All three images are in `train`.
- All share stripped source name `image1_10_png`.
- Every pair is a high-confidence derivative, with aligned correlations from 0.990 to 0.999.
- The three files form one derivative group.
- Every image also contains a separate `humerus` annotation.

The effective independent source count for `humerus fracture` is therefore **one**, subject to the limits of the export's provenance.

The median derived relative area of `humerus fracture` on these images is 0.4272, while the median `humerus` area on the same images is 0.0196. The larger polygon appears to cover much of the bone/arm, while the smaller polygon is localized. This nested relationship is real, but its intended semantic meaning is not documented.

The Kaggle description lists `Humerus Fracture` but not the bare `humerus` class, while the exported `data.yaml` contains both. This is a source-level inconsistency. It is not enough evidence to merge, delete, rename, or reverse either class automatically.

## 8. Canonical raw tree

The second local raw tree, `data/raw/bone-fracture-detection/BoneFractureYolo8`, contains the same 8,298 relative file paths as the versioned tree.

- 8,297 files are byte-identical.
- The only byte-different file is `README.dataset.txt`.
- All 4,148 images, all 4,148 labels, and `data.yaml` are byte-identical.

It is therefore safe to use only the versioned tree as the canonical analysis input while retaining both raw copies unchanged. This prevents accidental double-counting without discarding source data.

The full Phase 1 raw manifest remains 16,596 files with SHA-256 digest:

```text
7aef5a3f9c7a054dbd4d86ebb2fcbc9f5c19a828ebc80e24484a510258725ff9
```

## 9. What is resolved and what remains open

### Resolved

- The export contains offline rotation and exposure augmentation in `train`.
- The dominant training triplets are expected derivatives, not independent source X-rays.
- No exact or confirmed derivative leakage across splits was found.
- The three `humerus fracture` files represent one derivative family.
- Five empty-label files in four derivative groups have strong label-inconsistency evidence.
- The versioned YOLOv8 tree is the canonical analysis input; the second tree is a redundant raw copy.

### Still unresolved

- The precise intended distinction among `positive`, `fracture`, and bare `humerus`.
- Whether the remaining empty labels are verified negatives, missing annotations, or a mixture.
- Patient- and study-level independence of the current splits.
- The correct Phase 2 policy for the five confirmed inconsistent empty labels.
- The final handling of `humerus fracture` and `humerus`.
- The later polygon-to-box conversion policy for detection training.

## 10. Required decisions before Phase 2

1. Obtain a radiologist/supervisor review of the 20-image empty-label sample and separately flag the four known mixed-label derivative groups.
2. Seek source or supervisor clarification for the seven class names, especially the nested `humerus fracture` and `humerus` annotations.
3. Decide whether to request patient/study provenance from the dataset author or formally accept that patient-level split independence cannot be verified.
4. Define a traceable derived-data rule for confirmed label inconsistencies. Do not edit the raw labels.
5. Treat augmented variants as related observations in later dataset accounting and avoid presenting 3,631 training files as 3,631 independent source X-rays.
6. Keep the test split unused while these train/validation data-quality decisions are made.

These are preconditions and discussion points. No class remapping, split change, preprocessing, augmentation policy, training configuration, or Phase 2 implementation was performed in this follow-up.

## 11. Reproducible commands and artifacts

Install the pinned dependency ranges in `requirements.txt`, then run from the repository root with the package installed in editable mode or `PYTHONPATH=src`:

```powershell
python -m bone_fracture_audit.review
python -m bone_fracture_audit.provenance
python -m unittest discover -s tests -v
```

Generated artifacts are intentionally ignored by Git:

- `outputs/data_quality/radiologist_empty_label_review/`
- `outputs/data_quality/dataset_followup/provenance/pair_analysis.csv`
- `outputs/data_quality/dataset_followup/provenance/derivative_groups.csv`
- `outputs/data_quality/dataset_followup/provenance/summary.json`
- `outputs/data_quality/dataset_followup/provenance/cross_split_review/`
- `outputs/data_quality/dataset_followup/provenance/mixed_label_review/`

The raw dataset was read only. Phase 1 image, label, annotation, class, and empty-label statistics were not changed.

## 12. External sources checked

- [Roboflow Universe — bone fracture detection, version 4](https://universe.roboflow.com/veda/bone-fracture-detection-daoon/dataset/4): exact version counts, split sizes, preprocessing, augmentation multiplier, rotation, and exposure settings.
- [Roboflow documentation — Image Augmentation](https://docs.roboflow.com/datasets/dataset-versions/image-augmentation): confirms dataset-version augmentation is offline and applies to training images.
- [Roboflow documentation — Preprocess Images](https://docs.roboflow.com/datasets/image-preprocessing): distinguishes preprocessing across all splits from training-only augmentation.
- [Kaggle — Bone Fracture Detection: Computer Vision Project](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project): selected dataset description, named classes, author, and license context.

External pages were checked on 2026-09-14. Local generated evidence remains the reproducible basis for all dataset-specific conclusions in this report.
