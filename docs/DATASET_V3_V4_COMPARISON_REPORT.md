# Dataset v3-v4 Comparison Report

Generated from the local immutable exports and reproducible audit artifacts on 2026-09-15. No training, preprocessing, relabeling, split change, or raw-data modification was performed.

Evidence terms used below:

- **Verified fact:** directly counted, hashed, parsed, or decoded from local files.
- **Strong evidence:** supported by conservative image matching and/or consistent independent evidence, but not by patient/study identifiers.
- **Inference:** a reasoned interpretation of the verified evidence.
- **Unresolved:** not established by the available files.
- **Recommendation:** proposed for student review; not a recorded final project decision.

## 1. Purpose

This investigation asks whether Roboflow dataset version 3 (v3) or version 4 (v4) provides the cleaner source for Phase 2 and for the planned baseline/preprocessing/augmentation experiment matrix. It audits v3 independently, compares the versions directly, and preserves the test set from experimental use.

## 2. Dataset versions examined

| Version | Local reporting root | Role in this comparison |
|---|---|---|
| v3 | `data/raw/bone-fracture-detection/bone-fracture-detection-v3-yolov8` | Candidate unaugmented/base export |
| v4 | `data/raw/bone-fracture-detection/bone fracture detection.v4-v4.yolov8` | Previously audited augmented export |
| v4 duplicate | `data/raw/bone-fracture-detection/BoneFractureYolo8` | Redundant copy, not a third dataset |

**Verified fact:** the v3 local README identifies a 1,728-image v3 export and states that no image augmentation was applied. The [Roboflow v3 project page](https://universe.roboflow.com/veda/bone-fracture-detection-daoon) identifies the same six-class, 1,728-image v3 dataset. The [Roboflow v4 version page](https://universe.roboflow.com/veda/bone-fracture-detection-daoon/dataset/4) reports 4,148 images, three outputs per training example, rotation from -15° to +15°, exposure from -25% to +25%, and no preprocessing.

**Verified fact:** the two local v4 trees contain the same 8,298 relative paths; 8,297 are byte-identical and only `README.dataset.txt` differs. All images, labels, and `data.yaml` are byte-identical, so the duplicate was excluded from dataset counts and comparisons.

## 3. v3 independent audit results

| Split | Images | Labels | Matched pairs | Annotations | Empty labels | Empty rate |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,211 | 1,211 | 1,211 | 698 | 607 | 50.12% |
| Validation | 348 | 348 | 348 | 204 | 175 | 50.29% |
| Test | 169 | 169 | 169 | 96 | 86 | 50.89% |
| **Total** | **1,728** | **1,728** | **1,728** | **998** | **868** | **50.23%** |

**Verified fact:** every image has one matching label, there are no orphan labels, and all 1,728 `.jpg` files decode as JPEG. No malformed rows, unexpected class IDs, non-finite or out-of-range coordinates, degenerate polygons, duplicate annotation rows, or unreadable images were found.

**Verified fact:** all 998 annotations are YOLO segmentation polygons. The audit derived axis-aligned boxes for statistics only; it did not rewrite the raw labels. Widths range from 111 to 2,300 px, heights from 134 to 3,210 px, and 448 distinct dimension pairs occur. Derived box area has median 0.0195 and 95th percentile 0.0812 of image area; none crossed the audit's very-small or unusually-large review thresholds.

**Verified fact:** SHA-256 found no exact duplicate image group inside v3. The initial dHash screen produced 90 possible pairs, including 50 cross-split pairs, but these are candidates rather than confirmed duplicates.

## 4. v3 class structure

The actual v3 `data.yaml` declares six classes. Bare `humerus` does not exist.

| ID | Class | Train | Validation | Test | Total annotations | Images containing class |
|---:|---|---:|---:|---:|---:|---:|
| 0 | elbow positive | 113 | 29 | 17 | 159 | 143 |
| 1 | fingers positive | 178 | 48 | 27 | 253 | 208 |
| 2 | forearm fracture | 107 | 43 | 14 | 164 | 146 |
| 3 | humerus fracture | 104 | 36 | 15 | 155 | 145 |
| 4 | shoulder fracture | 120 | 20 | 17 | 157 | 139 |
| 5 | wrist positive | 76 | 28 | 6 | 110 | 79 |

**Verified fact:** v3 has 155 `humerus fracture` polygons across 145 images. It therefore does not contain v4's apparent three-image scarcity. The largest-to-smallest annotation ratio is 2.3:1, which is materially less extreme than the v4 interpretation in which `humerus fracture` has only three annotations.

**Inference:** v3 provides a more coherent six-class schema than v4 because it has one humerus-related label with substantial coverage. The clinical meaning of every class name still requires domain confirmation.

## 5. v3 empty-label analysis

**Verified fact:** v3 contains 607 train, 175 validation, and 86 test empty labels: 868 in total. The validation and test empty-label counts are the same as v4; v3 train has 1,220 fewer empty files because v4 has approximately three training outputs per v3 image.

**Unresolved:** an empty label proves only that the file contains no annotation row. It does not prove that the radiograph was reviewed and contains no visible fracture.

A reproducible sanity-check package was created at `outputs/data_quality/radiologist_empty_label_review/` with seed `20260915`: 16 train images, four validation images, and no test images. All 20 have distinct source keys and byte-identical source/copy SHA-256 values. Pairwise conservative matching found no exact, high-confidence, or probable derivative pair in the sample; the five borderline pairs were visually different radiographs. All 20 images were visually checked for a varied mix of anatomy and projections. This review package cannot validate all 868 empty labels.

## 6. v3 augmentation/provenance check

| Finding | v3 result |
|---|---:|
| Candidate pairs from repeated names/dHash | 581 |
| High-confidence derivative pairs | 0 |
| Probable derivative pairs | 0 |
| Inconclusive pairs | 19 |
| Rejected as unrelated | 562 |
| High-confidence derivative groups | 0 |
| Cross-split high-confidence groups | 0 |

**Verified fact:** the cautious ORB/affine analysis found no high-confidence or probable derivative pair within v3. All 19 inconclusive pairs were rendered and visually inspected; they show distinct radiographs or anatomical views rather than obvious rotation/exposure derivatives.

**Strong evidence:** the local v3 files are substantially closer to a source-level, non-offline-augmented export than v4. This agrees with the v3 README and contrasts with v4's 2,895 high-confidence and 348 probable derivative pairs across 1,107 high-confidence groups.

**Unresolved:** absence of detected derivative groups is not proof that no two images ever belong to the same patient or study. The export provides no reliable patient/study identifier.

## 7. v3-v4 validation/test comparison

| Split | Records in each version | Same filename | Byte-identical images | Byte-identical labels | Same geometry | Same class semantics |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 348 | 348 | 348 | 281 | 348 | 317 |
| Test | 169 | 168 | 168 | 133 | 169 | 155 |

**Verified fact:** every validation and test record maps across versions, and all annotation geometries match. The 31 validation and 14 test semantic differences are the v3 `humerus fracture` rows being renamed to v4 `humerus` (51 matched annotation rows total).

One test record, `distal-humerus-fracture-1_jpg`, has a different Roboflow suffix and different image bytes. Conservative alignment classifies the pair as a high-confidence derivative (correlation 0.9985, normalized residual RMSE 0.0119); its elbow annotation is geometrically and semantically equivalent within the documented 0.001 coordinate tolerance. All other held-out images are byte-identical.

**Conclusion:** validation and test are the same held-out content for practical experimental purposes, except for one re-exported/re-encoded test image and the systematic humerus class rename. They are not wholly byte-identical datasets.

## 8. v3-v4 training relationship

**Verified fact:** v4 train has 3,631 images, which is `3 × 1,211 - 2`. Both versions contain the same 1,002 source-style keys. For 1,000 keys, v4 contains exactly three records for every corresponding v3 record; two single-record v3 keys have only two v4 outputs.

Direct image evidence for the 3,631 v4 training records is:

| Best v3 relationship | v4 images | Share of v4 train |
|---|---:|---:|
| Exact byte copy | 1,210 | 33.32% |
| High-confidence derivative | 2,011 | 55.38% |
| Probable derivative | 186 | 5.12% |
| Inconclusive | 51 | 1.40% |
| Rejected by pixel matcher | 173 | 4.76% |

**Strong evidence:** 3,407 v4 images (93.83%) link directly to a v3 training image by exact bytes or conservative high/probable derivative matching. All 3,631 share a v3 source-style key, and the group-size pattern independently matches Roboflow's documented three-output rule except for two missing outputs. Generic source keys can collide, so the 224 inconclusive/rejected pixel matches are not counted as confirmed direct links.

**Inference:** v4 train is predominantly an offline-augmented export of v3 train, not a larger independent training cohort. Its nominal 3,631 images should not be interpreted as 3,631 independent source radiographs.

## 9. Class and annotation differences

| Class | v3 total | v4 total | Main relationship |
|---|---:|---:|---|
| elbow positive | 159 | 385 | v4 training expansion; held-out unchanged |
| fingers positive | 253 | 606 | v4 training expansion, three fewer than exact 3× train expectation |
| forearm fracture | 164 | 373 | v4 training expansion, five fewer than exact 3× train expectation |
| humerus fracture | 155 | 3 | v3 label was systematically remapped in v4; three new v4 rows remain |
| humerus | 0 | 362 | exists only in v4; corresponds to v3 `humerus fracture` geometry |
| shoulder fracture | 157 | 397 | exact 3× training expansion; held-out unchanged |
| wrist positive | 110 | 262 | exact 3× training expansion; held-out unchanged |

**Verified fact:** both versions use polygon rows and pass syntax/geometry validation. Among the 1,210 byte-identical v3-v4 training image pairs, 1,209 have identical annotation geometry, 949 have byte-identical label files, and 1,110 have identical class semantics. All 104 matched v3 training `humerus fracture` rows become v4 `humerus`; the held-out comparison shows the same 51-row remap.

The one exact-image geometry exception is source `image1_10_png`: v3 has one `humerus fracture` polygon; v4 preserves that geometry as `humerus` and adds a second, much larger `humerus fracture` polygon. That added row is repeated across all three v4 derivatives.

## 10. Empty-label relationship between versions

| Split | v3 empty | v4 empty | Relationship |
|---|---:|---:|---|
| Train | 607 | 1,827 | approximately 3×, plus five inconsistent derivatives |
| Validation | 175 | 175 | same records/status |
| Test | 86 | 86 | same records/status |
| **Total** | **868** | **2,088** | v4 total reflects training expansion |

**Verified fact:** five confidently mapped v4 derivatives are empty even though their v3 source images have one annotation. They occur in four source groups: `image1_244_png` (one empty derivative), `image2_821_png` (two), `image3_1128_png` (one), and `image3_278_png` (one).

**Strong evidence:** these five inconsistencies were introduced during the v4 generation/export path; the corresponding v3 sources are annotated. V4 therefore did not merely amplify pre-existing v3 empty labels in these four groups.

**Unresolved:** this does not establish the correctness of the v3 annotations or of the other empty labels; it only traces the specific v4 inconsistencies.

## 11. `humerus fracture` / `humerus` findings

**Verified fact:** bare `humerus` exists only in v4. Across all 155 comparable v3 humerus-fracture rows (104 on byte-identical train originals and 51 held-out rows), the same polygon geometry is labeled `humerus` in v4.

**Verified fact:** v4's three `humerus fracture` rows are three transformed versions of one source image, `image1_10_png`; each co-occurs with the remapped `humerus` polygon and covers about 42-45% of the image. V3 contains only the smaller original humerus-fracture polygon on that source image.

**Strong evidence:** v4 changed the class schema and introduced a second large annotation on one source group. The apparent three-example v4 `humerus fracture` class is therefore not evidence of a rare independent clinical class.

**Unresolved:** local files do not explain why the v3 label was renamed to anatomy-only `humerus`, why the three large rows were added, or which semantic naming policy is medically intended. No class should be removed or merged without student/supervisor/domain review.

## 12. Data-quality comparison

| Criterion | v3 | v4 | Methodological effect |
|---|---|---|---|
| Training size | 1,211 source-level records | 3,631 offline outputs | v4 count overstates independent evidence |
| Offline augmentation | none detected; none declared | rotation/exposure confirmed | v3 preserves augmentation as a controllable factor |
| Class schema | six classes; one humerus-fracture label | seven classes; systematic rename plus three added rows | v3 is clearer locally |
| Empty-label consistency | source labels include unresolved empties | five known derivative contradictions | v3 avoids the confirmed export inconsistency |
| Annotation syntax | valid polygons | valid polygons | equivalent syntactic quality |
| Exact/high-confidence cross-split leakage | none detected | none detected | neither proves patient independence |
| Patient/study independence | unavailable | unavailable | unresolved for both |
| Controlled experiments | suitable after open issues are resolved | augmentation is already baked into baseline | v3 is easier to interpret |

**Inference:** v3 is the cleaner methodological starting point, not because it has perfect labels, but because it avoids v4's baked-in augmentation and exposes a simpler class schema closer to the source records.

## 13. Implications for experimental design

The planned design is scientifically meaningful again if v3 is approved as the source dataset:

| Experiment | Custom preprocessing | Controlled augmentation | Interpretation with v3 |
|---|---|---|---|
| A | No | No | genuine non-offline-augmented baseline |
| B | Yes | No | isolates preprocessing |
| C | No | Yes | isolates project-controlled augmentation |
| D | Yes | Yes | measures their combination |

**Recommendation:** apply any future augmentation through explicit, recorded training configuration or a traceable derived dataset, with identical fixed splits and unrelated settings controlled. Do not use v4 as the “no augmentation” baseline.

Before A-D is valid, Phase 2 still needs decisions on polygon-to-box conversion, class semantics, the handling of empty labels and known inconsistencies, patient/study uncertainty, fixed seeds, and the exact augmentation policy. The test set must remain unused for these choices.

## 14. Advantages and disadvantages of using v3

Advantages:

- preserves a genuine no-offline-augmentation baseline;
- provides 1,211 training records closer to independent source images;
- has a simpler six-class schema and 155 humerus-fracture annotations rather than a misleading three-row class;
- avoids the five confirmed v4 derivative empty-label contradictions;
- makes augmentation an explicit experimental variable.

Disadvantages:

- has fewer nominal training files, so augmentation must be designed and recorded by the project;
- still has 868 unverified empty labels;
- still lacks patient/study identifiers and medical class definitions;
- remains polygon-based and therefore needs a traceable detection-label conversion decision.

## 15. Advantages and disadvantages of using v4

Advantages:

- supplies ready-made rotation/exposure variants and more nominal training files;
- retains the same held-out content, enabling continuity with earlier reported statistics;
- annotations are syntactically valid polygons.

Disadvantages:

- cannot support an honest “no augmentation” baseline because augmentation is already baked into train;
- its 3,631 train images are not 3,631 independent source radiographs;
- introduces five confirmed empty/annotated derivative contradictions;
- changes the humerus schema and introduces three large extra rows on one source group;
- makes later custom augmentation harder to isolate and interpret.

## 16. Recommended dataset version for Phase 2

**Recommendation:** use v3 as the raw source for the Phase 2 prepared dataset and controlled A-D experiments, while retaining v4 only as provenance evidence and, if useful later, a secondary comparison. Preserve the existing held-out membership, subject to the unresolved patient/study review.

This recommendation is based on the independently verified v3 audit, the direct version mapping, and the clearer experimental interpretation. It is not a final project decision. The student should approve or reject it before `DECISIONS.md` records a final Phase 2 dataset choice or any preparation begins.

## 17. Remaining uncertainties

- What do `positive`, `fracture`, and the v4 anatomy-only `humerus` label mean in the original annotation policy?
- Are empty labels reviewed no-fracture cases, incomplete annotations, or a mixture?
- Can patient/study identifiers be recovered to verify split independence?
- Why are two expected v4 training outputs absent?
- Why was one test image re-exported under a different suffix?
- Can the 224 v4 training images without a confident direct pixel match be resolved through original Roboflow provenance? Their group counts fit the documented export, but direct matching is insufficient.
- What correction policy should a future derived dataset apply to the five empty derivatives and the `image1_10_png` extra annotation?
- Should polygons be converted to boxes, and if so, what tested conversion and review protocol will be used?

## 18. Decisions still requiring student review

1. Approve or reject v3 as the Phase 2 source dataset.
2. Confirm the intended six-class or alternative class policy with the supervisor/domain reviewer.
3. Decide how radiologist feedback will affect empty labels without modifying raw data.
4. Decide whether existing split membership is acceptable given missing patient/study identifiers.
5. Approve a traceable polygon-to-box conversion and correction policy for derived data.
6. Define the controlled augmentation methods, ranges, seeds, and logging only after the above decisions.

Machine-readable evidence is separated under `outputs/data_quality/dataset_versions/v3/`, `outputs/data_quality/dataset_comparison/v3_vs_v4/`, and `outputs/data_quality/radiologist_empty_label_review/`. The key files are `audit_summary.json`, `summary.json`, `comparison_summary.json`, `held_out_file_comparison.csv`, `training_v3_to_v4_mapping.csv`, `training_exact_annotation_comparison.csv`, `class_comparison.csv`, and `review_index.csv`.
