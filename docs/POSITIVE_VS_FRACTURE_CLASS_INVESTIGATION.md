# Positive vs fracture class investigation

**Date:** 2026-09-22

**Scope:** Canonical Roboflow v3 source and the existing six-class detection dataset

**Status:** Evidence review for a possible future Experiment E; no class remapping or experiment change made

## Investigation objective

Determine whether `elbow positive`, `fingers positive`, and `wrist positive` refer to annotated fracture regions in the same broad sense as `forearm fracture`, `humerus fracture`, and `shoulder fracture`. The question is whether mapping all six IDs to one `fracture` ID would be a defensible **class-agnostic fracture-localization** experiment. This review distinguishes the dataset author's stated task from what local labels and a small visual sample can verify. It does not validate clinical diagnoses.

## Evidence examined and limits

| Evidence | What it establishes | Limit |
|---|---|---|
| Raw v3 `data.yaml` and `README.roboflow.txt` under `data/raw/bone-fracture-detection/bone-fracture-detection-v3-yolov8/` | Exact six names/IDs; Roboflow workspace `veda`, project `bone-fracture-detection-daoon`, version 3; 1,728-image export | Neither file defines `positive` or an annotation policy; the README says “Bones are annotated,” which is imprecise |
| [Kaggle listing by P. K. Darabi](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project), checked 2026-09-22 | The identified dataset listing calls **all six named classes** types of bone fracture and describes boxes or masks as indicating detected fracture location and extent | Source-level narrative, not a per-image annotation manual or proof that every label is correct |
| [Original Roboflow project](https://universe.roboflow.com/veda/bone-fracture-detection-daoon), checked 2026-09-22, and v3 URL embedded in `data.yaml` | Same six labels and 1,728-image project identity | Public project page does not publish a definition of `positive`; the exact v3 version page was not accessible in this review |
| `docs/DATASET_V3_V4_COMPARISON_REPORT.md`, `docs/DATASET_AUDIT_REPORT.md`, `docs/DATASET_FOLLOWUP_REPORT.md`, `docs/PHASE_2_PLAN.md`, `docs/DECISIONS.md` | Dataset provenance, class counts, conversion decisions, and previously recorded semantic uncertainty | Prior technical audits did not perform a clinical label adjudication |
| Raw v3 train/validation label rows and `outputs/phase2/phase2b/v3_detection/visual_review_index.csv` with six train overlay images | Actual class IDs, polygon structure, and representative spatial placement for both name groups | Six visual examples are a small, non-random sample; no radiologist reviewed these six labels for this investigation |
| `docs/evidence/phase2a/preparation_summary.json` and `docs/PHASE_2B_VALIDATION_REPORT.md` | All 998 v3 polygons were converted one-to-one to minimum enclosing boxes while preserving IDs and splits | Geometry validation does not establish pathology semantics |

Only existing train and validation labels and train review overlays were newly inspected. No test image or label was opened, and no training or evaluation was run. Historical dataset totals quoted below come from the established project evidence.

## Local class and annotation findings

The canonical v3 `data.yaml` declares these six IDs. The approved Phase 2A evidence records 998 polygons in total; its one-to-one conversion gives the same count of detection boxes. The counts below are historical dataset counts, not a new test-split analysis.

| ID | Source class | Polygon annotations | Name group |
|---:|---|---:|---|
| 0 | `elbow positive` | 159 | positive |
| 1 | `fingers positive` | 253 | positive |
| 2 | `forearm fracture` | 164 | fracture |
| 3 | `humerus fracture` | 155 | fracture |
| 4 | `shoulder fracture` | 157 | fracture |
| 5 | `wrist positive` | 110 | positive |

### `positive` classes

The three `positive` IDs have polygon annotations in the same v3 YOLO segmentation files as the three `fracture` IDs. In train and validation, all inspected rows used a class ID followed by coordinate pairs; none carried a separate attribute for a different disease or a different label type. Their prepared detection labels are derived by the same one-polygon-to-one-box rule as the `fracture` classes.

The checked train overlays for `elbow positive` (`image1_1427_png`), `fingers positive` (`image1_2978_png`), and `wrist positive` (`image1_134_png`) show focal regions marked inside radiographs, rather than a whole-image yes/no label. The fingers and wrist examples have multiple localized polygons. Four non-generic train/validation filenames attached to `elbow positive` rows contain `fracture` or `fractures`, including `coronoid-process-fracture_jpg` and `fracture-of-the-humeral-capitellum-milch-type-1-1-1-_jpg`. One also mentions dislocation **with** a coronoid-process fracture. These filename clues support a fracture interpretation for that class; filenames are not diagnoses or a complete class definition. No comparable descriptive filenames were found for `fingers positive` or `wrist positive` in the inspected train/validation rows; most names are generic `image1_`, `image2_`, or `image3_` identifiers.

The Kaggle source description is the direct semantic evidence: it includes the three `positive` names in its list of fracture classes and describes the annotations as fracture locations. Therefore, in the source author's intended task, `positive` most plausibly means fracture-positive for the named anatomical group. This remains an interpretation of the author's description, not independent medical confirmation of every polygon.

### `fracture` classes

The three names explicitly contain `fracture`. Checked train overlays for `forearm fracture` (`image1_220_png`), `humerus fracture` (`image1_10_png`), and `shoulder fracture` (`image1_1399_png`) likewise mark localized regions with polygons. Their raw row structure and Phase 2A conversion are the same as for the `positive` group. The word `fracture` in a class name and the Kaggle description support the intended target, but neither proves the accuracy or completeness of each underlying annotation.

V3 has one `humerus fracture` ID and no bare `humerus` class. The v3/v4 comparison documented that v4 relabeled the comparable v3 humerus-fracture geometry as bare `humerus` and added three large `humerus fracture` rows on one generated source group. This is concrete evidence that the later export's taxonomy can drift; it is **not** evidence that v3's `positive` IDs encode a separate pathology. The v4 export is not the canonical experiment source.

## Comparison of annotation semantics

At the **declared task level**, the Kaggle author groups all six labels as fracture categories and describes their boxes/masks as localizing detected fractures. This is the strongest affirmative evidence for a broad common target. The Roboflow page confirms the six labels but adds no finer semantic definition.

At the **file and geometry level**, both name groups are class IDs on v3 segmentation polygons, converted identically into detection boxes. The inspected overlays from both groups show focal spatial regions. No local metadata, class mapping, inspected row, or source description identifies `positive` as a different pathology, image-level classification target, or background class. A lack of such evidence cannot prove that no mislabeled or non-fracture regions exist. The visual sample cannot determine whether the regions are medically equivalent, whether boundaries were drawn to the same clinical standard, or whether every relevant fracture was annotated.

The naming split appears more likely to reflect inconsistent anatomical class naming than a deliberately different task: the source author explicitly describes all six names as fracture classes, while the export provides one shared annotation format and no distinction rule. This is an **inference**, not a documented source-taxonomy decision. The v4 humerus rename independently shows that labels were not perfectly stable between export versions, but it does not explain why the three v3 names use `positive`.

## Evidence against or unresolved for a merge

- There is no original labeling manual, patient-level ground truth, or explicit source statement that defines `positive` word-for-word as “fracture-positive.” The Kaggle narrative supports that reading but does not certify individual labels.
- No radiologist has adjudicated the six inspected positive/fracture overlays as part of this investigation. A focal polygon can be visually plausible without being a correct fracture annotation.
- The dataset's anatomical labels may themselves be imperfect: descriptive filenames are sparse and cannot settle the exact bone or injury for generic image names. A one-class merge would remove anatomical distinctions from the model output, but it would not repair such annotation errors.
- Empty labels remain only partially checked. The limited 20-image radiologist review documented in `docs/PHASE_2_PLAN.md` supports treating them as intended background examples for the working dataset; it does not establish that all 868 are fracture-free.
- Patient/study independence and subtle duplicate/leakage risks remain unresolved in the existing audits. A class merge does not resolve them.
- All six labels describe **source-annotated suspected fracture regions**, not a clinically verified universal fracture class. Do not interpret a merged class as evidence of diagnostic validity.

No examined source explicitly contradicts the broad fracture interpretation of the three `positive` classes. The limitations above concern proof strength, annotation quality, and generalizability.

## Implications for a future Experiment E

**Conclusion:** A single `fracture` target is scientifically defensible for an explicitly framed, exploratory **class-agnostic localization** experiment on this dataset. The justification is the identified dataset author's treatment of all six names as fracture categories, supported by the common local polygon/box annotation mechanism and focal train-image examples. Semantic equivalence is supported at the intended task level; per-instance clinical equivalence and label accuracy are **not proven**.

If Experiment E is later approved, its methodology should state the exact six-to-one mapping as a new, traceable derived-label decision; preserve image identities, splits, empty-label files, and box geometry; and retain the original six-class datasets and A/B/C/D run records unchanged. The one-class result would answer a different output-label question, so its metrics should not be presented as a direct like-for-like improvement over six-class mAP without explaining the changed evaluation task. The test split remains reserved for final evaluation. The next step is to design and validate E separately; this report does not implement it or change the frozen historical A/B/C/D matrix.
