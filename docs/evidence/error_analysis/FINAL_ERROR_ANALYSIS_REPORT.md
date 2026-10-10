# Final Phase 4 Error Analysis: Experiment D

Executed on 2026-10-10. This is a technical research record for later student-authored thesis writing. **Phase 4 analysis is complete, with the unresolved Stage 1 official-AP reproduction failure explicitly retained.** Experiment F and thesis Word edits have not started.

## 1. Objective and methodology

The objective was to characterize why the preserved Experiment D checkpoint detects few validation annotations, using fixed quantitative diagnostics and direct inspection of representative cases. Stage 3 reused the existing post-NMS predictions, annotation records, deterministic matching, and numerical results. It performed no new inference, training, tuning, relabeling, or held-out evaluation.

Sources are [Stage 2 report](STAGE2_REPORT.md), [quantitative evidence](stage2_quantitative.json), [Stage 2 verification](stage2_verification.json), [Stage 3 case evidence](stage3_qualitative.json), and [Stage 3 verification](stage3_verification.json). Exact numeric definitions remain in `stage2_quantitative.json: analysis.method`. Case-level visual observations, plausible interpretations, and unverified claims are separate fields in the Stage 3 evidence.

The diagnostic reference is confidence **>=0.25**, IoU **>=0.50**, six-class equality, and confidence-first greedy one-to-one matching. These TP/FP/FN measurements are separate from native Ultralytics precision/recall/AP. Changing the diagnostic IoU or score for descriptive sensitivity analysis does not change native AP or select a new deployment threshold.

## 2. Setup and verified provenance

Experiment D uses YOLOv8s, the six original classes, canonical CLAHE images, and the frozen conservative training augmentation condition. Its checkpoint is `outputs/training/official/D_seed42/weights/best.pt`, selected from epoch 57 by validation mAP50–95. Its SHA-256 remains `498e265f90bbf07fc5112b7153a7a11ac09ac819f8b7689e9e832ac302ef2e8a`.

The reused validation export is `outputs/error_analysis/stage1/D_seed42_repeat_verified/validation_predictions.jsonl`, SHA-256 `8fb1b0004ecf00ece249ff46c9bc12dc0de40d876489ecf13488400794178b22`. It contains 348 validation images, 204 annotations, 175 empty-label images, and 1,755 retained candidates. The export uses original-image pixel coordinates; FP32 fused inference, image size 640, rectangular batches of 16, strict score >0.001, class-aware multi-label NMS at IoU 0.70, and maximum 300 detections. No image hits that maximum. Predictions filtered before export are unavailable to this analysis.

The preserved training export SHA-256 is `c381b93442b894fdf27112082796ff578ca07472235f7cc6f11a40633eb01d5b`. Its existing Stage 2 inference covers 1,211 training images and 698 annotations using the same checkpoint and inference condition. Stage 3 hashes this export and reuses its reported statistics; it does not reopen training images or infer again. All 12 referenced Stage 2 numerical artifacts, Stage 1/2 evidence, validation image/label identities, and checkpoint provenance were reverified before use. Full paths and digests are in `stage3_qualitative.json: provenance`.

**Unresolved reproduction limitation:** official epoch-57 CSV mAP50/mAP50–95 are 0.14861/0.04732. Standalone checkpoint validation observed 0.1485241076082241/0.04736909409864185. The fixed absolute tolerance remains 0.00005. mAP50 fails (absolute difference 0.0000858924); mAP50–95 passes (0.0000490941). Passing Stage 3 consistency checks does not overturn the failed Stage 1 gate. Checkpoint serialization is a plausible residual explanation, not a proven resolution. The user's later authorization permits analysis with this limitation preserved.

## 3. Quantitative findings retained from Stage 2

At the diagnostic reference:

| Split | Images | GT | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Training | 1211 | 698 | 659 | 126 | 39 | 83.9% | 94.4% | 88.9% |
| Validation | 348 | 204 | 30 | 46 | 174 | 39.5% | 14.7% | 21.4% |

The **79.7 percentage-point training/validation recall gap** is the strongest evidence of poor generalization under the current protocol. High training recall across all six classes shows that the checkpoint can fit the training annotations. This does not distinguish memorization, image-domain differences, annotation differences, or other causes.

The 174 validation FN partition into **63 retained-candidate complete misses (36.2%)**, **48 low-confidence cases (27.6%)**, **3 localization failures (1.7%)**, and **60 mixed/unresolved cases (34.5%)**. No FN receives an exclusive primary classification or matching-conflict label. Nonexclusive strong signals remain: 56 low-confidence, 9 localization, 2 classification, 0 matching-conflict. Therefore, three primary localization cases do not mean only three annotations have localization-related evidence; zero primary classification cases do not mean no class confusion. These are candidate-based descriptions, not causal percentages or guaranteed recoverable counts.

At confidence 0.25, relaxing IoU from 0.50 to 0.10 raises recall only from 14.7% to 18.1%. At confidence 0.05 the same change raises recall from 24.5% to 35.3%. Geometry matters, particularly among weaker candidates, but looser matching does not eliminate the recall deficit. At IoU 0.50, lowering confidence from 0.25 to 0.05 changes TP/FP from 30/46 to 50/204; at 0.01 it gives 63/523. Precision falls from 39.5% to 19.7% and 10.8%, respectively. These tradeoffs do not support threshold lowering as a satisfactory solution.

Class-agnostic matching gives 32 TP, 44 FP, and 172 FN at the reference: recall 15.7%, versus 14.7% class-aware. Class confusion contributes, but removing equality adds only two net matches. This comparison retains the original class-aware NMS candidates and is neither a one-class retraining experiment nor class-agnostic AP.

Validation class differences remain substantial:

| Original class | GT | TP | FN | Recall |
| --- | --- | --- | --- | --- |
| elbow positive | 29 | 1 | 28 | 3.4% |
| fingers positive | 48 | 1 | 47 | 2.1% |
| forearm fracture | 43 | 12 | 31 | 27.9% |
| humerus fracture | 36 | 13 | 23 | 36.1% |
| shoulder fracture | 20 | 1 | 19 | 5.0% |
| wrist positive | 28 | 2 | 26 | 7.1% |

Forearm and humerus account for 25/30 TP from 79/204 annotations. Larger annotated regions are associated with higher recall:

| Nominal-640 GT short-side group | GT | TP | Recall |
| --- | --- | --- | --- |
| short_side_lt32 | 18 | 0 | 0.0% |
| short_side_32_to_lt64 | 83 | 5 | 6.0% |
| short_side_ge64 | 103 | 25 | 24.3% |

Size is the shorter side of the original GT box multiplied by `640 / max(original_width, original_height)`, not an extra inference resize. Class composition and size are confounded, many class-by-size cells have low support, and the association does not establish a resolution bottleneck. The candidate score/max-IoU Spearman associations are weak (0.140 any class; 0.165 correct class). They are clustered, descriptive associations, not confidence calibration or causal estimates.

Of 46 annotation-relative FP, 36 occur on annotated images and 10 on nine of 175 empty-label images (5.14%). Two FP are duplicate-like: a same-class box sufficiently overlaps an annotation already claimed by a matched same-class prediction. Mere overlap between arbitrary predictions does not satisfy this definition. Empty labels do not establish clinically healthy images.

## 4. Deterministic qualitative selection and limits

The method was saved before case selection in local `method_pre_selection.json`. Seed **42** drives SHA-256 pseudorandom ranking of category/sample/GT/prediction identities, independent of source ordering. Selection first includes all available primary localization cases and wrong-class geometric matches, then prefers unused images, unused target/prediction identities, and underrepresented original-class/GT-size strata within each category. It deliberately includes a duplicate-like FP and one stronger-class and one weaker-class successful comparison. No visual appearance influenced the selection.

All 36 case observations use **36 distinct validation images**, with no substitutions, shortfalls, duplicated case identities, or repeated images across categories. Complete-miss, mixed, and low-confidence subsets each cover all six source classes. There are 34 case observations with a GT or contextual annotation: 6 small, 11 medium, 17 large; the two empty-label FP have no GT-size group. For annotated FP, class/size describe the highest-IoU contextual annotation and do not imply a successful match. Disjoint ties use the stable GT ID; this is not Euclidean nearest-neighbor selection.

| Category | Available observations | Selected | Inspected | GT small/medium/large |
| --- | --- | --- | --- | --- |
| complete_miss | 63 | 8 | 8 | 2/3/3 |
| mixed_or_unresolved | 60 | 8 | 8 | 2/3/3 |
| low_confidence_detection | 48 | 6 | 6 | 1/2/3 |
| localization_failure | 3 | 3 | 3 | 0/0/3 |
| wrong_class_geometric_match | 3 | 3 | 3 | 0/0/3 |
| fp_annotated | 36 | 4 | 4 | 1/2/1 |
| fp_empty_label | 10 | 2 | 2 | 0/0/0 |
| successful_detection | 30 | 2 | 2 | 0/1/1 |

Every individual figure (1840×1320) contains a clean full image and a detailed focus crop. Green denotes GT; orange score >=0.25, blue 0.01–<0.25, purple >0.001–<0.01. The selected prediction is emphasized, including a low-score focus in the clean view when appropriate. At most six clean and eight detailed predictions are listed; off-crop candidates are identified, all source GT boxes are shown where visible, and exact original coordinates remain in the manifest. Dense detail labels can overlap; the accompanying tables and clean view resolve identities. Resizing is for display only, with subpixel rounding of the rendered image and no source-image modification.

All 36 individual figures and eight category sheets were directly inspected with the local image viewer. This is Codex visual review, not radiologist adjudication. The quota sample deliberately oversamples rare categories and useful comparisons. It cannot estimate the prevalence of device-like structures, marker responses, low contrast, or other visual conditions in the dataset. Category pools can overlap diagnostically even though selected images are distinct; the evidence retains primary FN categories, all nonexclusive flags, exact candidate combinations, and reference assignments.

## 5. Direct visual findings by category

**Complete misses — S3-01–08.** Small annotations occur in S3-03/06, including a thin region and crowded local contours. However, S3-01/05/07 have large GT regions. S3-07 contains many retained boxes clustered away from its large selected annotation. S3-02 shows a confident response around a bright peripheral marker rather than the annotated digit. S3-05/08 contain visually adjacent but disjoint responses. Thus “complete miss” means no retained candidate with GT IoU >=0.10; it does not necessarily mean no response nearby in ordinary spatial language. S3-04 receives a response at another annotation in the same image. Absence of sufficient nearby overlap occurs across all six classes, without revealing raw pre-NMS activity or a unique cause.

**Low-confidence cases — S3-17–22.** All six selected correct-class candidates have IoU >=0.50 but scores only 0.0025–0.0571. Some are reasonably aligned (S3-18/20); others extend below/right or only narrowly pass IoU (S3-17/22). In S3-19 a different, poorer-geometry candidate is close to the score boundary (0.2467), while the selected better-aligned candidate scores 0.0106. Bright device-like structures recur in S3-18/19/20, and broad unused canvas is visible in S3-17, but those conditions are not established causes. The representative sufficiently overlapping candidates are generally far below 0.25, and Stage 2 demonstrates the FP cost of accepting more weak responses.

**Mixed/unresolved — S3-09–16.** Weak displaced boxes occur around small targets in S3-09/10/11/12/14 and a larger target in S3-16. S3-13/15 have a very-low-score candidate covering the selected GT well, alongside a higher-score correct-class box with worse geometry. S3-11 also has a reference-score different-class box with poor overlap. Some unresolved cases have nearby weak poor boxes without any single strong taxonomy signal; others combine strong signals. These examples resist a single exclusive explanation and show why confidence, class, and placement must remain separately recorded.

**All three primary localization failures — S3-23–25.** S3-23's confident shoulder box is smaller and shifted toward the lower-left of a broad GT (score 0.7794, IoU 0.180). S3-24's humerus box covers the upper-left of a tall annotation but misses its lower extent (0.4781, IoU 0.227). S3-25's humerus box is too wide around a narrower annotation and falls just below the threshold (0.8122, IoU 0.472); better-overlap weak candidates use other class IDs. All three targets are in the >=64-pixel short-side group. Confident geometric failures therefore also occur at larger annotated scales.

**All three wrong-class geometric matches — S3-26–28.** The selected predictions overlap their annotations sufficiently but use a different stored class: wrist→forearm (IoU 0.726), fingers→forearm (0.846), wrist→fingers (0.551). These are source-taxonomy disagreements, not independent anatomical diagnoses. S3-28 also has a lower-score wrist prediction already matched class-aware (score 0.3052, IoU 0.629). The higher-score fingers prediction takes the annotation in class-agnostic matching. Hence three wrong-class matched pairs coexist with only two net additional GT matches; S3-28 is not a newly rescued FN.

**Annotated-image FP — S3-29–32.** S3-29/30 confidently predict spatially separate regions above the supplied annotation. S3-32 encloses a tiny annotation with a much larger box (IoU 0.084), showing that enclosure alone does not satisfy IoU matching. S3-31 is the deliberate duplicate-like example: the selected tight same-class box has IoU 0.864, but a higher-score same-class prediction already claims that GT. This assignment-based FP is distinguishable from a purely misplaced or arbitrary overlapping box.

**Empty-label FP — S3-33/34.** Confident shoulder/humerus boxes occur at visible contour junctions, with weaker candidates in similar regions. There is no annotation against which to establish clinical truth. Normality, annotation incompleteness, and genuine unannotated findings remain unverified alternatives; none is counted as a clinical finding.

**Successful comparisons — S3-35/36.** The shoulder true positive has visible plate-like structures despite a box extending beyond the GT (score 0.5000, IoU 0.567). The forearm true positive closely encloses a medium-size target (0.3364, IoU 0.794). Device-like appearance can coexist with both success and failure, and success is not confined to the largest group. Two selected successes do not establish causal advantages or clinical reliability.

## 6. Integrated interpretation

**Strongest supported explanation at the behavioral level:** the checkpoint fits training annotations much better than validation annotations. The direct review illustrates the resulting failure through absent retained overlap, severely weak aligned candidates, misplaced confident boxes, and interacting uncertain responses. It does not identify the underlying cause of the train/validation gap.

**Relative contributions:** missed detections dominate the reference outcome (174 FN for 204 GT). Retained misses and mixed/unresolved cases together account for 123/174 FN, while 48 have an exclusive low-confidence signal. Localization also appears nonexclusively and visually; class confusion exists but label removal has a modest net effect. These groups describe exported candidates and assignment rules; their sizes are not separable causal contributions or predicted gains from a repair.

**Scale and class heterogeneity:** lower recall for small annotations and substantial class differences are measured associations. Small targets, cluttered contours, tilted or narrow framing, broad blank margins, bright markers, surrounding texture, and device-like components are visible in selected failures. Large misses and localization failures, a small target with a sufficient low-score box, and a successful device-containing shoulder case qualify any universal size- or appearance-based story. The evidence does not prove that insufficient image resolution, class naming, device appearance, CLAHE, augmentation, or architecture is the dominant cause. Low contrast remains an unquantified hypothesis; no comparative contrast measure or controlled visual rating was added.

**Supported versus hypothetical:** the gap, class/size associations, threshold tradeoffs, exact FP/FN assignments, and stated visible patterns are supported. Overfitting, distribution shift, annotation geometry/incompleteness, semantic inconsistency, and appearance-dependent failure are plausible competing explanations. Clinical label correctness, population generalization, and patient independence are unresolved. The qualitative sample adds concrete examples and counterexamples; it does not overturn Stage 2 counts or establish new population rates.

## 7. Limitations and threats to validity

This is one checkpoint and seed with a validation-selected best epoch. Training inference describes fitting, not an independent performance estimate. Validation supports are small, classes and box sizes are intertwined, source polygons were converted to enclosing boxes, and unknown patient/study relationships limit independence claims. Filename/hash separation does not rule out shared patients or near-duplicates. Empty labels and source-class semantics remain unadjudicated.

The export is post-NMS, score-filtered, and capped; raw detector-head activity is unavailable. Candidate-based error categories depend on fixed operational thresholds and matching order. Clean/detail displays cap candidates and crop context; exact retained records remain available. Visual review is a single nonclinical Codex review without expert agreement estimates, controlled contrast measurement, or a train-versus-validation visual-domain survey. No causal hypothesis was tested by an intervention. The official-AP discrepancy remains unresolved at its original tolerance.

Medical-image redistribution permissions have not been verified or approved. **All X-rays, crops, overlays, repeated renders, and contact sheets remain under ignored local outputs.** This report publishes text and numerical identities only. Clinical safety, diagnostic validity, and replacement of medical professionals are not claimed.

## 8. Implications for Experiment F

The user's current definition of Experiment F is a **controlled alternative-detector comparison**. This supersedes the earlier Stage 2 report's provisional augmentation/regularization suggestion; the historical Stage 2 evidence and report remain unchanged. No alternative model, configuration, implementation, or training is selected or created here.

The comparison should investigate whether a different detector changes validation generalization, rather than assuming that it will. A later approved protocol should preserve canonical data, split membership, six-class targets, annotation conversion, and the D comparison condition, while predeclaring the architecture change, initialization, comparable compute/training budget, checkpoint selection, random seeds, and necessary architecture-specific differences. Those controls and feasibility choices require later design; they are not fixed by this report. Keep validation as the iterative decision source and reserve test evaluation for the final frozen setup.

Relevant outcomes are native validation AP reported separately from the same diagnostic TP/FP/FN reference, the training/validation gap, per-class and size-stratified behavior with support counts, confidence/FP tradeoffs, and error-category changes. Track whether missed overlap and weak aligned candidates improve without simply increasing FP or trading away stronger classes. Reuse the selected case IDs as illustrations, not as a tuning subset or a new independent benchmark. Any later reproduction mismatch should be reported rather than hidden. A different architecture may improve, preserve, or worsen the observed behavior; current evidence does not predict a guaranteed gain.

## 9. Conclusions, verification, and local reproduction

Phase 4 now has executed quantitative diagnostics and direct qualitative inspection of 36 traceable cases. The principal conclusion is a substantial training/validation generalization gap, accompanied by missed retained overlap, low-confidence aligned predictions, mixed geometry/class/score behavior, and uneven class/scale results. The visual findings support that multifactor description while challenging simple universal explanations based only on small regions, devices, or class confusion. They justify investigating a controlled alternative detector, without identifying a proven remedy.

Stage 3 verification checks every selected identity, class, score, coordinate, IoU, size, matching flag, crop and displayed-box transform against deterministic reselection from the verified sources. It confirms all quotas, 36 unique images, all three localization and wrong-class cases, recorded overlap handling, 36 byte-identical repeat overlays and eight byte-identical repeat sheets. All 187 protected files retain their pre-Stage-3 hashes, including A–E artifacts, configurations, dependency files, Stage 1/2 evidence/exports/implementation, and the three unrelated Phase 2F edits. Guarded revalidation and rendering recorded zero held-out test accesses. No training, new inference, model modification, dependency changes, raw/canonical data changes, or Word-document edits occurred. Focused and full regression results are recorded in [stage3_verification.json](stage3_verification.json).

The authoritative local package is `outputs/error_analysis/stage3/D_seed42_verified/`: `visual_manifest.json`, `selection_manifest.json`, `method_pre_selection.json`, completed `review_forms.json`, `individual/S3-01.png` through `S3-36.png`, and eight `contact_sheets/*.png`. Inspection notes are `outputs/error_analysis/stage3/review_notes.json`. The public evidence records each local figure's digest, full sample ID, exact target/prediction IDs and boxes, class/score/IoU, source hashes, candidate flags, and separate observations/interpretations/unknowns.

From the repository root, using existing local data and preserved exports, render a fresh package:

```powershell
.venv/Scripts/python.exe -B -m bone_fracture_pipeline.error_analysis_visual `
  --output outputs/error_analysis/stage3/D_seed42_new_review
```

This requires the preserved Stage 3 protection baseline, rejects existing output directories, and creates pending review forms. Inspect the images and fill explicit case and contact-sheet records before finalizing; the finalizer verifies overlay hashes, notes, source identities, and geometry and does not generate observations automatically:

```powershell
.venv/Scripts/python.exe -B -m bone_fracture_pipeline.error_analysis_visual `
  --output outputs/error_analysis/stage3/D_seed42_new_review `
  --finalize --reviews outputs/error_analysis/stage3/new_review_notes.json
```

Finalization writes text-only `stage3_qualitative.json` and updates the local manifest/forms. It does not regenerate this interpretive report or its separate verification record; refresh those explicitly for any subsequent analysis. Phase 4 completion means the authorized analysis is finished, not that Stage 1 numerical equivalence or clinical validity has been established. Experiment F and thesis-document work await separate authorization.

## Appendix: selected case identifiers and focus measurements

GT sizes use nominal-640 shorter-side pixels. A contextual GT for an annotated-image FP is not a matched target. The focus candidate in a miss or mixed case is an illustrative retained candidate, not necessarily a valid detection. Exact values and full per-case observation notes are in `stage3_qualitative.json`; rounded values below are for reading.

| Case | Validation sample ID | GT ID | Prediction ID | GT/context class (predicted if empty) | Prediction class | Score | IoU | GT short side |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S3-01 | valid/image1_2358_png.rf.84f9ec7307749d01f6f471fa0de652b7 | gt/0001 | none | elbow positive | none | none | n/a | 73.0 |
| S3-02 | valid/image1_1332_png.rf.9a84efb79d01516ea4ca07f134749fdc | gt/0001 | pred/000000 | fingers positive | humerus fracture | 0.3611 | 0.000 | 34.9 |
| S3-03 | valid/image1_1090_png.rf.de645f822a5e36175c5e988223f4eeb0 | gt/0001 | pred/000000 | wrist positive | shoulder fracture | 0.0061 | 0.000 | 30.4 |
| S3-04 | valid/image1_1607_png.rf.ecf3e995bfb4045c5ccbbd914a0fab8c | gt/0002 | pred/000000 | forearm fracture | forearm fracture | 0.0904 | 0.000 | 51.7 |
| S3-05 | valid/image1_1669_png.rf.66de9f8a4d5a71ec06d4c5c4c475ff6c | gt/0001 | pred/000000 | elbow positive | elbow positive | 0.0157 | 0.000 | 90.1 |
| S3-06 | valid/image1_111_png.rf.3893d8f7588cea4d796d26119e52637f | gt/0002 | pred/000000 | wrist positive | forearm fracture | 0.0010 | 0.000 | 24.9 |
| S3-07 | valid/image2_632_png.rf.65725898246d01bd76a6610f79591127 | gt/0001 | pred/000012 | humerus fracture | shoulder fracture | 0.0011 | 0.074 | 143.9 |
| S3-08 | valid/image1_4311_png.rf.94bdbf6b601e08d736b5fca3590c95c1 | gt/0001 | pred/000001 | shoulder fracture | shoulder fracture | 0.0139 | 0.000 | 50.0 |
| S3-09 | valid/image1_1793_png.rf.27e45bd8854ac26a87f4170628b848cd | gt/0002 | pred/000005 | fingers positive | fingers positive | 0.0049 | 0.206 | 30.6 |
| S3-10 | valid/image1_1309_png.rf.792e90812ae6932e3459f8df5b9167e9 | gt/0001 | pred/000003 | elbow positive | elbow positive | 0.0222 | 0.256 | 34.5 |
| S3-11 | valid/image2_1722_png.rf.450ed72977e9af3f01fb362ea0f7d01e | gt/0001 | pred/000014 | forearm fracture | forearm fracture | 0.0028 | 0.142 | 28.2 |
| S3-12 | valid/image2_142_png.rf.d563056b013def35087f83ce9f0119be | gt/0001 | pred/000002 | forearm fracture | forearm fracture | 0.0017 | 0.117 | 42.9 |
| S3-13 | valid/image2_1224_png.rf.7c20f6876b3efc39ea916cda7db4c00c | gt/0002 | pred/000014 | humerus fracture | humerus fracture | 0.0023 | 0.672 | 138.0 |
| S3-14 | valid/image1_1992_png.rf.379b87658e6c985d51218b5383d5450b | gt/0003 | pred/000002 | wrist positive | elbow positive | 0.0060 | 0.319 | 42.9 |
| S3-15 | valid/image1_8373_png.rf.e4c1577e54e74dc96a3989672dd54301 | gt/0001 | pred/000007 | shoulder fracture | shoulder fracture | 0.0014 | 0.760 | 95.1 |
| S3-16 | valid/image1_2476_png.rf.8c979b58ab2997484f0de77252f4bae8 | gt/0001 | pred/000001 | elbow positive | elbow positive | 0.0056 | 0.114 | 70.0 |
| S3-17 | valid/image2_819_png.rf.8ebb71ef2e3e83019dc323683c56af2b | gt/0001 | pred/000000 | forearm fracture | forearm fracture | 0.0118 | 0.517 | 40.0 |
| S3-18 | valid/image1_1095_png.rf.528793dae32e5d8d6aca1d9bbc7a4511 | gt/0001 | pred/000001 | humerus fracture | humerus fracture | 0.0571 | 0.643 | 129.4 |
| S3-19 | valid/image1_2291_png.rf.62cfeef40c65e7f1fe873335c535e13d | gt/0001 | pred/000004 | wrist positive | wrist positive | 0.0106 | 0.683 | 51.8 |
| S3-20 | valid/image2_56_png.rf.07ffbc9bafd21d80db9c78b4f935ba3a | gt/0001 | pred/000000 | elbow positive | elbow positive | 0.0491 | 0.773 | 66.1 |
| S3-21 | valid/image1_1223_png.rf.0a498c2e88c243ab32fbec80233b5e72 | gt/0002 | pred/000001 | fingers positive | fingers positive | 0.0031 | 0.544 | 28.0 |
| S3-22 | valid/image1_7615_png.rf.304bc2c5cf0a941b6846f0e132be5c3d | gt/0001 | pred/000004 | shoulder fracture | shoulder fracture | 0.0025 | 0.501 | 134.6 |
| S3-23 | valid/image2_342_png.rf.c1998dcfe68bc1bab84ea57af9694c51 | gt/0001 | pred/000000 | shoulder fracture | shoulder fracture | 0.7794 | 0.180 | 150.0 |
| S3-24 | valid/image1_104_png.rf.86a9d1eeedeec79216455a5b9be63e17 | gt/0001 | pred/000000 | humerus fracture | humerus fracture | 0.4781 | 0.227 | 64.4 |
| S3-25 | valid/image1_1186_png.rf.3cfc6477bbc733ebdf562bb8d455e907 | gt/0001 | pred/000000 | humerus fracture | humerus fracture | 0.8122 | 0.472 | 74.4 |
| S3-26 | valid/image1_2027_png.rf.d64c9d07c29a2c6b7b76b6df9f8933e5 | gt/0001 | pred/000000 | wrist positive | forearm fracture | 0.7978 | 0.726 | 72.6 |
| S3-27 | valid/image1_2152_png.rf.5c2e8ea5ec087257e61a7026e9994011 | gt/0001 | pred/000000 | fingers positive | forearm fracture | 0.7280 | 0.846 | 88.0 |
| S3-28 | valid/image1_1900_png.rf.b5bcead0522f3d8b2bee79cc15a5477f | gt/0001 | pred/000000 | wrist positive | fingers positive | 0.3857 | 0.551 | 67.4 |
| S3-29 | valid/image1_4344_png.rf.6c8260143f2213e8f88f00e2db33f712 | gt/0001 | pred/000000 | fingers positive | fingers positive | 0.3357 | 0.000 | 51.1 |
| S3-30 | valid/image2_1776_png.rf.a86ba5c63c5c70eb7718401a9e15e50e | gt/0001 | pred/000000 | forearm fracture | forearm fracture | 0.3903 | 0.000 | 34.4 |
| S3-31 | valid/image2_1259_png.rf.ebfab9ae92058e7cca0b9e0fbb64c948 | gt/0001 | pred/000003 | humerus fracture | humerus fracture | 0.3636 | 0.864 | 106.0 |
| S3-32 | valid/image1_3789_png.rf.ecb45f4073420238817c80bb9dfb4eac | gt/0001 | pred/000000 | elbow positive | elbow positive | 0.3373 | 0.084 | 27.6 |
| S3-33 | valid/image1_1514_png.rf.8aa5c1d4670140b9b9b8b84d6ad70fbc | none | pred/000000 | shoulder fracture | shoulder fracture | 0.8382 | n/a | n/a |
| S3-34 | valid/image1_96_png.rf.84e1d4c27219711b3bab09d6f54745b0 | none | pred/000000 | humerus fracture | humerus fracture | 0.3245 | n/a | n/a |
| S3-35 | valid/image1_5969_png.rf.92335c3790abb78579bfc9aaa2cbd87c | gt/0001 | pred/000000 | shoulder fracture | shoulder fracture | 0.5000 | 0.567 | 84.7 |
| S3-36 | valid/image1_811_png.rf.754f721fa3c67e3b94dfbc54d4611a83 | gt/0001 | pred/000000 | forearm fracture | forearm fracture | 0.3364 | 0.794 | 63.5 |

Case ranges map to the categories in Section 5. Local figures use the same case IDs.
