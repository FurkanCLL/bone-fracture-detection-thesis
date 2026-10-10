"""Build the compact Stage 2 report directly from its numeric evidence."""

from __future__ import annotations

from pathlib import Path

from bone_fracture_pipeline.error_analysis_quantitative import CATEGORIES, validate_numerical_tree


LABELS = {"complete_miss": "Complete miss (retained scope)", "localization_failure": "Localization failure",
          "low_confidence_detection": "Low-confidence detection", "classification_error": "Classification error",
          "matching_conflict": "Matching conflict", "mixed_or_unresolved": "Mixed or unresolved"}


def _number(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _rate(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _table(headers, rows):
    # A single formatter keeps generated counts and denominators aligned across sections.
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows), ""]


def _metrics(row):
    return [row["tp"], row["fp"], row["fn"], _rate(row["precision"]), _rate(row["recall"]), _rate(row["f1"])]


# Use the recorded measurements, rather than hand-copied values that can drift after a rerun.
def write_report(record: dict[str, object], path: Path) -> None:
    result = record["analysis"]
    if set(result["split_counts"]) != {"train", "valid"}:
        raise ValueError("The report accepts train/validation diagnostics only.")
    validate_numerical_tree(result)
    a = result["analysis_a_iou_sensitivity"]
    b = result["analysis_b_confidence_sensitivity"]
    c = result["analysis_c_fn_taxonomy"]
    d = result["analysis_d_classes"]
    e = result["analysis_e_false_positives"]
    f = result["analysis_f_generalization"]
    validation, training = result["validation_reference"], result["training_reference"]
    quality = b["candidate_localization"]
    checkpoint, train_export = record["checkpoint"], record["training_export"]
    comparison = record["stage1_official_reproduction"]["comparison"]
    settings = train_export["settings"]
    framework = train_export["framework"]
    aware_confidence = {row["confidence"]: row for row in b["operating_points"] if row["class_aware"]}
    agnostic_confidence = {row["confidence"]: row for row in b["operating_points"] if not row["class_aware"]}
    aware_iou = {(row["confidence"], row["iou"]): row for row in a if row["class_aware"]}
    low, reference = aware_confidence[0.01], aware_confidence[0.25]
    forearm, humerus = d["class_rows"][2:4]
    strong_tp = forearm["tp"] + humerus["tp"]
    strong_support = forearm["support"] + humerus["support"]
    other_recalls = [row["recall"] for row in d["class_rows"] if row["class_id"] not in (2, 3)]
    confusion = d["class_confusion"]
    empty = e["groups"]["empty_label"]
    annotated = e["groups"]["annotated"]
    lines = ["# Experiment D: Phase 4, Stage 2 quantitative error analysis", "",
             "This is an executed research diagnostic report, not final thesis prose or clinical validation. "
             "Stage 3 and Experiment F have not started.", "",
             "## Evidence and methods", "",
             "The numerical source is [stage2_quantitative.json](stage2_quantitative.json), with exact definitions in "
             "`analysis.method`. [stage2_verification.json](stage2_verification.json) records regression tests, "
             "independent count checks, repeatability, figure inspection, and preservation checks. "
             "Counts are exact; displayed percentages and descriptive statistics are rounded.", "",
             f"Checkpoint: `{checkpoint['path']}` (YOLOv8s, six original classes, seed 42, best epoch "
             f"{checkpoint['best_epoch_from_csv']}); SHA-256 `{checkpoint['sha256']}`. "
             "Canonical source: `data/prepared/v3_detection_clahe`.", "",
             f"Validation export: `{record['validation_export']['path']}`; SHA-256 "
             f"`{record['validation_export']['sha256']}`. Training export: `{train_export['path']}`; "
             f"SHA-256 `{train_export['sha256']}`. Both complete exports remain under ignored `outputs/`.", ""]
    lines += _table(["Split", "Images", "Annotated images", "Empty-label images", "GT boxes", "Exported predictions"],
                    [[split, counts["images"], counts["images"] - counts["empty_label_images"], counts["empty_label_images"],
                      counts["annotations"], counts["predictions"]] for split, counts in result["split_counts"].items()])
    lines += [f"Inference: image size {settings['imgsz']}, rectangular batches of {settings['batch']}, fused model, "
              f"{framework['precision']}, AMP disabled, export confidence >{settings['conf']}, class-aware multi-label NMS "
              f"at IoU {settings['iou']}, max_det={settings['max_det']}, no class filter or merged classes, workers=0. "
              "Training augmentation and gradient tracking were disabled. The guarded training snapshot aliases only "
              "training images to the framework's evaluation loader; its YAML has no test key. "
              "The loader recorded 1,211 training, zero validation, and zero test images.", "",
              f"Runtime: Ultralytics {framework['ultralytics']}, PyTorch {framework['torch']}, "
              f"{framework['gpu']}; deterministic seed {framework['seed']}. "
              "Train and validation use the same inference options apart from output paths.", "",
              "Matching reuses Stage 1's confidence-first greedy one-to-one matcher. Predictions are processed by "
              "descending score; score ties use stable prediction IDs, overlap ties use stable GT IDs. "
              "Each prediction claims its highest-IoU eligible unmatched GT. Confidence and matching IoU thresholds "
              "are inclusive. Class-agnostic matching removes class equality only. TP+FN equals GT support and "
              "TP+FP equals retained predictions. Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). "
              "Zero denominators follow Stage 1's zero convention; absent descriptive statistics are null. "
              "These are fixed-threshold diagnostics, not native Ultralytics AP.", "",
              "Reference condition: confidence >=0.25, IoU >=0.50, class-aware. Complete-miss and FN candidate "
              "analyses include all exported candidates above 0.001; NMS-suppressed or lower-scored responses are unavailable.", "",
              "### Preserved Stage 1 reproduction discrepancy", ""]
    lines += _table(["Native metric", "Official epoch 57", "Standalone checkpoint", "Absolute difference", "Original check"],
                    [[name, _number(values["official"], 5), _number(values["observed"], 8),
                      _number(values["absolute_difference"], 8), "Pass" if values["passed"] else "FAIL"]
                     for name, values in comparison["metrics"].items()])
    lines += [f"The original absolute tolerance remains {comparison['absolute_tolerance']:.5f}; overall reproduction "
              "remains **failed**. The user explicitly authorized Stage 2 despite this discrepancy. "
              "Official A-E results remain unchanged. Preserved-checkpoint outputs supply these reproducible diagnostics; "
              "exact training-time metric reproduction is not claimed. FP16 checkpoint serialization versus the original "
              "FP32 EMA remains a plausible, unproven explanation.", "",
              "## A. IoU sensitivity", "",
              "Source: `analysis.analysis_a_iou_sensitivity`. Matched GT count equals TP in every row.", ""]
    lines += _table(["Mode", "Confidence", "IoU", "TP / matched GT", "FP", "FN", "Precision", "Recall", "F1"],
                    [["Aware" if row["class_aware"] else "Agnostic", f"{row['confidence']:.2f}", f"{row['iou']:.2f}", *_metrics(row)] for row in a])
    lines += ["![Validation recall across IoU thresholds](../../figures/error_analysis/iou_recall.png)", "",
              f"Measured: at confidence 0.05, class-aware recall increases from {_rate(aware_iou[0.05, 0.50]['recall'])} "
              f"({aware_iou[0.05, 0.50]['tp']}/{validation['ground_truth_count']}) at IoU 0.50 to "
              f"{_rate(aware_iou[0.05, 0.10]['recall'])} ({aware_iou[0.05, 0.10]['tp']}/{validation['ground_truth_count']}) "
              f"at IoU 0.10. At confidence 0.25, it increases from {_rate(validation['recall'])} "
              f"({validation['tp']}/{validation['ground_truth_count']}) to {_rate(aware_iou[0.25, 0.10]['recall'])} "
              f"({aware_iou[0.25, 0.10]['tp']}/{validation['ground_truth_count']}). "
              "Interpretation: looser geometry recovers some matches, but most annotations remain unmatched even "
              "under permissive overlap. IoU 0.10 is approximate spatial overlap, not clinically adequate localization. "
              "These differences do not uniquely isolate localization error or any other mechanism.", "",
              "## B. Confidence sensitivity and localization association", "",
              "Source: `analysis.analysis_b_confidence_sensitivity`. IoU is fixed at 0.50; the class-agnostic "
              "comparison appears in F2. No deployment threshold is selected.", ""]
    lines += _table(["Confidence", "TP", "FP", "FN", "Precision", "Recall", "F1"],
                    [[f"{row['confidence']:.2f}", *_metrics(row)] for row in b["operating_points"] if row["class_aware"]])
    lines += ["![Precision, recall, and F1 sensitivity](../../figures/error_analysis/confidence_sensitivity.png)", "",
              f"Measured: lowering confidence from 0.25 to 0.01 adds {low['tp']-reference['tp']} TP "
              f"({reference['tp']} to {low['tp']}), while FP rises from {reference['fp']} to {low['fp']}. "
              f"Recall increases from {_rate(reference['recall'])} to {_rate(low['recall'])}, with precision falling "
              f"from {_rate(reference['precision'])} to {_rate(low['precision'])}. "
              "Thus useful lower-confidence candidates coexist with a large FP burden.", "",
              f"Candidate localization uses only the {quality['annotated_image_candidate_count']} predictions on "
              f"annotated images; {quality['excluded_empty_image_candidate_count']} predictions on empty-label images "
              "are excluded because they have no GT localization target. Each candidate is compared with its best "
              "GT overlap, independently of one-to-one matching. Multiple candidates can overlap the same annotation.", ""]
    lines += _table(["Confidence interval", "Candidates", "Median max IoU (any / correct class)", "IoU>=0.50 any class", "IoU>=0.50 correct class"],
                    [[f"[{row['confidence_lower_inclusive']:g}, {row['confidence_upper']:g}{']' if row['upper_inclusive'] else ')'}",
                      row["candidates"], f"{_number(row['best_any_class_iou']['median'])} / {_number(row['best_correct_class_iou']['median'])}",
                      f"{row['any_class_iou_at_least_0_5_count']} ({_rate(row['any_class_iou_at_least_0_5_count']/row['candidates'])})",
                      f"{row['correct_class_iou_at_least_0_5_count']} ({_rate(row['correct_class_iou_at_least_0_5_count']/row['candidates'])})"]
                     for row in quality["bins"]])
    lines += [f"Measured Spearman score/max-IoU association: {_number(quality['spearman_confidence_vs_best_any_class_iou'])} "
              f"for any GT class and {_number(quality['spearman_confidence_vs_best_correct_class_iou'])} for correct-class GT. "
              "The highest-confidence bin has better overlap on average, while the overall rank association is weak. "
              "This is a descriptive candidate association with clustered observations; it is not calibration, "
              "one-to-one precision, statistical significance, or causal evidence.", "",
              "## C. False-negative taxonomy", "",
              "Source: `analysis.analysis_c_fn_taxonomy`; per-GT candidate identities and flags: "
              f"`{Path(record['analysis_results_path']).parent.as_posix()}/false_negatives.jsonl`. "
              "Definitions were saved before real-data categorization; the pre-execution record and hash are in the evidence.", "",
              "A nearby retained candidate has IoU>=0.10. Strong signals are: same-class/confidence>=0.25/IoU>=0.50 "
              "already assigned to another GT (conflict); same-class/confidence<0.25/IoU>=0.50 (low confidence); "
              "wrong-class/confidence>=0.25/IoU>=0.50 (classification); same-class/confidence>=0.25/0.10<=IoU<0.50 "
              "(localization). Conflict takes precedence. Otherwise two or more strong signals give mixed/unresolved, "
              "one gives its corresponding category, no nearby candidate gives complete miss, and remaining cases "
              "are mixed/unresolved. A low-confidence wrong-class poorly localized candidate does not establish any "
              "one simple mechanism. All eight class/confidence/geometry combinations and nonexclusive flags remain available.", ""]
    lines += _table(["Exclusive primary category", "FN count", "% of all reference FN"],
                    [[LABELS[category], c["primary_categories"][category]["count"],
                      f"{c['primary_categories'][category]['percentage_of_fn']:.1f}%"] for category in CATEGORIES])
    lines += ["![False-negative primary categories](../../figures/error_analysis/fn_categories.png)", "",
              "Nonexclusive strong evidence: " + "; ".join(f"{LABELS[category]}: {c['nonexclusive_flag_counts'].get(category, 0)}"
                                                           for category in CATEGORIES[1:5]) + ". "
              f"The {c['nonexclusive_flag_counts'].get('classification_error', 0)} classification flags coexist with "
              "other strong evidence and therefore receive mixed primary labels. "
              "Zero primary classification cases does not mean no wrong-class responses. Low-confidence candidate "
              "existence does not guarantee recovery under one-to-one competition. Complete miss refers only to retained "
              "candidates; model-head responses removed by export filtering cannot be examined.", "",
              "Class-wise primary categories below use count (% of that class's FN), so row percentages sum to 100%.", ""]
    lines += _table(["Class", "FN", "Miss", "Localization", "Low confidence", "Classification", "Conflict", "Mixed"],
                    [[row["class_name"], row["fn"], *(f"{row['categories'][category]['count']} "
                      f"({row['categories'][category]['percentage_of_class_fn']:.1f}%)" for category in CATEGORIES)] for row in c["class_rows"]])
    lines += ["## D. Class-wise, size, and confusion analysis", "",
              "Source: `analysis.analysis_d_classes`. All six original class IDs remain intact; reference class-aware matching.", ""]
    lines += _table(["ID / class", "GT (% of 204)", "TP", "FP", "FN", "Precision", "Recall", "Median matched IoU (n)"],
                    [[f"{row['class_id']} / {row['class_name']}", f"{row['support']} ({row['support_percentage']:.1f}%)",
                      row["tp"], row["fp"], row["fn"], _rate(row["precision"]), _rate(row["recall"]),
                      f"{_number(row['matched_iou']['median'])} ({row['matched_iou']['count']})"] for row in d["class_rows"]])
    lines += [f"Measured: forearm and humerus account for {strong_tp} of {validation['tp']} TP "
              f"({_rate(strong_tp/validation['tp'])}), despite only {strong_support} of {validation['ground_truth_count']} "
              f"GT boxes ({_rate(strong_support/validation['ground_truth_count'])}). Their fixed-threshold recalls are "
              f"{_rate(forearm['recall'])} and {_rate(humerus['recall'])}; other class recalls range from "
              f"{_rate(min(other_recalls))} to {_rate(max(other_recalls))}. "
              "This corroborates the earlier class contrast without establishing its cause. Elbow, shoulder, and wrist "
              "have fewer than 30 GT boxes; all class-specific matched-IoU groups have fewer than 30 observations "
              "(four classes have only one or two TP). Those localization medians are particularly unstable and selection-conditional.", "",
              "Size groups use the GT shorter side after nominal scaling by min(640/width, 640/height), "
              "not the actual padded tensor shape. Each GT enters exactly one group. Cells below show TP/GT "
              "(recall); * marks support<30. These descriptive boundaries were fixed before analysis.", ""]
    size_rows = d["size_rows"]
    lines += _table(["Class", "<32 px", "32 to <64 px", ">=64 px"],
                    [[name, *(f"{row['tp']}/{row['support']} ({_rate(row['recall'])}){'*' if row['low_support_warning'] else ''}"
                              for row in size_rows if row["class_name"] == name)]
                     for name in ["All classes", *(row["class_name"] for row in d["class_rows"])]])
    lines += ["Measured validation recall from smaller to larger groups: " + "; ".join(
              f"{row['tp']}/{row['support']} ({_rate(row['recall'])})" for row in size_rows if row["class_id"] is None) + ". "
              "Class composition and very limited support in most class/size cells confound interpretation. "
              "The <30 marker is a caution rule, not a statistical power calculation or evidence of a universal size cutoff.", "",
              "![Class confusion of geometric matches](../../figures/error_analysis/class_confusion.png)", "",
              f"At the class-agnostic reference point, {confusion['wrong_class_matches']} of "
              f"{confusion['geometric_matches']} geometric matches ({confusion['wrong_class_percentage_of_geometric_matches']:.1f}%) "
              "have the wrong class: " + "; ".join(
                  f"{count} {d['class_rows'][gt_id]['class_name']} GT predicted as {d['class_rows'][pred_id]['class_name']}"
                  for gt_id, row in enumerate(confusion["matrix"]) for pred_id, count in enumerate(row)
                  if gt_id != pred_id and count) + f". There are {confusion['correct_class_matches']} diagonal matches. "
              "Unmatched GT and predictions are outside the matrix; it is not a "
              "full confusion matrix with a background class. Removing class eligibility can rearrange greedy assignments, "
              f"so {confusion['wrong_class_matches']} wrong-class matches and "
              f"{confusion['additional_gt_matched_when_agnostic']} additional matched GT are different quantities.", "",
              "## E. False positives", "",
              "Source: `analysis.analysis_e_false_positives`; reference class-aware matching. "
              "A false positive means no eligible matching annotation, not a clinically false diagnosis.", ""]
    lines += _table(["Image group", "Images", "FP (% of all FP)", "Images with FP", "% of group images", "FP / all group images", "Median confidence [P25, P75]"],
                    [[name, row["images"], f"{row['fp']} ({row['percentage_of_total_fp']:.1f}%)", row["images_with_fp"],
                      f"{row['percentage_images_with_fp']:.1f}%", _number(row["fp_per_image"]),
                      f"{_number(row['confidence']['median'])} [{_number(row['confidence']['p25'])}, {_number(row['confidence']['p75'])}]"]
                     for name, row in e["groups"].items()])
    lines += _table(["Predicted class", f"All FP (% of {e['total_fp']})", "Annotated-image FP", "Empty-label-image FP"],
                    [[row["class_name"], f"{row['fp']} ({row['percentage_of_total_fp']:.1f}%)",
                      e["groups"]["annotated"]["by_predicted_class"][index]["fp"],
                      e["groups"]["empty_label"]["by_predicted_class"][index]["fp"]] for index, row in enumerate(e["by_predicted_class"])])
    lines += ["![False positives by class and image group](../../figures/error_analysis/false_positives.png)", "",
              f"All {e['total_fp']} FP have median confidence {_number(e['confidence']['median'])}, "
              f"P10-P90 {_number(e['confidence']['p10'])}-{_number(e['confidence']['p90'])}, "
              f"range {_number(e['confidence']['min'])}-{_number(e['confidence']['max'])}. "
              f"At prediction-pair IoU>=0.50, {e['fp_with_same_class_overlap_count']} FP overlap another retained same-class prediction "
              f"and {e['fp_with_cross_class_overlap_count']} overlap another-class prediction. "
              f"{e['duplicate_like_fp_count']} FP overlap a matched same-class TP (duplicate-like). Unique unordered pairs "
              f"involving an FP: {e['unique_overlapping_pairs_involving_fp']['same_class']} same-class and "
              f"{e['unique_overlapping_pairs_involving_fp']['different_class']} cross-class. FP counts and pair counts have "
              "different denominators and signals can overlap. These measures do not identify anatomical causes.", "",
              f"Empty-label images account for {empty['fp']}/{e['total_fp']} FP; {empty['images_with_fp']}/{empty['images']} "
              f"({empty['percentage_images_with_fp']:.1f}%) have a prediction at the reference threshold. "
              f"Most FP occur on annotated images ({annotated['fp']}/{e['total_fp']}). Empty labels are not medically confirmed negatives, "
              "and annotation completeness is unresolved. Detailed visual assessment is reserved for Stage 3.", "",
              "## F. Generalization and class-agnostic matching", "",
              "### F1. Train versus validation", "",
              "Source: `analysis.analysis_f_generalization`. Same preserved checkpoint, inference settings, "
              "IoU 0.50, and class-aware matcher; training evaluation does not fit or update the model.", ""]
    lines += _table(["Split", "Confidence", "TP", "FP", "FN", "Precision", "Recall", "F1"],
                    [[row["split"], f"{row['confidence']:.2f}", *_metrics(row)] for row in f["operating_points"]])
    lines += _table(["Class", "Train TP/GT (recall)", "Validation TP/GT (recall)"],
                    [[train["class_name"], f"{train['tp']}/{train['support']} ({_rate(train['recall'])})",
                      f"{valid['tp']}/{valid['support']} ({_rate(valid['recall'])})"]
                     for train, valid in zip(f["train_class_rows"], d["class_rows"])])
    lines += ["![Train and validation fixed-threshold comparison](../../figures/error_analysis/train_validation.png)", ""]
    lines += _table(["Reference distribution (conditional on matching)", "Train median", "Validation median"],
                    [[label, _number(f["train_reference_distributions"][key]["median"]),
                      _number(f["validation_reference_distributions"][key]["median"])]
                     for key, label in (("matched_prediction_confidence", "TP confidence"), ("matched_iou", "TP IoU"))])
    lines += [f"Measured reference recall gap: {100 * (training['recall'] - validation['recall']):.1f} percentage points. "
              "Training recall is high across all classes, including those with very low validation recall. "
              "Interpretation: the checkpoint can fit the training annotations, with a substantial failure to generalize "
              "under this protocol. This is consistent with overfitting and/or distribution or annotation differences; "
              "it does not distinguish those causes. TP confidence and overlap distributions are conditional on meeting "
              "the reference thresholds and are not population-wide calibration or localization-error estimates.", "",
              "### F2. Class-aware versus class-agnostic", "",
              "Both modes use identical D candidates and IoU 0.50. P/R/F1 below are percentages.", ""]
    lines += _table(["Confidence", "Aware TP/FP/FN", "Aware P/R/F1", "Agnostic TP/FP/FN", "Agnostic P/R/F1"],
                    [[f"{aware['confidence']:.2f}", f"{aware['tp']}/{aware['fp']}/{aware['fn']}",
                      "/".join(_rate(aware[key]) for key in ("precision", "recall", "f1")),
                      f"{agnostic['tp']}/{agnostic['fp']}/{agnostic['fn']}",
                      "/".join(_rate(agnostic[key]) for key in ("precision", "recall", "f1"))]
                     for aware, agnostic in zip([row for row in b["operating_points"] if row["class_aware"]],
                                               [row for row in b["operating_points"] if not row["class_aware"]])])
    lines += [f"At the reference point, ignoring labels increases TP from {reference['tp']} to "
              f"{agnostic_confidence[0.25]['tp']} and recall from {_rate(reference['recall'])} to "
              f"{_rate(agnostic_confidence[0.25]['recall'])}. At confidence 0.01 it increases TP from "
              f"{low['tp']} to {agnostic_confidence[0.01]['tp']} and recall from {_rate(low['recall'])} to "
              f"{_rate(agnostic_confidence[0.01]['recall'])}. "
              "Classification contributes some errors, but removing class equality does not resolve low validation recall. "
              "NMS was still class-aware; this is neither independently class-agnostic NMS nor class-agnostic AP. "
              "The optional D/E comparison is omitted: E needs a separate export and has a different, merged target definition. "
              "Its native AP is not directly interchangeable with these six-class diagnostics.", "",
              "## Interpretation, limitations, and follow-up", "",
              "Measured evidence points to several interacting limitations: retained misses, usable low-confidence "
              "candidates, incomplete localization, strongly uneven class behavior, annotation-based FP, and a large "
              "train/validation gap. The exclusive taxonomy intentionally leaves many cases mixed/unresolved. "
              "Neither these counts nor the small class-agnostic gain proves one dominant causal factor.", "",
              "Limitations: one checkpoint/seed, small class and size groups, unadjudicated annotations and uncertain "
              "empty-label semantics, post-NMS/export truncation, conditional matched-box statistics, and unknown "
              "patient/study independence. Encoded image hashes and filenames have zero exact train/validation intersections; "
              "this does not rule out near-duplicates or shared patients. No held-out test files were opened. "
              "The Stage 1 reproduction failure remains unresolved and separate from passing Stage 2 consistency checks.", "",
              f"Preservation checks compare {record['preservation']['protected_file_count']} protected files with the "
              "pre-execution hash baseline, covering official A-E artifacts, configurations, Stage 1 evidence/exports, "
              "dependency files, and the three pre-existing Phase 2F changes. Restricted train/validation source "
              "fingerprints and checkpoint hashes remain unchanged. No model training, split changes, dependency "
              "changes, qualitative overlays, or thesis Word edits were performed.", "",
              "Stage 3 should visually inspect a deterministic class/size-stratified selection of retained misses, "
              "low-confidence and mixed FN, the few wrong-class geometric matches, duplicate-like FP, and FP in "
              "both annotated and empty-label groups. Review annotation alignment/completeness and image-domain "
              "differences as questions, without presuming anatomical or medical explanations.", "",
              "For a later Experiment F, the strong generalization gap supports prioritizing one controlled, justified "
              "generalization intervention after Stage 3 (for example a constrained augmentation or regularization change). "
              "Class and scale support may motivate a targeted hypothesis, but current evidence does not select the "
              "intervention, justify class merging as a solution, or support threshold tuning. Preserve A-E, fixed "
              "splits and test isolation; predefine one changed factor and validation criteria before any new experiment. "
              "No Experiment F configuration or training is created here.", "",
              "## Reproduction", "",
              "Run from the repository root with the existing project environment:", "", "```powershell",
              ".venv/Scripts/python.exe -B -m bone_fracture_pipeline.error_analysis_quantitative",
              "```", "",
              "A fresh ignored output directory is required. By default the command exports training predictions "
              "in inference mode, verifies the Stage 1 export and source identities, runs all groups, and generates "
              "this report and six figures. To reuse the verified training export without new inference:", "", "```powershell",
              ".venv/Scripts/python.exe -B -m bone_fracture_pipeline.error_analysis_quantitative `",
              "  --output outputs/error_analysis/stage2/D_seed42_rerun `",
              "  --reuse-training outputs/error_analysis/stage2/D_seed42_train_export", "```", "",
              "Use `--no-publish` for an ignored numerical rerun that leaves the tracked report/figures unchanged. "
              "Raw FN/FP/per-image JSONL, CSV tables, logs, snapshot caches, and full exports remain ignored. "
              "No command accepts a test-split option.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
