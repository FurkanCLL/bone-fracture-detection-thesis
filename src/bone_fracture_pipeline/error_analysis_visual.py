"""Deterministic validation-case selection and local-only X-ray review figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties, findfont

from bone_fracture_audit.audit import sha256_file
from bone_fracture_pipeline.detection_evaluation import GroundTruth, box_iou
from bone_fracture_pipeline.error_analysis import ErrorAnalysisError, validate_image_paths, _write_json
from bone_fracture_pipeline.error_analysis_quantitative import evaluate_rows, size_group, verify_protected_files
from bone_fracture_pipeline.error_analysis_training import read_stage1_validation
from bone_fracture_pipeline.prepare_dataset import CLASS_NAMES


SEED = 42
QUOTAS = {"complete_miss": 8, "mixed_or_unresolved": 8, "low_confidence_detection": 6,
          "localization_failure": 3, "wrong_class_geometric_match": 3,
          "fp_annotated": 4, "fp_empty_label": 2, "successful_detection": 2}
LABELS = {"complete_miss": "Retained-candidate complete miss", "mixed_or_unresolved": "Mixed or unresolved FN",
          "low_confidence_detection": "Low-confidence FN", "localization_failure": "Localization failure FN",
          "wrong_class_geometric_match": "Wrong-class geometric match", "fp_annotated": "FP on annotated image",
          "fp_empty_label": "FP on empty-label image", "successful_detection": "Successful reference detection"}
COLORS = {"ground_truth": "#23B85B", "reference": "#FFB000", "low": "#36B6FF", "very_low": "#E376FF"}
SELECTION_METHOD = {
    "seed": SEED, "quotas": QUOTAS,
    "ranking": "SHA-256 of seed/category/sample/GT/prediction; seeded pseudorandom tie-breaking independent of input order.",
    "priority": ["Include all three primary localization failures and all three wrong-class geometric matches first.",
                 "Prefer unused images, then unused GT/prediction identities, then underrepresented class and size strata within each category.",
                 "Include a duplicate-like annotated FP when available, followed by balanced FP sampling.",
                 "Select one stronger-class and one weaker-class successful comparison when available."],
    "strata": "Original GT class and Stage 2 nominal-640 shorter-side group; predicted class for empty-label FP.",
    "scope": "Validation-only case observations, purposive quota sampling; no estimate of visual-pattern prevalence.",
    "reference": {"confidence_at_least": 0.25, "iou_at_least": 0.50, "class_aware": True},
    "display": {"clean": "All GT plus up to six reference-score predictions and the selected focus prediction.",
                "detail": "GT and up to eight relevant retained candidates in a labelled focus crop; off-crop candidates are listed.",
                "prediction_bands": {"reference": ">=0.25", "low": "0.01 to <0.25", "very_low": ">0.001 to <0.01"},
                "interpolation": "Bicubic display resizing only; original pixel coordinates and source bytes are preserved."},
}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


# Verify every reused numeric artifact before deriving any qualitative case from it.
def load_sources(root: Path) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    root = root.resolve()
    stage2_path = root / "docs/evidence/error_analysis/stage2_quantitative.json"
    verification_path = root / "docs/evidence/error_analysis/stage2_verification.json"
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if not verification["passed"] or sha256_file(stage2_path) != verification["quantitative_evidence_sha256"]:
        raise ErrorAnalysisError("The Stage 2 evidence differs from its verified record.")
    if sha256_file(root / "docs/evidence/error_analysis/STAGE2_REPORT.md") != verification["report_sha256"]:
        raise ErrorAnalysisError("The preserved Stage 2 report changed.")
    directory = (root / verification["numeric_repeatability"]["verified"]).resolve()
    if not directory.is_relative_to(root / "outputs/error_analysis/stage2"):
        raise ErrorAnalysisError("Stage 2 raw records must remain in their own output directory.")
    hashes = {}
    for artifact in verification["numeric_repeatability"]["artifacts"]:
        path = (directory / artifact["path"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise ErrorAnalysisError("A locally referenced Stage 2 numerical artifact is missing or changed.")
        hashes[path.relative_to(root).as_posix()] = artifact["sha256"]
    train_path = (root / stage2["training_export"]["path"]).resolve()
    if not train_path.is_relative_to(root / "outputs/error_analysis/stage2") or sha256_file(train_path) != stage2["training_export"]["sha256"]:
        raise ErrorAnalysisError("The preserved training export failed its hash check.")
    # This checks all validation identities, source image/label hashes, geometry, and the D checkpoint.
    rows, stage1 = read_stage1_validation(root)
    if stage1["prediction_export"]["sha256"] != stage2["validation_export"]["sha256"]:
        raise ErrorAnalysisError("Stage 1 and Stage 2 refer to different validation predictions.")
    raw = {name: _read_jsonl(directory / f"{name}.jsonl") for name in ("false_negatives", "false_positives", "reference_images")}
    provenance = {"stage1_evidence_sha256": sha256_file(root / "docs/evidence/error_analysis/stage1_validation.json"),
                  "stage2_evidence_sha256": sha256_file(stage2_path), "stage2_verification_sha256": sha256_file(verification_path),
                  "validation_export": stage2["validation_export"], "training_export_sha256": stage2["training_export"]["sha256"],
                  "stage2_artifact_hashes": hashes, "checkpoint": stage2["checkpoint"],
                  "original_stage1_reproduction_passed": False, "original_tolerance": 0.00005}
    return rows, raw, provenance


def _rank(case: Mapping[str, object], seed: int) -> str:
    key = [seed, case["category"], case["sample_id"], case.get("ground_truth_id"), case.get("prediction_id")]
    return hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()


def _candidate(row, identifiers, target):
    eligible = [pred for pred in row["predictions"] if pred["box_id"] in identifiers]
    return min(eligible, key=lambda pred: (-box_iou(pred["xyxy"], target["xyxy"]), -pred["confidence"], pred["box_id"])) if eligible else None


def _make_case(row, category, target=None, prediction=None, fn_record=None, reason=""):
    if row["split"] != "valid" or not row["sample_id"].startswith("valid/"):
        raise ErrorAnalysisError("Qualitative cases must originate from validation only.")
    group, short_side = size_group(GroundTruth(target["box_id"], target["class_id"], tuple(target["xyxy"])), row) if target else (None, None)
    return {"sample_id": row["sample_id"], "category": category,
            "ground_truth_id": target["box_id"] if target else None, "prediction_id": prediction["box_id"] if prediction else None,
            "ground_truth": target, "prediction": prediction,
            "class_id": target["class_id"] if target else prediction["class_id"],
            "class_name": CLASS_NAMES[target["class_id"] if target else prediction["class_id"]],
            "prediction_class_name": CLASS_NAMES[prediction["class_id"]] if prediction else None,
            "iou": box_iou(target["xyxy"], prediction["xyxy"]) if target and prediction else None,
            "effective_gt_short_side_px": short_side, "size_group": group,
            "source_dimensions": {"width": row["width"], "height": row["height"]},
            "source_image_sha256": row["image_sha256"], "source_label_sha256": row["label_sha256"],
            "image": row["image"], "image_has_annotations": bool(row["ground_truth"]),
            "candidate_counts": {"all_retained": len(row["predictions"]),
                                 "reference_score": sum(pred["confidence"] >= 0.25 for pred in row["predictions"]),
                                 "low_score": sum(0.01 <= pred["confidence"] < 0.25 for pred in row["predictions"]),
                                 "very_low_score": sum(pred["confidence"] < 0.01 for pred in row["predictions"]),
                                 "nearby_gt_iou_at_least_0_1": sum(box_iou(pred["xyxy"],target["xyxy"]) >= 0.1 for pred in row["predictions"]) if target else None},
            "fn_evidence": fn_record, "selection_reason": reason}


# Select cases from existing numerical labels; visual content cannot influence this selection.
def select_cases(rows: Sequence[Mapping[str, object]], raw: Mapping[str, object], *, seed: int = SEED,
                 quotas: Mapping[str, int] | None = None) -> dict[str, object]:
    if any(row["split"] != "valid" for row in rows):
        raise ErrorAnalysisError("Train/test rows cannot enter visual selection.")
    quotas = dict(QUOTAS if quotas is None else quotas)
    if set(quotas) != set(QUOTAS) or any(type(value) is not int or value < 0 for value in quotas.values()):
        raise ErrorAnalysisError("Sampling quotas must be nonnegative integers for the documented categories.")
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ErrorAnalysisError("Validation sample identities must be unique.")
    pools = {name: [] for name in QUOTAS}
    fn_by_id = {(item["sample_id"], item["ground_truth_id"]): item for item in raw["false_negatives"]}
    for item in raw["false_negatives"]:
        category = item["primary_category"]
        if category not in pools:
            continue
        row = by_id[item["sample_id"]]
        target = next(gt for gt in row["ground_truth"] if gt["box_id"] == item["ground_truth_id"])
        combination = {"low_confidence_detection": "correct_class_low_confidence_sufficient_overlap",
                       "localization_failure": "correct_class_high_confidence_poor_overlap"}.get(category)
        identifiers = item["candidate_ids_by_combination"][combination] if combination else [pred["box_id"] for pred in row["predictions"]]
        pred = _candidate(row, identifiers, target)
        pools[category].append(_make_case(row, category, target, pred, item, "Primary FN category from Stage 2; class/size-balanced seeded selection."))
    for item in raw["false_positives"]:
        row = by_id[item["sample_id"]]
        prediction = next(pred for pred in row["predictions"] if pred["box_id"] == item["prediction_id"])
        category = "fp_annotated" if item["image_group"] == "annotated" else "fp_empty_label"
        # Maximum-IoU GT supplies context only; disjoint ties use the GT ID, not spatial distance.
        target = min(row["ground_truth"], key=lambda gt: (-box_iou(gt["xyxy"], prediction["xyxy"]),gt["box_id"])) if row["ground_truth"] else None
        case = _make_case(row, category, target, prediction, fn_by_id.get((row["sample_id"], target["box_id"])) if target else None,
                          "Unmatched reference prediction from Stage 2; nearest GT is context only.")
        case["fp_evidence"] = item
        case["ground_truth_role"] = "nearest_annotation_only" if target else "no_annotations"
        pools[category].append(case)
    for image in evaluate_rows(rows, confidence=0.25, iou=0.50, class_aware=False):
        row = image["row"]
        gt_by_id = {gt["box_id"]: gt for gt in row["ground_truth"]}
        pred_by_id = {pred["box_id"]: pred for pred in row["predictions"]}
        for match in image["result"].matches:
            target, prediction = gt_by_id[match.ground_truth_id], pred_by_id[match.prediction_id]
            if target["class_id"] != prediction["class_id"]:
                pools["wrong_class_geometric_match"].append(_make_case(row, "wrong_class_geometric_match", target, prediction,
                    fn_by_id.get((row["sample_id"],target["box_id"])), "All available wrong-class class-agnostic reference matches."))
    for item in raw["reference_images"]:
        row = by_id[item["sample_id"]]
        for match in item["matches"]:
            target = next(gt for gt in row["ground_truth"] if gt["box_id"] == match["ground_truth_id"])
            prediction = next(pred for pred in row["predictions"] if pred["box_id"] == match["prediction_id"])
            pools["successful_detection"].append(_make_case(row,"successful_detection",target,prediction,
                fn_by_id.get((row["sample_id"],target["box_id"])), "Reference TP comparison; one stronger and one weaker class when feasible."))
    selected, used_images, used_targets, used_predictions = [], set(), set(), set()
    category_order = ["localization_failure", "wrong_class_geometric_match", *[name for name in QUOTAS if name not in ("localization_failure","wrong_class_geometric_match")]]
    for category in category_order:
        remaining = list(pools[category])
        class_counts, size_counts = Counter(), Counter()
        for index in range(min(quotas[category],len(remaining))):
            eligible = remaining
            if category == "fp_annotated" and index == 0:
                duplicates = [case for case in eligible if case["fp_evidence"]["same_class_overlap_with_matched_tp"]]
                eligible = duplicates or eligible
            if category == "successful_detection" and index < 2:
                allowed = (2,3) if index == 0 else (0,1,4,5)
                eligible = [case for case in eligible if case["class_id"] in allowed] or eligible
            case = min(eligible,key=lambda case:(case["sample_id"] in used_images,
                (case["sample_id"],case["ground_truth_id"]) in used_targets if case["ground_truth_id"] else False,
                (case["sample_id"],case["prediction_id"]) in used_predictions if case["prediction_id"] else False,
                class_counts[case["class_id"]],size_counts[case["size_group"]],_rank(case,seed)))
            remaining.remove(case)
            case = dict(case)
            case["selection_rank"] = _rank(case,seed)
            case["reuses_selected_image"] = case["sample_id"] in used_images
            selected.append(case)
            used_images.add(case["sample_id"])
            if case["ground_truth_id"]:
                used_targets.add((case["sample_id"],case["ground_truth_id"]))
            if case["prediction_id"]:
                used_predictions.add((case["sample_id"],case["prediction_id"]))
            class_counts[case["class_id"]] += 1
            size_counts[case["size_group"]] += 1
    # Display order follows the requested categories, independently of selection priority.
    selected.sort(key=lambda case:(list(QUOTAS).index(case["category"]),case["selection_rank"]))
    for index,case in enumerate(selected,1):
        case["case_id"] = f"S3-{index:02d}"
        case["same_image_case_ids"] = []
    for case in selected:
        case["same_image_case_ids"] = [other["case_id"] for other in selected if other["sample_id"] == case["sample_id"] and other["case_id"] != case["case_id"]]
        reference = next(item for item in raw["reference_images"] if item["sample_id"] == case["sample_id"])
        case["reference_gt_is_fn"] = case["ground_truth_id"] in reference["false_negative_ids"] if case["ground_truth_id"] else None
        case["reference_prediction_is_fp"] = case["prediction_id"] in reference["false_positive_ids"] if case["prediction_id"] else None
        case["reference_matches"] = reference["matches"]
    counts = Counter(case["category"] for case in selected)
    return {"seed":seed,"quotas":quotas,"pool_counts":{name:len(pool) for name,pool in pools.items()},
            "selected_counts":{name:counts[name] for name in QUOTAS},"selected_cases":len(selected),"distinct_images":len(used_images),
            "shortfalls":{name:quotas[name]-counts[name] for name in QUOTAS if counts[name] < quotas[name]},"cases":selected}


def focus_viewport(case, row):
    focus = case["prediction"] if case["category"].startswith("fp_") else case["ground_truth"] or case["prediction"]
    box = list(focus["xyxy"])
    if case["ground_truth"] and case["prediction"] and case["iou"] >= 0.1:
        other = case["ground_truth"]["xyxy"] if case["category"].startswith("fp_") else case["prediction"]["xyxy"]
        box = [min(box[0],other[0]),min(box[1],other[1]),max(box[2],other[2]),max(box[3],other[3])]
    width,height = row["width"],row["height"]
    cx,cy = (box[0]+box[2])/2,(box[1]+box[3])/2
    span_x,span_y = min(width,max(160,(box[2]-box[0])*2.5)),min(height,max(160,(box[3]-box[1])*2.5))
    x0,y0 = max(0,min(width-span_x,cx-span_x/2)),max(0,min(height-span_y,cy-span_y/2))
    return [math.floor(x0),math.floor(y0),math.ceil(x0+span_x),math.ceil(y0+span_y)]


def panel_transform(viewport, panel):
    x0,y0,x1,y1 = viewport
    px,py,pw,ph = panel
    scale = min(pw/(x1-x0),ph/(y1-y0))
    return {"scale":scale,"offset_x":px+(pw-(x1-x0)*scale)/2,"offset_y":py+(ph-(y1-y0)*scale)/2,
            "viewport":list(viewport),"panel":list(panel)}


def map_box(box, transform):
    vx,vy,x1,y1 = transform["viewport"]
    clipped = [max(box[0],vx),max(box[1],vy),min(box[2],x1),min(box[3],y1)]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None
    return [transform["offset_x"]+(clipped[index]-vx)*transform["scale"] if index%2 == 0 else
            transform["offset_y"]+(clipped[index]-vy)*transform["scale"] for index in range(4)]


def display_predictions(case,row,*,detailed):
    by_id = {pred["box_id"]:pred for pred in row["predictions"]}
    selected = [case["prediction_id"]] if case["prediction_id"] else []
    if detailed and case["fn_evidence"]:
        for identifiers in case["fn_evidence"]["candidate_ids_by_combination"].values():
            best = _candidate(row,identifiers,case["ground_truth"])
            if best and best["box_id"] not in selected:
                selected.append(best["box_id"])
    target = case["ground_truth"] or case["prediction"]
    remaining = sorted(row["predictions"],key=lambda pred:(-box_iou(pred["xyxy"],target["xyxy"]),-pred["confidence"],pred["box_id"])) if detailed else sorted(
        (pred for pred in row["predictions"] if pred["confidence"] >= 0.25),key=lambda pred:(-pred["confidence"],pred["box_id"]))
    cap = 8 if detailed else 6
    for pred in remaining:
        if pred["box_id"] not in selected and len(selected) < cap:
            selected.append(pred["box_id"])
    return [by_id[identifier] for identifier in selected[:cap]]


def _font(size,*,bold=False):
    return ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans",weight="bold" if bold else "normal")),size)


def _band(prediction):
    return "reference" if prediction["confidence"] >= 0.25 else "low" if prediction["confidence"] >= 0.01 else "very_low"


# Coordinates are transformed only for display; the manifest retains exact original-image boxes.
def _draw_panel(canvas, source, row, case, panel, viewport, predictions):
    draw = ImageDraw.Draw(canvas)
    transform = panel_transform(viewport,panel)
    crop = source.crop(tuple(viewport))
    size = (round(crop.width*transform["scale"]),round(crop.height*transform["scale"]))
    canvas.paste(crop.resize(size,Image.Resampling.BICUBIC),(round(transform["offset_x"]),round(transform["offset_y"])))
    geometry = []
    # Predictions first and GT last keep the annotation outline visible when boxes coincide.
    for role,items in (("prediction",predictions),("ground_truth",row["ground_truth"])):
        for item in items:
            mapped = map_box(item["xyxy"],transform)
            if mapped is None:
                continue
            focused = item["box_id"] == case["ground_truth_id"] if role == "ground_truth" else item["box_id"] == case["prediction_id"]
            color = COLORS["ground_truth"] if role == "ground_truth" else COLORS[_band(item)]
            draw.rectangle(mapped,outline=color,width=5 if focused else 2)
            label = ("G" if role == "ground_truth" else "P")+str(int(item["box_id"].split("/")[-1]))
            if role == "prediction":
                label += f" {item['confidence']:.3f}"
            font = _font(17,bold=focused)
            bounds = draw.textbbox((0,0),label,font=font)
            x = min(mapped[0],transform["offset_x"]+size[0]-(bounds[2]+6))
            y = max(transform["offset_y"],mapped[1]-bounds[3]-7)
            # GT labels go below the box when possible to separate them from prediction labels.
            if role == "ground_truth":
                y = min(transform["offset_y"]+size[1]-bounds[3]-6,mapped[3]+2)
            draw.rectangle((x,y,x+bounds[2]+6,y+bounds[3]+5),fill="#101820")
            draw.text((x+3,y+1),label,font=font,fill=color)
            geometry.append({"role":role,"box_id":item["box_id"],"original_xyxy":item["xyxy"],"display_xyxy":mapped})
    return {"transform":transform,"displayed_boxes":geometry,"listed_prediction_ids":[pred["box_id"] for pred in predictions]}


def render_case(root:Path,row:Mapping[str,object],case:dict[str,object],output:Path):
    root = root.resolve()
    output = output.resolve()
    if not output.is_relative_to(root / "outputs/error_analysis/stage3"):
        raise ErrorAnalysisError("X-ray overlays must remain under ignored Stage 3 outputs.")
    if row["split"] != "valid" or case["sample_id"] != row["sample_id"]:
        raise ErrorAnalysisError("Overlay source must be its selected validation sample.")
    path = root / "data/prepared/v3_detection_clahe" / row["image"]
    validate_image_paths([path],root / "data/prepared/v3_detection_clahe/valid/images")
    if sha256_file(path) != row["image_sha256"]:
        raise ErrorAnalysisError("An overlay source image changed after export.")
    with Image.open(path) as image:
        if image.size != (row["width"],row["height"]):
            raise ErrorAnalysisError("Original image dimensions differ from the export.")
        source = image.convert("RGB")
    canvas = Image.new("RGB",(1840,1320),"#F3F5F7")
    draw = ImageDraw.Draw(canvas)
    draw.text((30,15),f"{case['case_id']} | {LABELS[case['category']]}",font=_font(27,bold=True),fill="#18212A")
    draw.text((30,58),f"{row['sample_id']} | original {row['width']} x {row['height']} px",font=_font(20),fill="#18212A")
    draw.text((30,100),"Clean: full image, reference scores + selected focus",font=_font(19,bold=True),fill="#18212A")
    viewport = focus_viewport(case,row)
    draw.text((950,100),f"Detail: focus crop {viewport}; retained candidates",font=_font(19,bold=True),fill="#18212A")
    clean_preds,detail_preds = display_predictions(case,row,detailed=False),display_predictions(case,row,detailed=True)
    clean = _draw_panel(canvas,source,row,case,(30,135,860,690),[0,0,row["width"],row["height"]],clean_preds)
    detail = _draw_panel(canvas,source,row,case,(950,135,860,690),viewport,detail_preds)
    legend = [("GT (focused box thicker)",COLORS["ground_truth"]),("Pred >=0.25",COLORS["reference"]),
              ("Pred 0.01 to <0.25",COLORS["low"]),("Pred >0.001 to <0.01",COLORS["very_low"])]
    for index,(text,color) in enumerate(legend):
        x = 30+index*450
        draw.rectangle((x,845,x+22,867),fill=color)
        draw.text((x+30,843),text,font=_font(18),fill="#18212A")
    for x,predictions,panel_info in ((30,clean_preds,clean),(950,detail_preds,detail)):
        y = 890
        target = case["ground_truth"]
        role = "Context GT" if case["category"].startswith("fp_") else "Focus GT"
        gt_label = f"{role} {target['box_id']}: {CLASS_NAMES[target['class_id']]}" if target else "No ground-truth annotations in this image"
        draw.text((x,y),gt_label,font=_font(19,bold=True),fill="#18212A")
        y += 29
        context = f"GT size at 640: {case['effective_gt_short_side_px']:.1f}px ({case['size_group']})" if target else "GT size / GT IoU: not applicable"
        draw.text((x,y),context,font=_font(18),fill="#18212A")
        y += 29
        visible = {box["box_id"] for box in panel_info["displayed_boxes"] if box["role"] == "prediction"}
        for pred in predictions:
            overlap = f"IoU to focus GT={box_iou(pred['xyxy'],target['xyxy']):.3f}" if target else "no annotation target"
            status = " [off crop]" if pred["box_id"] not in visible else ""
            label = f"P{int(pred['box_id'].split('/')[-1])}: {CLASS_NAMES[pred['class_id']]} | {pred['confidence']:.4f} | {overlap}{status}"
            draw.text((x,y),label,font=_font(17),fill="#18212A")
            y += 26
        if not predictions:
            draw.text((x,y),"No candidates displayed in this view.",font=_font(18),fill="#18212A")
    draw.text((30,1240),f"Retained candidates: {case['candidate_counts']['all_retained']}; >=0.25: {case['candidate_counts']['reference_score']}; "
              f"0.01-<0.25: {case['candidate_counts']['low_score']}; <0.01: {case['candidate_counts']['very_low_score']}. "
              "Shown candidates are capped; see manifest for exact IDs and flags.",font=_font(17),fill="#18212A")
    draw.text((30,1275),"Original-coordinate boxes. Bicubic resizing for display only. Annotation-based diagnostic labels; no clinical adjudication.",font=_font(18),fill="#18212A")
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output)
    return {"path":output.relative_to(root).as_posix(),"sha256":sha256_file(output),"width":canvas.width,"height":canvas.height,
            "clean":clean,"detail":detail,"focus_crop":viewport}


def contact_sheets(root:Path,cases:Sequence[Mapping[str,object]],directory:Path):
    if not directory.resolve().is_relative_to(root.resolve()/"outputs/error_analysis/stage3"):
        raise ErrorAnalysisError("X-ray contact sheets must remain in ignored local outputs.")
    directory.mkdir(parents=True,exist_ok=True)
    sheets=[]
    for category in QUOTAS:
        subset=[case for case in cases if case["category"] == category]
        if not subset:
            continue
        sheet=Image.new("RGB",(1840,110+math.ceil(len(subset)/2)*640),"#F3F5F7")
        draw=ImageDraw.Draw(sheet)
        draw.text((25,15),f"{LABELS[category]} | {len(subset)} selected case observations",font=_font(28,bold=True),fill="#18212A")
        draw.text((25,58),"Green GT | orange >=0.25 | blue 0.01-<0.25 | purple <0.01. Individual figures contain full labels and IDs.",font=_font(19),fill="#18212A")
        for index,case in enumerate(subset):
            x,y=(index%2)*920,110+(index//2)*640
            with Image.open(root/case["overlay"]["path"]) as image:
                thumbnail=image.crop((20,95,1820,825)).resize((900,365),Image.Resampling.LANCZOS)
            sheet.paste(thumbnail,(x+10,y+70))
            draw.text((x+20,y+10),f"{case['case_id']} | {case['class_name']}",font=_font(23,bold=True),fill="#18212A")
            pred=case["prediction"]
            score=f"{pred['confidence']:.4f}" if pred else "none"
            iou=f"{case['iou']:.3f}" if case["iou"] is not None else "n/a"
            draw.text((x+20,y+455),f"Focus prediction score={score}; GT IoU={iou}",font=_font(20),fill="#18212A")
            draw.text((x+20,y+490),f"Retained={case['candidate_counts']['all_retained']}; nearby={case['candidate_counts']['nearby_gt_iou_at_least_0_1']}",font=_font(20),fill="#18212A")
            draw.text((x+20,y+525),case["sample_id"][6:66],font=_font(17),fill="#18212A")
            draw.text((x+20,y+560),"See individual overlay and review manifest for full sample ID.",font=_font(17),fill="#18212A")
        path=directory/f"{category}.png"
        sheet.save(path)
        sheets.append({"category":category,"path":path.relative_to(root).as_posix(),"sha256":sha256_file(path),
                       "case_ids":[case["case_id"] for case in subset],"width":sheet.width,"height":sheet.height})
    return sheets


def run_visual(root:Path,output:Path):
    root,output=root.resolve(),output.resolve()
    if output.exists() or not output.is_relative_to(root/"outputs/error_analysis/stage3"):
        raise ErrorAnalysisError("Use a fresh local-only directory under outputs/error_analysis/stage3.")
    protection=root/"outputs/error_analysis/stage3/protection_before.json"
    if not protection.is_file():
        raise ErrorAnalysisError("Capture Stage 1/2 and official artifact protection hashes before Stage 3.")
    verify_protected_files(root,protection)
    output.mkdir(parents=True)
    _write_json(output/"method_pre_selection.json",{"recorded_at_utc":datetime.now(UTC).isoformat(),"method":SELECTION_METHOD})
    rows,raw,provenance=load_sources(root)
    selection=select_cases(rows,raw)
    if selection != select_cases(list(reversed(rows)),{key:list(reversed(value)) for key,value in raw.items()}):
        raise ErrorAnalysisError("Selection changed with source row ordering.")
    _write_json(output/"selection_manifest.json",{"method":SELECTION_METHOD,"provenance":provenance,**selection})
    by_id={row["sample_id"]:row for row in rows}
    for case in selection["cases"]:
        case["overlay"]=render_case(root,by_id[case["sample_id"]],case,output/"individual"/f"{case['case_id']}.png")
        case["source_export_sha256"]=provenance["validation_export"]["sha256"]
        case["source_export_identifier"]=provenance["validation_export"]["path"]
    sheets=contact_sheets(root,selection["cases"],output/"contact_sheets")
    manifest={"schema_version":1,"stage":3,"method":SELECTION_METHOD,"provenance":provenance,**selection,
              "contact_sheets":sheets,"medical_images_publication":"Local ignored outputs only; redistribution not verified or approved.",
              "visual_review_status":"pending_direct_inspection","preservation":verify_protected_files(root,protection),
              "test_files_opened":0,"training_performed":False,"experiment_f_started":False}
    _write_json(output/"visual_manifest.json",manifest)
    _write_json(output/"review_forms.json",{case["case_id"]:{"inspection_status":"pending","direct_observations":[],
                "plausible_interpretations":[],"unverified_claims":["Clinical fracture presence and annotation correctness are not adjudicated."]} for case in selection["cases"]})
    return manifest


# A review is an explicit inspection record, not something inferred from an error label.
def validate_reviews(cases, notes):
    reviews = notes.get("reviews", {})
    if set(reviews) != {case["case_id"] for case in cases}:
        raise ErrorAnalysisError("Every selected case needs exactly one direct inspection record.")
    for case in cases:
        review = reviews[case["case_id"]]
        if review.get("inspection_status") != "inspected" or set(review.get("inspected_views", [])) != {"clean_full_image", "detailed_focus_crop"}:
            raise ErrorAnalysisError("Both rendered views must be inspected before finalization.")
        if review.get("overlay_sha256") != case["overlay"]["sha256"]:
            raise ErrorAnalysisError("The review refers to a different rendered overlay.")
        for field in ("direct_observations", "plausible_interpretations", "unverified_claims"):
            values = review.get(field)
            if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ErrorAnalysisError("Keep observations, interpretations, and unverified claims explicit.")
    return reviews


def _check_visual_file(root, output, descriptor):
    path = (root / descriptor["path"]).resolve()
    if not path.is_relative_to(output) or not path.is_file() or sha256_file(path) != descriptor["sha256"]:
        raise ErrorAnalysisError("A reviewed visual is missing, changed, or outside the local review package.")
    with Image.open(path) as image:
        if image.size != (descriptor["width"], descriptor["height"]):
            raise ErrorAnalysisError("A rendered visual has unexpected dimensions.")


# Recheck source identities and display transforms before publishing text-only evidence.
def finalize_review(root: Path, output: Path, notes_path: Path):
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root / "outputs/error_analysis/stage3"):
        raise ErrorAnalysisError("Finalize an existing ignored Stage 3 review package.")
    manifest_path = output / "visual_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    rows, raw, provenance = load_sources(root)
    selection = select_cases(rows, raw)
    if manifest["method"] != SELECTION_METHOD or manifest["provenance"] != provenance:
        raise ErrorAnalysisError("Review methods or source provenance changed after rendering.")
    for key, value in selection.items():
        if key != "cases" and manifest[key] != value:
            raise ErrorAnalysisError("Review sampling counts differ from deterministic reselection.")
    if len(manifest["cases"]) != len(selection["cases"]):
        raise ErrorAnalysisError("The rendered selection has an unexpected case count.")
    by_id = {row["sample_id"]: row for row in rows}
    for actual, expected in zip(manifest["cases"], selection["cases"], strict=True):
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ErrorAnalysisError("A selected case differs from the original numerical records.")
        if actual["source_export_sha256"] != provenance["validation_export"]["sha256"] or actual["source_export_identifier"] != provenance["validation_export"]["path"]:
            raise ErrorAnalysisError("A case has inconsistent prediction-export provenance.")
        row, overlay = by_id[actual["sample_id"]], actual["overlay"]
        _check_visual_file(root, output, overlay)
        if overlay["focus_crop"] != focus_viewport(actual, row):
            raise ErrorAnalysisError("A focus crop differs from its original-image geometry.")
        for name, detailed, viewport, panel in (("clean", False, [0, 0, row["width"], row["height"]], [30, 135, 860, 690]),
                                               ("detail", True, overlay["focus_crop"], [950, 135, 860, 690])):
            rendered = overlay[name]
            transform = panel_transform(viewport, panel)
            predictions = display_predictions(actual, row, detailed=detailed)
            expected_boxes = []
            for role, items in (("prediction", predictions), ("ground_truth", row["ground_truth"])):
                for item in items:
                    mapped = map_box(item["xyxy"], transform)
                    if mapped is not None:
                        expected_boxes.append({"role": role, "box_id": item["box_id"], "original_xyxy": item["xyxy"], "display_xyxy": mapped})
            if rendered["transform"] != transform or rendered["displayed_boxes"] != expected_boxes or rendered["listed_prediction_ids"] != [pred["box_id"] for pred in predictions]:
                raise ErrorAnalysisError("Displayed boxes differ from the verified source coordinates.")
    sheets = manifest["contact_sheets"]
    if {sheet["category"] for sheet in sheets} != {case["category"] for case in manifest["cases"]}:
        raise ErrorAnalysisError("Each selected category needs a contact sheet.")
    sheet_reviews = notes.get("contact_sheet_reviews", {})
    if set(sheet_reviews) != {sheet["category"] for sheet in sheets}:
        raise ErrorAnalysisError("Every category sheet needs a recorded readability check.")
    for sheet in sheets:
        _check_visual_file(root, output, sheet)
        expected_ids = [case["case_id"] for case in manifest["cases"] if case["category"] == sheet["category"]]
        review = sheet_reviews[sheet["category"]]
        if sheet["case_ids"] != expected_ids or review.get("inspection_status") != "inspected" or review.get("sha256") != sheet["sha256"]:
            raise ErrorAnalysisError("Contact-sheet membership or inspection evidence is inconsistent.")
    reviews = validate_reviews(manifest["cases"], notes)
    protection = verify_protected_files(root, root / "outputs/error_analysis/stage3/protection_before.json")
    for case in manifest["cases"]:
        case["review"] = reviews[case["case_id"]]
    manifest.update(visual_review_status="direct_inspection_complete", preservation=protection,
                    reviewer=notes["reviewer"], contact_sheet_reviews=sheet_reviews)
    _write_json(manifest_path, manifest)
    _write_json(output / "review_forms.json", reviews)
    # Public evidence contains numerical boxes and text; medical pixels stay in ignored outputs.
    public_cases = []
    for case in manifest["cases"]:
        record = {key: value for key, value in case.items() if key != "overlay"}
        record["local_overlay"] = {key: case["overlay"][key] for key in ("path", "sha256", "width", "height", "focus_crop")}
        record["displayed_prediction_ids"] = {name: case["overlay"][name]["listed_prediction_ids"] for name in ("clean", "detail")}
        public_cases.append(record)
    evidence = {key: value for key, value in manifest.items() if key != "cases"}
    evidence.update(status="complete_with_preserved_stage1_reproduction_limitation", cases=public_cases,
                    reviewed_manifest={"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256_file(manifest_path)},
                    review_notes_sha256=sha256_file(notes_path), inspected_cases=len(reviews),
                    class_counts=dict(sorted(Counter(case["class_name"] for case in public_cases).items())),
                    gt_size_group_counts=dict(Counter(case["size_group"] for case in public_cases if case["ground_truth"])),
                    experiment_f_direction="Controlled alternative-detector comparison; not chosen, configured, implemented, or trained.",
                    interpretation_limits=["Purposive quota selection does not estimate visual-pattern prevalence.",
                                           "Direct inspection by Codex is not radiologist adjudication.",
                                           "Clinical correctness, annotation completeness, patient independence, and error causes remain unverified."])
    _write_json(root / "docs/evidence/error_analysis/stage3_qualitative.json", evidence)
    return evidence


def main(argv=None):
    parser=argparse.ArgumentParser(description="Select and render local-only validation X-ray diagnostics; no inference.")
    parser.add_argument("--project-root",type=Path,default=Path.cwd())
    parser.add_argument("--output",type=Path,default=Path("outputs/error_analysis/stage3/D_seed42"))
    parser.add_argument("--finalize", action="store_true", help="Validate completed inspection notes and write text-only evidence.")
    parser.add_argument("--reviews", type=Path, help="JSON notes with explicit inspection records for every case and category sheet.")
    args=parser.parse_args(argv)
    output=args.output if args.output.is_absolute() else args.project_root/args.output
    if args.finalize:
        if args.reviews is None:
            parser.error("--finalize requires --reviews")
        notes = args.reviews if args.reviews.is_absolute() else args.project_root / args.reviews
        manifest = finalize_review(args.project_root, output, notes)
    else:
        manifest=run_visual(args.project_root,output)
    print(json.dumps({key:manifest[key] for key in ("selected_counts","distinct_images","shortfalls","visual_review_status")},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
