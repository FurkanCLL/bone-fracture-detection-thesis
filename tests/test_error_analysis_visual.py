from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from bone_fracture_audit.audit import sha256_file
from bone_fracture_pipeline.detection_evaluation import GroundTruth, Prediction
from bone_fracture_pipeline.error_analysis import ErrorAnalysisError
from bone_fracture_pipeline.error_analysis_quantitative import analyze_exports
from bone_fracture_pipeline.error_analysis_visual import (
    QUOTAS, _make_case, contact_sheets, display_predictions, focus_viewport, load_sources,
    finalize_review, map_box, panel_transform, render_case, run_visual, select_cases, validate_reviews,
)


class VisualAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def row(self,name="example",*,split="valid",targets=(),predictions=()):
        return {"sample_id":f"{split}/{name}","split":split,"image":f"{split}/images/{name}.png",
                "width":400,"height":300,"image_sha256":name,"label_sha256":"label",
                "ground_truth":[{"box_id":item.box_id,"class_id":item.class_id,"xyxy":list(item.xyxy)} for item in targets],
                "predictions":[{"box_id":item.box_id,"class_id":item.class_id,"xyxy":list(item.xyxy),"confidence":item.confidence} for item in predictions]}

    def fixture(self):
        rows=[]
        for index in range(6):
            target=GroundTruth("gt/0001",index,(20,20,80,80))
            rows.append(self.row(f"miss{index}",targets=[target]))
            rows.append(self.row(f"low{index}",targets=[target],predictions=[Prediction("pred/000000",index,target.xyxy,.1)]))
            rows.append(self.row(f"tp{index}",targets=[target],predictions=[Prediction("pred/000000",index,target.xyxy,.9)]))
        rows.append(self.row("wrong",targets=[GroundTruth("gt/0001",5,(20,20,80,80))],
                             predictions=[Prediction("pred/000000",1,(20,20,80,80),.8)]))
        rows.append(self.row("local",targets=[GroundTruth("gt/0001",3,(20,20,80,80))],
                             predictions=[Prediction("pred/000000",3,(20,20,40,80),.8)]))
        rows.append(self.row("duplicate",targets=[GroundTruth("gt/0001",2,(20,20,80,80))],
                             predictions=[Prediction("pred/000000",2,(20,20,80,80),.9),Prediction("pred/000001",2,(20,20,70,80),.7)]))
        rows.append(self.row("empty",predictions=[Prediction("pred/000000",4,(20,20,80,80),.6)]))
        training=[self.row("train",split="train")]
        _,raw=analyze_exports(rows,training)
        return rows,raw

    def test_selection_is_seeded_and_input_order_independent(self):
        rows,raw=self.fixture()
        first=select_cases(rows,raw)
        second=select_cases(list(reversed(rows)),{key:list(reversed(value)) for key,value in raw.items()})
        self.assertEqual(first,second)
        self.assertEqual(first["seed"],42)
        self.assertNotEqual(first,select_cases(rows,raw,seed=43))

    def test_shortfalls_are_reported_without_duplicate_observations(self):
        rows,raw=self.fixture()
        result=select_cases(rows,raw)
        identities=[(case["category"],case["sample_id"],case["ground_truth_id"],case["prediction_id"]) for case in result["cases"]]
        self.assertEqual(len(identities),len(set(identities)))
        self.assertEqual(result["selected_counts"]["localization_failure"],1)
        self.assertEqual(result["shortfalls"]["localization_failure"],2)
        self.assertEqual(result["selected_counts"]["wrong_class_geometric_match"],1)

    def test_sampled_misses_cover_all_six_available_classes(self):
        rows,raw=self.fixture()
        result=select_cases(rows,raw)
        misses=[case for case in result["cases"] if case["category"]=="complete_miss"]
        self.assertEqual({case["class_id"] for case in misses},set(range(6)))
        self.assertEqual(len(misses),len({case["sample_id"] for case in misses}))

    def test_wrong_class_case_preserves_exact_ids_geometry_and_overlap(self):
        rows,raw=self.fixture()
        case=next(case for case in select_cases(rows,raw)["cases"] if case["category"]=="wrong_class_geometric_match")
        self.assertEqual((case["ground_truth_id"],case["prediction_id"]),("gt/0001","pred/000000"))
        self.assertEqual((case["class_id"],case["prediction"]["class_id"]),(5,1))
        self.assertEqual(case["iou"],1)
        self.assertTrue(case["reference_gt_is_fn"])
        self.assertTrue(case["reference_prediction_is_fp"])

    def test_duplicate_like_fp_and_stronger_weaker_success_are_included(self):
        rows,raw=self.fixture()
        cases=select_cases(rows,raw)["cases"]
        self.assertTrue(any(case.get("fp_evidence",{}).get("same_class_overlap_with_matched_tp") for case in cases))
        successful=[case for case in cases if case["category"]=="successful_detection"]
        self.assertTrue(any(case["class_id"] in (2,3) for case in successful))
        self.assertTrue(any(case["class_id"] in (0,1,4,5) for case in successful))

    def test_cross_category_image_reuse_is_transparent(self):
        rows,raw=self.fixture()
        cases=select_cases(rows,raw)["cases"]
        for case in cases:
            expected=[other["case_id"] for other in cases if other["sample_id"]==case["sample_id"] and other is not case]
            self.assertEqual(case["same_image_case_ids"],expected)

    def test_nonvalidation_rows_and_invalid_quotas_are_rejected(self):
        rows,raw=self.fixture()
        for split in ("train","test"):
            with self.assertRaises(ErrorAnalysisError):
                select_cases([self.row(split=split)],raw)
        for quotas in ({}, {**QUOTAS,"complete_miss":-1},{**QUOTAS,"complete_miss":1.2}):
            with self.assertRaises(ErrorAnalysisError):
                select_cases(rows,raw,quotas=quotas)

    def test_display_transform_preserves_aspect_ratio_and_original_coordinates(self):
        transform=panel_transform([50,20,250,120],[10,30,400,400])
        self.assertEqual(transform["scale"],2)
        self.assertEqual(map_box([75,40,100,70],transform),[60,170,110,230])
        self.assertEqual(map_box([0,0,100,50],transform),[10,130,110,190])
        self.assertIsNone(map_box([0,0,20,10],transform))

    def test_fp_focus_crop_follows_prediction_even_without_gt_overlap(self):
        row=self.row(targets=[GroundTruth("gt/0001",1,(0,0,10,10))],predictions=[Prediction("pred/000000",1,(300,200,390,290),.8)])
        case=_make_case(row,"fp_annotated",row["ground_truth"][0],row["predictions"][0])
        crop=focus_viewport(case,row)
        self.assertLessEqual(crop[0],300)
        self.assertGreaterEqual(crop[2],390)
        self.assertLessEqual(crop[1],200)
        self.assertGreaterEqual(crop[3],290)

    def test_crops_stay_inside_original_image(self):
        for box in ((0,0,10,10),(390,290,400,300),(0,0,400,300)):
            row=self.row(targets=[GroundTruth("gt/0001",0,box)])
            case=_make_case(row,"complete_miss",row["ground_truth"][0])
            x0,y0,x1,y1=focus_viewport(case,row)
            self.assertTrue(0 <= x0 < x1 <= row["width"] and 0 <= y0 < y1 <= row["height"])

    def test_display_caps_preserve_low_confidence_focus(self):
        predictions=[Prediction(f"pred/{index:06d}",0,(20,20,80,80),.0011+index*.05) for index in range(15)]
        row=self.row(targets=[GroundTruth("gt/0001",0,(20,20,80,80))],predictions=predictions)
        case=_make_case(row,"low_confidence_detection",row["ground_truth"][0],row["predictions"][0])
        clean=display_predictions(case,row,detailed=False)
        detailed=display_predictions(case,row,detailed=True)
        self.assertLessEqual(len(clean),6)
        self.assertEqual(len(detailed),8)
        self.assertEqual(clean[0]["box_id"],"pred/000000")
        self.assertEqual(detailed[0]["box_id"],"pred/000000")

    def test_medical_visual_output_guard_runs_before_source_access(self):
        row=self.row()
        with patch("bone_fracture_pipeline.error_analysis_visual.sha256_file",side_effect=AssertionError("No reads")):
            with self.assertRaises(ErrorAnalysisError):
                render_case(self.root,row,{"sample_id":row["sample_id"]},self.root/"docs/forbidden.png")
        with self.assertRaises(ErrorAnalysisError):
            contact_sheets(self.root,[],self.root/"docs/forbidden")
        with self.assertRaises(ErrorAnalysisError):
            run_visual(self.root,self.root/"docs/forbidden")

    def test_render_verifies_source_hash_and_keeps_original_geometry(self):
        row=self.row(targets=[GroundTruth("gt/0001",0,(20,20,80,80))],predictions=[Prediction("pred/000000",0,(22,22,78,78),.5)])
        path=self.root/"data/prepared/v3_detection_clahe/valid/images/example.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB",(400,300),"#808080").save(path)
        original_hash=sha256_file(path)
        row["image_sha256"]=original_hash
        case=_make_case(row,"successful_detection",row["ground_truth"][0],row["predictions"][0])
        case["case_id"]="S3-01"
        output=self.root/"outputs/error_analysis/stage3/example.png"
        rendered=render_case(self.root,row,case,output)
        self.assertEqual(sha256_file(path),original_hash)
        self.assertEqual((rendered["width"],rendered["height"]),(1840,1320))
        self.assertEqual(rendered["clean"]["displayed_boxes"][0]["original_xyxy"],row["predictions"][0]["xyxy"])
        row["image_sha256"]="incorrect"
        with self.assertRaises(ErrorAnalysisError):
            render_case(self.root,row,case,output)

    def test_missing_or_changed_evidence_stops_before_validation_loading(self):
        directory=self.root/"docs/evidence/error_analysis"
        directory.mkdir(parents=True)
        (directory/"stage2_quantitative.json").write_text("{}",encoding="utf-8")
        (directory/"stage2_verification.json").write_text(json.dumps({"passed":True,"quantitative_evidence_sha256":"wrong"}),encoding="utf-8")
        with patch("bone_fracture_pipeline.error_analysis_visual.read_stage1_validation",side_effect=AssertionError("No source reads")):
            with self.assertRaises(ErrorAnalysisError):
                load_sources(self.root)

    def test_review_requires_both_inspected_views_and_exact_overlay_hash(self):
        case = {"case_id": "S3-01", "overlay": {"sha256": "figure"}}
        review = {"inspection_status": "inspected", "inspected_views": ["clean_full_image", "detailed_focus_crop"],
                  "overlay_sha256": "figure", "direct_observations": ["A box lies above the annotation."],
                  "plausible_interpretations": ["Its cause is unresolved."], "unverified_claims": ["Clinical correctness is unknown."]}
        self.assertEqual(validate_reviews([case], {"reviews": {"S3-01": review}}), {"S3-01": review})
        for change in ({"inspection_status": "pending"}, {"inspected_views": ["clean_full_image"]},
                       {"overlay_sha256": "other"}, {"direct_observations": []}, {"unverified_claims": [""]}):
            with self.subTest(change=change), self.assertRaises(ErrorAnalysisError):
                validate_reviews([case], {"reviews": {"S3-01": {**review, **change}}})

    def test_missing_or_extra_case_reviews_are_rejected(self):
        for reviews in ({}, {"other": {}}):
            with self.assertRaises(ErrorAnalysisError):
                validate_reviews([{"case_id": "S3-01"}], {"reviews": reviews})

    def test_finalization_rechecks_cases_geometry_files_and_inspection_before_publication(self):
        rows, raw = self.fixture()
        # Synthetic images let the whole publication safeguard run without medical data.
        for row in rows:
            path = self.root / "data/prepared/v3_detection_clahe" / row["image"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (row["width"], row["height"]), "#808080").save(path)
            row["image_sha256"] = sha256_file(path)
        protection = self.root / "outputs/error_analysis/stage3/protection_before.json"
        protection.parent.mkdir(parents=True)
        protection.write_text("{}", encoding="utf-8")
        output = protection.parent / "synthetic"
        provenance = {"validation_export": {"path": "outputs/error_analysis/stage1/fake.jsonl", "sha256": "export"}}
        with patch("bone_fracture_pipeline.error_analysis_visual.load_sources", return_value=(rows, raw, provenance)):
            manifest = run_visual(self.root, output)
            notes = {"reviewer": "Synthetic test fixture; not an actual image review", "reviews": {}, "contact_sheet_reviews": {}}
            for case in manifest["cases"]:
                notes["reviews"][case["case_id"]] = {"inspection_status": "inspected", "inspected_views": ["clean_full_image", "detailed_focus_crop"],
                    "overlay_sha256": case["overlay"]["sha256"], "direct_observations": ["Synthetic test observation."],
                    "plausible_interpretations": ["Synthetic test interpretation."], "unverified_claims": ["No clinical review."]}
            for sheet in manifest["contact_sheets"]:
                notes["contact_sheet_reviews"][sheet["category"]] = {"inspection_status": "inspected", "sha256": sheet["sha256"]}
            notes_path = output / "notes.json"
            notes_path.write_text(json.dumps(notes), encoding="utf-8")
            manifest_path = output / "visual_manifest.json"
            for field in ("class_id", "geometry", "sheet_membership"):
                changed = copy.deepcopy(manifest)
                if field == "class_id":
                    changed["cases"][0]["class_id"] = 99
                elif field == "geometry":
                    changed["cases"][0]["overlay"]["clean"]["transform"]["scale"] += 1
                else:
                    changed["contact_sheets"][0]["case_ids"] = []
                manifest_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ErrorAnalysisError):
                    finalize_review(self.root, output, notes_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            overlay_path = self.root / manifest["cases"][0]["overlay"]["path"]
            original = overlay_path.read_bytes()
            overlay_path.write_bytes(b"changed image")
            with self.assertRaises(ErrorAnalysisError):
                finalize_review(self.root, output, notes_path)
            overlay_path.write_bytes(original)
            result = finalize_review(self.root, output, notes_path)
            self.assertEqual(result["inspected_cases"], len(manifest["cases"]))
            self.assertEqual(result["visual_review_status"], "direct_inspection_complete")
            self.assertNotIn("overlay", result["cases"][0])
            self.assertTrue((self.root / "docs/evidence/error_analysis/stage3_qualitative.json").is_file())


if __name__ == "__main__":
    unittest.main()
