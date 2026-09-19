# Phase 2B Conversion Validation Report

**Dataset:** Prepared v3 YOLO detection dataset

**Validation date:** 2026-09-19

**Status:** Phase 2B technically passed; prepared dataset is ready for Phase 2C protocol work

## Scope

Phase 2B independently checked the polygon-to-box conversion produced in Phase 2A. It did not modify the raw v3 source or `data/prepared/v3_detection`, and it did not introduce preprocessing, augmentation, training, or model evaluation.

The checks covered all 998 source polygons and their corresponding prepared detection boxes. Source and prepared labels were parsed separately. Expected polygon bounds, polygon areas, occupancy ratios, file counts, class counts, and image hashes were recalculated without trusting the Phase 2A summary as the source of those values.

## Geometry method

For each polygon, the validator independently calculated its coordinate extrema and minimum enclosing axis-aligned box. It then verified:

- the prepared row is a valid normalized YOLO detection box;
- source and prepared class IDs match;
- every source vertex lies inside or on the prepared box;
- the prepared box matches the independently calculated minimum bounds;
- polygon and box areas are non-zero;
- one source polygon corresponds to one prepared detection row.

Phase 2A serializes coordinates to 10 decimal places. A tolerance of `1e-10` was therefore used for containment and minimum-box comparisons. This covers the maximum possible edge displacement from the approved serialization precision while remaining far below a meaningful geometric difference.

## Validation results

| Check | Result |
|---|---:|
| Annotations checked | 998 |
| Vertex-containment errors | 0 |
| Minimum-box equivalence errors | 0 |
| Invalid normalized boxes | 0 |
| Source/prepared image hash mismatches | 0 |
| Images | 1,728 |
| Labels | 1,728 |
| Empty labels | 868 |

The independent split counts were reproduced exactly:

| Split | Images / labels | Annotations | Empty labels |
|---|---:|---:|---:|
| Train | 1,211 | 698 | 607 |
| Validation | 348 | 204 | 175 |
| Test | 169 | 96 | 86 |

The class mapping and annotation counts also remained unchanged:

| ID | Class | Annotations |
|---:|---|---:|
| 0 | elbow positive | 159 |
| 1 | fingers positive | 253 |
| 2 | forearm fracture | 164 |
| 3 | humerus fracture | 155 |
| 4 | shoulder fracture | 157 |
| 5 | wrist positive | 110 |

## Occupancy findings

Occupancy is polygon area divided by the area of its enclosing detection box. It is a diagnostic measure and was not used to remove or alter annotations.

| Scope | Count | Minimum | Q1 | Median | Mean | Q3 | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 998 | 0.0493 | 0.5851 | 0.6910 | 0.6854 | 0.7900 | 0.9774 |
| elbow positive | 159 | 0.0493 | 0.6109 | 0.6907 | 0.6804 | 0.7631 | 0.9291 |
| fingers positive | 253 | 0.3839 | 0.5971 | 0.6847 | 0.6858 | 0.7809 | 0.9774 |
| forearm fracture | 164 | 0.4607 | 0.6999 | 0.7817 | 0.7577 | 0.8365 | 0.9704 |
| humerus fracture | 155 | 0.3340 | 0.5232 | 0.5871 | 0.6186 | 0.7075 | 0.9195 |
| shoulder fracture | 157 | 0.4551 | 0.5789 | 0.6668 | 0.6766 | 0.7879 | 0.9663 |
| wrist positive | 110 | 0.3777 | 0.5856 | 0.7011 | 0.6905 | 0.7904 | 0.9340 |

The lowest occupancy, 0.0493, belongs to a validation `elbow positive` annotation. Its visual overlay confirms that the generated box is the correct minimum axis-aligned envelope around a narrow polygon. It remains a review-worthy shape but is not a conversion error. Humerus annotations have the lowest class median occupancy (0.5871), meaning their polygons generally occupy less of their enclosing boxes than the other classes.

## Review candidates and visual QA

The validator created 88 machine-readable annotation candidates. Selection uses ranks rather than arbitrary acceptance thresholds:

- 10 lowest and 10 highest occupancy annotations;
- 10 smallest and 10 largest boxes;
- 10 boxes closest to an image boundary;
- up to 10 multi-annotation images;
- 10 annotations from images with the most unusual aspect ratios;
- one median-occupancy representative for every class in both train and validation.

The visual package contains 36 unique images: 27 train and 9 validation, with no test images. Twelve mandatory representatives cover all six classes separately in both allowed review splits. Up to four genuine top-ranked images from each diagnostic category were then added. Each PNG shows the raw image, source polygon, prepared box, class, split, filename, and selection reason. The contact sheet and individually inspected low-occupancy, largest-box, boundary, and multi-annotation cases showed correct overlay alignment and no conversion displacement.

This visual review validates technical conversion behavior only. It does not establish that the original polygon is medically correct or that the dataset is clinically suitable.

## Reproducibility

Phase 2A was rerun into isolated temporary directories. The temporary build was compared against the approved prepared dataset and then removed.

- all 1,728 label hashes were identical;
- all 1,728 image hashes were identical;
- empty-label status was identical;
- annotation and class counts were identical;
- the complete prepared-tree fingerprints matched:
  `c5031d9937e2a1d917a2369f327b38a59a4b5e8bd40ae2da659451840c7d41b1`;
- the source fingerprint remained:
  `7dfa20e277473b91b742875bb7dee0e6a1fbd931925071ae3e32e77057675c68`.

The raw and approved prepared dataset fingerprints were unchanged before and after validation.

## Generated evidence

Machine-readable evidence is stored under `outputs/phase2b/v3_detection/`:

- `geometry_validation.csv` — one row per converted annotation;
- `occupancy_statistics.csv` — overall, split, and class summaries;
- `review_candidates.csv` — ranked candidate annotations and reasons;
- `visual_review_index.csv` — selected images, classes, and reasons;
- `visual_review/` — annotated PNG files and contact sheet;
- `reproducibility_comparison.json` — independent rebuild comparison;
- `validation_summary.json` — consolidated Phase 2B result.

## Conclusion and remaining concerns

No conversion, containment, count, class, image-integrity, or reproducibility errors were found. Confidence in the technical conversion result is high because all annotations and all image/label pairs were checked, not sampled.

The prepared v3 detection dataset is technically approved for Phase 2C baseline-protocol work. This approval does not authorize training choices outside the Phase 2 plan and is not medical or clinical validation.

Remaining concerns are inherited from the source dataset rather than introduced by conversion:

- patient or study-level independence cannot be verified from available identifiers;
- source annotation semantics and medical correctness are not established by geometry checks;
- low-occupancy annotations should remain visible during later error analysis, but must not be removed solely because of occupancy.
