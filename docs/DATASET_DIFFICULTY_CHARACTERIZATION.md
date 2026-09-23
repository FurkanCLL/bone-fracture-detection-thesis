# Dataset difficulty characterization (train and validation)

## Purpose and scope

This analysis describes measurable properties of the source-annotated suspected fracture regions in the canonical v3 detection dataset. Its purpose is to assess which properties *may* make localization difficult and to check earlier assumptions against the data. It is descriptive: no model was trained, evaluated, or changed. Only `train` and `valid` images and labels were opened. The held-out `test` images and labels were not opened, decoded, parsed, or summarized.

The measured source is `data/prepared/v3_detection`. Phase 2A converted the v3 polygons to deterministic minimum enclosing axis-aligned boxes, preserving the source images and empty labels; Phase 2B independently verified that conversion ([preparation evidence](evidence/phase2a/preparation_summary.json), [validation report](PHASE_2B_VALIDATION_REPORT.md)). The previously approved **full-dataset** fingerprint is `c5031d9937e2a1d917a2369f327b38a59a4b5e8bd40ae2da659451840c7d41b1`. It is cited from Phase 2B evidence and was **not recomputed** here because that would read held-out files. This analysis instead computed a restricted SHA-256 fingerprint over `data.yaml` plus the 1,559 train/validation images and 1,559 train/validation labels (3,119 files):

`6d630edad24c4a2ebad803ab88bf9298a767c1c3819dc30c35d35a2731979702`

The restricted fingerprint was identical before and after analysis. The PNG-control and CLAHE datasets were checked within these two splits only: each has the same 1,559 image identities, 1,559 matching label-file hashes, and 1,559 matching stored image dimensions as the canonical source. Thus the source box geometry and dimensions also describe those image conditions. Image appearance after CLAHE is outside this geometric characterization. Experiment E uses a different, merged class mapping, so its one-class labels were not used for the six-class analysis.

## Method

The reproducible implementation is [`dataset_difficulty.py`](../src/bone_fracture_pipeline/dataset_difficulty.py); [`dataset_difficulty_figures.py`](../src/bone_fracture_pipeline/dataset_difficulty_figures.py) creates the plots. The script pairs image and label names and rejects missing, ambiguous, or malformed train/validation inputs. A zero-byte label with no annotation rows is counted as an **empty-label image**, not as a verified healthy image. The six original class names and approved train/validation counts are checked against the preparation protocol. The script reads image dimensions from stored files and parses their YOLO detection labels.

For normalized box width (w), height (h), image width (W), and height (H), box area as a fraction of image area is (wh). Original-pixel box dimensions are (wW) and (hH). To approximate the size at `imgsz=640`, the script uses the aspect-ratio-preserving scale (s=\min(640/W,640/H)), giving effective width (wWs), height (hHs), area ((wWs)(hHs)), and shorter side (\min(wWs,hHs)). Padding changes position, not dimensions. This is a geometric estimate at the 640-pixel canvas, not a measurement of the fracture cue visible inside the box or of every augmentation applied during training.

Percentiles use NumPy's linear interpolation. Standard deviations are population standard deviations. Threshold counts use strict `<`; the area and pixel cutoffs are **descriptive**, not universal medical or object-detection definitions. “Approximately square” means width/height within 5% of 1. All detailed distributions, including normalized width, height, area, shorter and longer sides and original/effective pixel dimensions, are in the [generated summary](../outputs/dataset_difficulty/dataset_difficulty_summary.json) and [annotation table](../outputs/dataset_difficulty/annotation_statistics.csv). The [committed evidence summary](evidence/dataset_difficulty/dataset_difficulty_summary.json) preserves the principal numbers.

## Dataset composition

| Split | Images | Positive images | Empty-label images | Positive share | Empty-label share | Boxes | Boxes per positive image: mean / median / min–max / p90–p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Train | 1,211 | 604 | 607 | 49.88% | 50.12% | 698 | 1.156 / 1 / 1–3 / 2–2 |
| Validation | 348 | 173 | 175 | 49.71% | 50.29% | 204 | 1.179 / 1 / 1–4 / 2–2 |

About half of both splits have no annotation rows, and most positive images have one box. This creates a training regime with only 698 source boxes across 604 positive training images, despite 1,211 training images overall. Empty labels cannot by themselves establish that all unannotated images are genuinely fracture-free.

## Image dimensions and heterogeneity

| Quantity | Train mean / median | Train min–max; p5–p95 | Validation mean / median | Validation min–max; p5–p95 |
| --- | ---: | ---: | ---: | ---: |
| Width (px) | 412.24 / 420 | 111–2,300; 201.5–512 | 408.22 / 420 | 130–1,409; 197.35–512 |
| Height (px) | 485.96 / 512 | 134–2,300; 386.5–512 | 484.61 / 512 | 152–1,355; 384.9–512 |
| Image area (MP) | 0.2014 / 0.2079 | 0.0548–5.2900; 0.1016–0.2606 | 0.1972 / 0.2079 | 0.0666–1.9092; 0.0995–0.2599 |
| Aspect ratio (width/height) | 0.884 / 0.820 | 0.217–3.821; 0.394–1.325 | 0.881 / 0.820 | 0.254–3.368; 0.385–1.331 |

There are **359** distinct train resolutions and **157** distinct validation resolutions. Train contains 826 portrait, 300 landscape, and 85 approximately square images; validation contains 238, 88, and 22, respectively. The most common resolution is 420×512 (173 train, 48 validation), followed by 406×512 (148, 32). The resolution count and long aspect-ratio tails demonstrate heterogeneity, though both splits have similar central dimensions and 5th–95th percentile ranges. A few large images extend the full-range axes; they should not be mistaken for the typical image size.

See [image dimensions](figures/dataset_difficulty/image_dimensions.png) and [aspect ratios](figures/dataset_difficulty/aspect_ratio_distribution.png).

## Box geometry

The table summarizes selected distribution statistics; the machine-readable summary has p5, p10, p25, p50, p75, p90, and p95 for every box measure.

| Quantity | Train mean / median / SD | Train p5 / p25 / p75 / p95; min–max | Validation mean / median / SD | Validation p5 / p25 / p75 / p95; min–max |
| --- | --- | --- | --- | --- |
| Normalized box width | 0.1784 / 0.1623 / 0.0897 | 0.0620 / 0.1101 / 0.2265 / 0.3506; 0.0335–0.5885 | 0.1745 / 0.1603 / 0.0808 | 0.0670 / 0.1149 / 0.2262 / 0.3104; 0.0459–0.5064 |
| Normalized box height | 0.1383 / 0.1219 / 0.0745 | 0.0492 / 0.0874 / 0.1715 / 0.2860; 0.0298–0.5077 | 0.1441 / 0.1253 / 0.0849 | 0.0518 / 0.0793 / 0.1722 / 0.3303; 0.0231–0.5236 |
| Box area (% of image) | 2.776 / 1.965 / 2.556 | 0.375 / 1.121 / 3.546 / 8.181; 0.135–19.274 | 2.805 / 1.935 / 2.580 | 0.393 / 1.134 / 3.549 / 8.275; 0.204–16.721 |
| Original box width (px) | 66.56 / 58.57 / 36.62 | 25.95 / 42.46 / 83.56 / 128.53; 14.07–555.91 | 64.12 / 56.61 / 30.22 | 25.77 / 42.99 / 81.85 / 126.26; 17.70–158.49 |
| Original box height (px) | 67.78 / 57.74 / 43.01 | 24.81 / 41.34 / 81.24 / 138.48; 15.24–660.19 | 69.85 / 60.21 / 42.13 | 24.63 / 39.59 / 84.34 / 151.92; 11.83–268.08 |
| Original box area (px²) | 5,783.52 / 3,443.39 / 14,790.30 | 721.34 / 1,897.10 / 6,323.16 / 16,351.73; 291.23–367,005.46 | 5,438.60 / 3,445.78 / 5,730.93 | 661.42 / 1,767.74 / 6,731.30 / 17,174.36; 253.47–36,212.79 |

| Box area strictly below | Train boxes / 698 | Validation boxes / 204 |
| --- | ---: | ---: |
| 0.5% of image | 54 (7.74%) | 18 (8.82%) |
| 1% | 148 (21.20%) | 44 (21.57%) |
| 2% | 359 (51.43%) | 104 (50.98%) |
| 5% | 594 (85.10%) | 173 (84.80%) |
| 10% | 680 (97.42%) | 200 (98.04%) |

The middle box occupies about 2% of its image, while about one fifth of boxes occupy under 1%. The [box-area plot](figures/dataset_difficulty/bbox_area_distribution.png) uses a log x-axis so the low-area tail remains visible. These are **enclosing detection rectangles**, not pixel-accurate fracture areas. The prior Phase 2B audit reported a median polygon-to-enclosing-box occupancy of 0.691 across the full preparation audit. That earlier result warns that box area can overstate the extent of the polygon, and neither shape establishes the extent of a true diagnostic signal.

## Effective box size at `imgsz=640`

| Quantity | Train mean / median / SD | Train p5 / p10 / p25 / p75 / p90 / p95 | Validation mean / median / SD | Validation p5 / p10 / p25 / p75 / p90 / p95 |
| --- | --- | --- | --- | --- |
| Effective width (px) | 82.10 / 73.12 / 38.95 | 32.81 / 38.99 / 53.45 / 103.23 / 137.90 / 159.87 | 79.55 / 70.30 / 36.98 | 32.21 / 38.03 / 53.74 / 102.02 / 138.59 / 156.81 |
| Effective height (px) | 83.50 / 72.02 / 45.53 | 31.01 / 36.68 / 51.77 / 101.03 / 146.60 / 172.29 | 86.77 / 74.58 / 52.33 | 30.79 / 36.45 / 49.48 / 105.03 / 166.48 / 189.91 |
| Effective area (px²) | 8,156.72 / 5,372.41 / 8,541.61 | 1,127.09 / 1,625.26 / 2,973.85 / 9,843.14 / 18,133.10 / 25,167.73 | 8,356.32 / 5,312.31 / 8,788.79 | 1,033.46 / 1,464.83 / 2,762.10 / 10,352.26 / 20,291.69 / 26,573.53 |
| Effective shorter side (px) | 71.46 / 63.77 / 35.59 | 28.93 / 34.01 / 46.77 / 86.35 / 119.98 / 145.85 | 71.09 / 64.05 / 34.67 | 27.68 / 33.33 / 45.80 / 87.99 / 120.68 / 143.44 |

The effective shorter side ranges from **17.59 to 242.96 px** in train and **14.79 to 180.97 px** in validation. These descriptive cutoffs quantify the small tail:

| Effective shorter side strictly below | Train boxes / 698 | Validation boxes / 204 |
| --- | ---: | ---: |
| 8 px | 0 (0.00%) | 0 (0.00%) |
| 16 px | 0 (0.00%) | 1 (0.49%) |
| 32 px | 57 (8.17%) | 18 (8.82%) |
| 64 px | 352 (50.43%) | 101 (49.51%) |

The [effective-size figure](figures/dataset_difficulty/effective_bbox_size_640.png) shows width, height, and cumulative shorter-side distributions. **The typical annotated rectangle is not exceptionally tiny at 640:** its median shorter side is about 64 px, and only about 8% fall below 32 px. Small boxes may be challenging for some examples, but the measurements do not support “all boxes are tiny” or size alone as the established main bottleneck. The visible fracture pattern within an enclosing box may still be finer than the rectangle.

## Six-class characterization and preserved Experiment D evidence

The next table uses the original six classes. A positive-image count means an image has at least one row of that class; an image may contribute to more than one class. The area and shorter-side entries show medians with interquartile ranges (p25–p75).

| Source class | Train / validation boxes | Train / validation positive images | Train area % (p25–p75) | Validation area % (p25–p75) | Train short side px (p25–p75) | Validation short side px (p25–p75) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Elbow positive | 113 / 29 | 102 / 28 | 1.90 (1.16–2.70) | 1.70 (1.11–2.33) | 66.69 (53.35–80.99) | 64.07 (52.38–85.61) |
| Fingers positive | 178 / 48 | 145 / 41 | 1.39 (0.63–2.71) | 2.04 (1.21–3.03) | 46.98 (34.66–66.80) | 49.80 (41.58–68.02) |
| Forearm fracture | 107 / 43 | 96 / 37 | 1.70 (1.13–2.53) | 1.50 (0.86–2.18) | 60.61 (46.87–73.55) | 59.87 (43.51–68.60) |
| Humerus fracture | 104 / 36 | 100 / 31 | 5.18 (2.42–8.15) | 5.55 (3.22–7.71) | 112.48 (77.65–142.13) | 110.92 (94.99–141.70) |
| Shoulder fracture | 120 / 20 | 105 / 19 | 2.09 (1.36–3.57) | 2.37 (1.69–3.96) | 73.82 (59.18–101.03) | 81.49 (67.92–96.34) |
| Wrist positive | 76 / 28 | 56 / 17 | 1.58 (1.08–2.84) | 1.11 (0.62–2.27) | 53.08 (43.15–65.34) | 45.04 (32.80–65.54) |

Median effective **width / height** in pixels at 640 is: elbow 70.99 / 73.87 (train), 70.61 / 70.02 (validation); fingers 52.88 / 54.67, 57.37 / 55.52; forearm 67.99 / 74.39, 62.99 / 64.02; humerus 115.93 / 142.00, 113.42 / 161.13; shoulder 87.45 / 85.68, 87.84 / 93.37; wrist 76.91 / 56.13, 55.22 / 45.04. See the complete [per-class CSV](../outputs/dataset_difficulty/per_class_statistics.csv) and [class-scale figure](figures/dataset_difficulty/class_bbox_scale_comparison.png).

The official D [`BoxPR_curve.png`](../outputs/training/official/D_seed42/BoxPR_curve.png) preserves **per-class AP at IoU 0.50**, rounded to three decimals. Its legend supports the following descriptive comparison. Per-class AP averaged over IoU 0.50–0.95 was not preserved in the available artifacts, so that column is unavailable; it was not recreated by rerunning the model.

| Source class | Train boxes | Validation boxes | Train median area % | Train median short side at 640 (px) | D validation AP50 (plot, rounded) | D validation AP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Elbow positive | 113 | 29 | 1.90 | 66.69 | 0.023 | Unavailable |
| Fingers positive | 178 | 48 | 1.39 | 46.98 | 0.055 | Unavailable |
| Forearm fracture | 107 | 43 | 1.70 | 60.61 | 0.391 | Unavailable |
| Humerus fracture | 104 | 36 | 5.18 | 112.48 | 0.355 | Unavailable |
| Shoulder fracture | 120 | 20 | 2.09 | 73.82 | 0.025 | Unavailable |
| Wrist positive | 76 | 28 | 1.58 | 53.08 | 0.043 | Unavailable |

Humerus has larger enclosing boxes and relatively high AP50, which is consistent with a scale contribution. Forearm has relatively high AP50 without unusually large boxes, while shoulder boxes are relatively large but have low AP50. Fingers has the largest train count but low AP50. Thus the six classes do **not** show a simple monotonic relationship between box size or annotation count and D AP50. The plot-rounded AP50 values, small number of classes, and unavailable per-class AP50–95 make correlation estimates uninformative here. These observations cannot identify why a particular class performs poorly or establish annotation quality.

## Train–validation comparison

| Measure | Train | Validation | Reading |
| --- | ---: | ---: | --- |
| Positive-image share | 49.88% | 49.71% | Nearly identical |
| Median width × height | 420 × 512 px | 420 × 512 px | Identical central dimensions |
| Median image area | 0.2079 MP | 0.2079 MP | Identical |
| Median aspect ratio | 0.8203 | 0.8203 | Identical |
| Median box area | 1.965% | 1.935% | Close |
| Boxes below 1% image area | 21.20% | 21.57% | Close |
| Median effective shorter side | 63.77 px | 64.05 px | Close |
| Boxes below 32 px shorter side | 8.17% | 8.82% | Close |

The [comparison figure](figures/dataset_difficulty/train_validation_comparison.png) also shows class shares. The largest share changes are forearm (15.33% train versus 21.08% validation) and shoulder (17.19% versus 9.80%). These are modest-count class mix differences, especially for shoulder's 20 validation boxes. No obvious large train–validation separation appears in the measured dimensions, positive/background ratios, box areas, or effective scales. This does **not** exclude differences in anatomy, patient/study source, image appearance, label quality, or other unmeasured factors.

## Relation to completed A–E experiments

For context only, the following values come from each preserved official [`results.csv`](../outputs/training/official/) at the saved best epoch selected by overall validation mAP50–95. They are not new evaluations. In particular, a run manifest's final-epoch values can differ from its best-epoch values.

| Experiment | Best epoch | Overall validation mAP50 | Overall validation mAP50–95 |
| --- | ---: | ---: | ---: |
| A | 19 | 0.06633 | 0.02414 |
| B | 18 | 0.06414 | 0.02566 |
| C | 40 | 0.12065 | 0.04163 |
| D | 57 | 0.14861 | 0.04732 |
| E (one merged class) | 27 | 0.14895 | 0.05115 |

The completed experiment record describes strong overfitting in the baseline, substantial validation improvement with conservative augmentation, limited benefit from CLAHE alone, and D as the strongest six-class condition. E's merged-class validation metric improves only modestly over D's overall metric; the one-class and six-class mAP values are **not exactly like-for-like**. Taken together, the experiments do not support a claim that class fragmentation alone explains low validation performance. This analysis does not revise their protocols, checkpoints, or conclusions.

## What the measurements support, and what remains uncertain

**Measured facts.** Positive training material is limited to 604 images and 698 source boxes. Roughly half of both splits have empty labels. The source dimensions vary, including long aspect-ratio tails, but train and validation have close central geometry. The box-area distribution has a small tail, while the median effective box shorter side is about 64 px at 640. Humerus boxes are notably larger; class counts and box sizes vary. The observed D per-class AP50 ordering does not track size or count consistently.

**Reasonable interpretations.** A modest positive sample count, a large empty-label share, and class-dependent representation may jointly make fitting and generalization difficult. The small-box tail plausibly adds difficulty for some examples. Resolution variation means aspect-ratio-preserving scaling changes the number of pixels available for a given annotation. The strong benefit of augmentation is consistent with generalization pressure in this limited training regime. These are compatible with multiple mechanisms and do not isolate a cause.

**Unproven hypotheses and limits.** The analysis cannot establish that any one factor caused poor validation mAP. The boxes are polygon extrema and may include background; Phase 2B's occupancy result illustrates this geometric limitation. Neither source annotations nor empty labels have been clinically adjudicated here. Patient/study independence, acquisition-domain differences, fracture visibility, and annotation consistency are not measured by these tables. The validation split was used for the original model selection; no held-out test inference was performed here. Class-wise AP50–95 could not be recovered reliably from preserved D evidence. Dataset-level performance does not establish clinical suitability.

For the thesis, the analysis provides quantitative context for the learning regime and a reproducible way to describe the dataset without asserting a single bottleneck. The clearest correction to the earlier size hypothesis is that **typical enclosing boxes are not extremely small at the actual 640-pixel training canvas**. The combination of limited positive examples, many empty labels, varied image geometry, a meaningful small-box tail, and class-specific differences remains plausible, but causality would require a separately designed experiment.

## Reproduction and generated artifacts

From the repository root in the configured project environment:

```powershell
.\.venv\Scripts\python.exe -m bone_fracture_pipeline.dataset_difficulty
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

The first command validates canonical train/validation counts, compares PNG-control and CLAHE geometry, checks the restricted source fingerprint before and after analysis, and writes:

- `outputs/dataset_difficulty/dataset_difficulty_summary.json`
- `outputs/dataset_difficulty/image_statistics.csv`
- `outputs/dataset_difficulty/annotation_statistics.csv`
- `outputs/dataset_difficulty/per_class_statistics.csv`
- `docs/evidence/dataset_difficulty/dataset_difficulty_summary.json`
- six PNGs in `docs/figures/dataset_difficulty/`

The `outputs/` tables remain local under the repository's generated-output convention. The compact evidence and figures are committed. No test image or label is needed to reproduce this characterization.
