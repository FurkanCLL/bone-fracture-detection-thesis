# Phase 2 Plan: Dataset Preparation and Experimental Protocol

**Project:** Deep Learning-Based Object Detection Pipeline for Bone Fracture Identification and Estimation in X-Ray Images  
**University:** Riga Technical University (RTU)  
**Main model family:** YOLOv8 object detection  
**Primary raw dataset:** Roboflow/Kaggle Bone Fracture Detection, Version 3 (v3)  
**Status:** Phase 2 planning approved, implementation not yet started  
**Purpose of this document:** Provide a stable, high-level implementation plan for Phase 2 so that the work can proceed in a controlled, reproducible, and consistent way. Detailed implementation choices for each subphase should still be reviewed and finalized immediately before that subphase is implemented.

---

## 1. Phase 2 Objective

Phase 2 prepares the thesis project for scientifically defensible model experiments.

The goal is not only to convert the existing polygon annotations into bounding boxes. Phase 2 must also establish:

- a clean and traceable prepared detection dataset;
- reliable polygon-to-bounding-box conversion;
- extensive conversion validation and quality assurance;
- a fixed baseline training protocol;
- a controlled X-ray preprocessing strategy;
- a controlled augmentation strategy;
- a reproducible A/B/C/D experiment design;
- smoke tests that confirm the full pipeline works before official model experiments begin.

The general workflow is:

```text
v3 raw dataset
    ↓
prepared detection dataset
    ↓
conversion validation and QA
    ↓
baseline training protocol
    ↓
custom preprocessing definition
    ↓
controlled augmentation definition
    ↓
experiment configuration freeze
    ↓
smoke tests
    ↓
Phase 3: official controlled model experiments
```

Phase 2 should prevent later ambiguity about what changed between experiments and why.

---

## 2. Current Dataset Decisions

The following decisions were already made before Phase 2 and should be treated as project assumptions unless new evidence requires reconsideration.

### 2.1 Primary dataset

Roboflow v3 is the primary raw source dataset.

Current v3 counts:

- 1,728 total images;
- 1,211 training images;
- 348 validation images;
- 169 test images;
- 998 total polygon annotations;
- 868 empty label files.

The six classes are:

1. `elbow positive`
2. `fingers positive`
3. `forearm fracture`
4. `humerus fracture`
5. `shoulder fracture`
6. `wrist positive`

The existing train/validation/test membership should be preserved during Phase 2.

### 2.2 Why v3 is used instead of v4

Version 4 was found to contain offline Roboflow augmentation and export inconsistencies.

The main problems identified in v4 included:

- training data expanded largely through offline augmentation rather than independent X-rays;
- class remapping involving `humerus fracture` and `humerus`;
- five known cases where augmented derivatives had empty labels while the corresponding source image was annotated;
- reduced interpretability for controlled augmentation experiments.

Version 3 is therefore a cleaner source for controlled experiments.

### 2.3 Empty-label policy

Empty labels are currently retained and treated as intended negative/background examples.

A reproducible sample of 20 empty-label v3 X-rays was reviewed by a radiologist:

- 16 training images;
- 4 validation images;
- 0 test images.

No visible fracture was identified in any of the 20 reviewed images.

This does not clinically validate all 868 empty-label images. Therefore:

- empty labels must not be described as guaranteed healthy X-rays;
- they may be treated as intended negative/background examples for the working dataset;
- the radiologist review must be described as a limited sanity check;
- subtle missed fractures remain a documented dataset limitation.

### 2.4 Annotation format

The v3 annotations are YOLO segmentation polygons.

The thesis remains an **object-detection thesis**, not a segmentation thesis.

The polygons appear to represent approximate fracture regions rather than precise pixel-level masks. This makes a carefully implemented polygon-to-bounding-box conversion appropriate for the thesis.

Raw annotations must never be overwritten.

---

# 3. Phase 2 Structure

Phase 2 is divided into six controlled subphases:

- **Phase 2A:** Prepared Detection Dataset
- **Phase 2B:** Conversion Validation and Quality Assurance
- **Phase 2C:** Baseline Training Protocol
- **Phase 2D:** Custom X-ray Preprocessing Pipeline
- **Phase 2E:** Controlled Augmentation Policy
- **Phase 2F:** Experiment Freeze and Smoke Testing

Each subphase should be discussed and finalized before implementation begins.

---

# 4. Phase 2A: Prepared Detection Dataset

## 4.1 Goal

Create a new derived YOLOv8 detection dataset from v3 without modifying the raw dataset.

The prepared dataset must preserve:

- image identity;
- split membership;
- class mapping;
- empty-label files;
- annotation count;
- traceability back to the original source.

Only the label representation changes from polygon regions to standard detection bounding boxes.

---

## 4.2 Raw-data immutability

The original v3 dataset must remain read-only from the project's perspective.

No Phase 2 script may:

- overwrite raw images;
- overwrite raw labels;
- rename raw files;
- move raw files;
- change raw class mappings;
- repair raw labels in place.

All generated content must be written to a separate derived location.

Example structure:

```text
data/
├── raw/
│   └── bone-fracture-detection/
│       └── bone-fracture-detection-v3-yolov8/
│
└── prepared/
    └── v3_detection/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        ├── data.yaml
        └── manifest.json
```

The exact directory naming should be finalized immediately before Phase 2A implementation.

---

## 4.3 Polygon-to-bounding-box conversion

Each valid source polygon should be converted to the **smallest axis-aligned bounding box that fully contains the polygon**.

For polygon coordinates:

```text
(x1, y1), (x2, y2), ..., (xn, yn)
```

calculate:

```text
xmin = minimum polygon x
xmax = maximum polygon x
ymin = minimum polygon y
ymax = maximum polygon y
```

Then convert to normalized YOLO detection format:

```text
x_center = (xmin + xmax) / 2
y_center = (ymin + ymax) / 2
width    = xmax - xmin
height   = ymax - ymin
```

Output format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

---

## 4.4 No artificial padding

No arbitrary padding should be added around generated boxes.

For example, the project should not automatically enlarge boxes by 5%, 10%, or another manually chosen margin.

Reason:

- the original polygon already represents the annotated suspected region;
- artificial enlargement would introduce a subjective transformation;
- the minimum enclosing rectangle preserves the original annotation as closely as standard axis-aligned detection allows.

Some additional background inside the rectangular box is unavoidable for irregular or rotated polygons. This is an inherent consequence of converting polygon regions to axis-aligned object-detection boxes.

---

## 4.5 Precision and determinism

The conversion must be deterministic.

Requirements:

- identical raw input must always generate identical output;
- floating-point serialization must be consistent;
- no random operation should occur during conversion;
- file order must not affect output;
- conversion results should be reproducible across repeated runs.

A fixed numeric output precision should be selected during detailed Phase 2A planning.

---

## 4.6 Images

The initial prepared detection dataset should preserve the original image content.

The safest default is to copy or reference the original v3 images without changing their pixels.

The custom preprocessing dataset is a separate Phase 2D product and should not be mixed into the base detection dataset.

---

## 4.7 Empty labels

Empty source labels must remain empty in the prepared detection dataset.

The converter must not:

- delete them;
- create artificial boxes;
- replace them with placeholder values;
- exclude the corresponding images.

They remain negative/background training examples.

---

## 4.8 Traceability

Every generated label should be traceable to its original raw label.

A machine-readable manifest should record at least:

- split;
- image filename;
- raw image relative path;
- raw label relative path;
- prepared image relative path;
- prepared label relative path;
- source annotation count;
- generated annotation count;
- empty/non-empty status;
- source file hashes where practical;
- generated label hash;
- conversion version or script version.

The final schema should be confirmed during Phase 2A planning.

---

# 5. Phase 2B: Conversion Validation and Quality Assurance

## 5.1 Goal

Treat label conversion as a scientific data-preparation procedure, not as a simple file-format operation.

Phase 2B should verify that the prepared detection dataset is:

- mathematically correct;
- structurally consistent;
- visually reasonable;
- deterministic;
- traceable;
- safe for training.

---

## 5.2 Validation layer 1: Unit tests

The conversion logic should have focused automated tests.

At minimum, test cases should cover:

- standard rectangular polygon;
- triangle;
- irregular polygon;
- rotated quadrilateral;
- polygon near image boundary;
- multiple annotations in one label file;
- multiple classes in one label file;
- empty label;
- invalid coordinate input;
- degenerate polygon;
- unexpected class ID;
- deterministic output;
- stable numeric serialization.

Known synthetic polygons should have explicitly known expected boxes.

---

## 5.3 Validation layer 2: Dataset-wide invariants

The prepared dataset should preserve expected v3 statistics.

Expected totals:

```text
Images:            1,728
Labels:            1,728
Annotations:         998
Empty labels:        868
Train images:      1,211
Validation images:   348
Test images:         169
Classes:               6
```

The class annotation counts should remain unchanged.

The conversion should never silently drop or duplicate annotations.

For every non-empty source label:

```text
source annotation count == generated detection annotation count
```

unless the validator intentionally rejects an invalid annotation and reports it. Because the existing v3 audit found no malformed annotations, the expected normal outcome is exact preservation.

---

## 5.4 Validation layer 3: Geometry checks

Each generated bounding box must satisfy:

```text
0 <= x_center <= 1
0 <= y_center <= 1
0 < width <= 1
0 < height <= 1
```

The box must remain inside normalized image bounds.

Most importantly:

**every original polygon vertex must lie inside or exactly on the generated bounding box.**

This should be verified programmatically for all 998 annotations.

---

## 5.5 Polygon occupancy ratio

A useful conversion-quality diagnostic should be calculated:

```text
occupancy_ratio = polygon_area / bounding_box_area
```

Interpretation:

- high ratio: bounding box closely follows the polygon region;
- low ratio: conversion introduces relatively more background around the polygon.

This is not a criterion for deleting an annotation.

Its purpose is to:

- quantify information expansion caused by polygon-to-box conversion;
- identify unusual cases for manual review;
- support transparent thesis reporting.

Summary statistics may include:

- minimum;
- median;
- mean;
- quartiles;
- maximum;
- per-class distribution.

The exact statistics to report should be finalized during Phase 2B.

---

## 5.6 Visual quality assurance

A visual review set should be generated.

Each review image should display both:

- the original polygon;
- the generated bounding box.

The review sample should not be purely random.

It should deliberately include:

- examples from every class;
- smallest boxes;
- largest boxes;
- lowest occupancy ratios;
- highest occupancy ratios if useful;
- images with multiple annotations;
- unusual image aspect ratios;
- annotations close to image borders.

The visual review should answer:

> Does the generated bounding box represent the original annotated fracture region reasonably and without unexpected expansion or displacement?

No official training should begin until the conversion passes this visual review.

---

## 5.7 Reproducibility check

The prepared dataset generation should be run twice independently.

Generated labels should be hashed.

The two runs must produce identical hashes for every generated label file.

If they do not, Phase 2A is not considered reproducible and official training must not begin.

---

## 5.8 Phase 2A/2B exit condition

Phase 2A and 2B are complete only when:

- raw v3 is unchanged;
- all prepared files are traceable;
- all expected images are present;
- all expected annotations are preserved;
- empty labels are preserved;
- all boxes are valid;
- every source polygon is contained by its box;
- unit tests pass;
- dataset-wide validation passes;
- visual QA is accepted;
- repeated conversion produces identical outputs.

---

# 6. Phase 2C: Baseline Training Protocol

## 6.1 Purpose

The baseline protocol defines the training variables that remain fixed during the controlled experiments.

The purpose is experimental fairness.

If preprocessing or augmentation is being evaluated, unrelated variables must not change between experiments.

The principle is:

```text
Change only the factor being tested.
Keep all unrelated training conditions constant.
```

---

## 6.2 Frozen baseline configuration

Phase 2C validated and froze the following configuration:

| Parameter | Frozen value |
|---|---|
| Model | YOLOv8s |
| Initialization | COCO pretrained weights |
| Image size | 640 |
| Epoch budget | 100 |
| Batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0005 |
| Primary experimental seed | 42 |
| Deterministic mode | Enabled where supported |
| Mixed precision / AMP | Enabled |
| Primary model-selection source | Validation set |
| Primary metric | mAP50-95 |
| Additional metrics | mAP50, Precision, Recall, per-class AP |

The software environment, framework semantics, prepared-dataset fingerprint, and pretrained weights were validated in Phase 2C. Batch 8 remains the fixed planned value, but its VRAM stability must still be confirmed by the Phase 2F smoke check before official training.

The goal is stability and reproducibility, not aggressive hyperparameter optimization.

---

## 6.3 Why YOLOv8s

YOLOv8s is selected as a practical middle ground.

It provides:

- more capacity than the smallest YOLOv8n model;
- substantially lower computational cost than larger YOLOv8 variants;
- practical training time for multiple controlled experiments;
- suitability for the available hardware.

The project should avoid changing YOLOv8 model size between A/B/C/D experiments.

---

## 6.4 Why 640 input size

An image size of 640 is a practical baseline because:

- it is standard for YOLO training;
- it limits GPU memory use;
- the dataset contains strongly varying image dimensions and aspect ratios;
- it provides a reproducible starting point without adding image resolution as another research variable.

Model input preparation may use aspect-ratio-preserving letterboxing as required by YOLO.

A later resolution experiment may be considered only as a separate experiment if time and thesis scope justify it.

It should not be mixed into the main A/B/C/D comparison.

---

## 6.5 Training budget and checkpoint selection

Official A/B/C/D comparisons should use the same maximum training budget.

The main comparison should not intentionally give one experiment substantially more training opportunity than another.

The selected checkpoint is the checkpoint with maximum validation mAP50-95. Under pinned Ultralytics 8.4.155, detection fitness is exactly mAP50-95, so `best.pt` implements this rule. The Phase 2C validation tool checks that equivalence and must fail rather than silently accept changed framework semantics.

---

## 6.6 Seed policy

The first development and progress-report runs may use a single fixed seed:

```text
42
```

For final thesis results, the preferred stronger design is to repeat each experiment using multiple seeds if computational time allows.

Candidate final seeds:

```text
42
43
44
```

This would produce:

```text
4 experiment conditions × 3 seeds = 12 official runs
```

Results could then be reported using mean and standard deviation.

This multi-seed plan is desirable but should remain conditional on available time and compute resources.

---

## 6.7 Test-set isolation

The test set must not be used to decide:

- preprocessing settings;
- augmentation settings;
- model size;
- epochs;
- optimizer;
- learning rate;
- checkpoint selection;
- experimental configuration.

The test set is reserved for final evaluation after model-selection decisions are complete.

Validation data is used during development and comparison.

---

# 7. Phase 2D: Custom X-ray Preprocessing Pipeline

## 7.1 Goal

Evaluate whether a simple, medically reasonable image-enhancement pipeline improves fracture-region detection.

The preprocessing should remain deliberately conservative.

The goal is not to maximize the number of image-processing operations.

The goal is to test one clearly defined and defensible preprocessing strategy.

---

## 7.2 Planned preprocessing

The main custom preprocessing condition should consist of:

1. grayscale standardization;
2. CLAHE local contrast enhancement;
3. conversion back to a three-channel representation for the pretrained YOLO model;
4. lossless storage of the derived images.

---

## 7.3 Grayscale standardization

Each image should be loaded and standardized as grayscale.

Although the X-rays are already visually grayscale, this creates a consistent image representation and removes unnecessary color-channel differences.

This operation should not resize, crop, rotate, or geometrically alter the image.

Therefore annotation coordinates remain unchanged.

---

## 7.4 CLAHE

The planned main enhancement technique is:

**Contrast Limited Adaptive Histogram Equalization (CLAHE)**

Initial proposed parameters:

```text
clipLimit = 2.0
tileGridSize = 8 × 8
```

These should be reviewed visually before the preprocessing protocol is frozen.

Rationale:

- fracture regions may contain subtle local contrast differences;
- CLAHE can improve local contrast;
- limiting contrast enhancement reduces the risk of excessive noise amplification;
- CLAHE is simple, reproducible, explainable, and suitable for a controlled thesis experiment.

The thesis must not assume CLAHE will improve performance.

The experiment is specifically intended to determine whether it helps, has negligible effect, or harms performance on this dataset.

---

## 7.5 Return to three channels

After grayscale processing, the same grayscale channel should be replicated three times:

```text
gray → [gray, gray, gray]
```

This preserves compatibility with pretrained YOLOv8 weights that expect three-channel input.

---

## 7.6 Lossless derived images

Custom-preprocessed images should preferably be saved using a lossless format such as PNG.

Reason:

- original source images are JPEG;
- saving processed images again as JPEG would introduce another lossy compression stage;
- re-compression may create artifacts that become an uncontrolled variable.

Therefore the intended path is:

```text
raw JPEG
    ↓
grayscale standardization
    ↓
CLAHE
    ↓
3-channel grayscale representation
    ↓
lossless PNG
```

---

## 7.7 What should not be included in the main preprocessing pipeline

The main controlled preprocessing condition should initially exclude:

- denoising;
- sharpening;
- edge detection;
- gamma correction;
- cropping;
- histogram-equalization chains;
- heavy filtering;
- learned enhancement models.

Reasons:

- denoising may remove subtle fracture information;
- sharpening may amplify artificial edges;
- complex pipelines make causal interpretation harder;
- each additional operation adds another experimental variable.

If later evidence strongly justifies another operation, it should be tested separately rather than silently added to the main preprocessing condition.

---

## 7.8 Base and preprocessed datasets

The project should conceptually maintain two image conditions:

```text
base detection images
custom preprocessed detection images
```

For example:

```text
v3_detection_original/
v3_detection_clahe/
```

Exact directory naming should be decided during Phase 2D planning.

Annotations must remain geometrically identical because the custom preprocessing does not alter image dimensions or spatial geometry.

---

# 8. Phase 2E: Controlled Augmentation Policy

## 8.1 Goal

Evaluate whether conservative, explicitly controlled training-time augmentation improves generalization.

The project should use **on-the-fly augmentation**, not an offline expansion that permanently creates multiple transformed copies of each X-ray.

---

## 8.2 Why on-the-fly augmentation

Version 4 used offline augmentation with:

```text
Outputs per training example: 3
Rotation: -15° to +15°
Exposure: -25% to +25%
```

This contributed to several methodological problems:

- nominal training size no longer represented independent radiographs;
- augmented derivatives existed as permanent files;
- provenance became more complicated;
- export inconsistencies were introduced;
- augmentation could not be cleanly separated from the baseline.

Phase 2 should avoid repeating that structure.

Instead, the training loader should apply controlled transformations dynamically.

A single image may therefore appear differently across epochs without creating additional permanent source images.

---

## 8.3 Proposed conservative augmentation

The initial planned augmentation condition is:

```text
Rotation:        approximately ±10°
Translation:     approximately ±5%
Scale:           approximately 0.90–1.10
Intensity/value: approximately ±15%
```

The exact implementation and parameter mapping must be verified against the training framework before Phase 2E is frozen.

The main principle is conservative medical-image variability rather than aggressive generic computer-vision augmentation.

---

## 8.4 Transformations planned to remain disabled

The main experiment should disable:

```text
Horizontal flip
Vertical flip
Mosaic
MixUp
CutMix
Shear
Perspective
Copy-paste
```

### Flip rationale

X-rays may contain:

- R/L laterality markers;
- orientation-specific text;
- acquisition markers.

Flipping can create physically unrealistic marker placement.

Therefore flips are excluded from the primary controlled augmentation policy.

### Mosaic / MixUp / CutMix rationale

These methods can create highly artificial composite X-rays that do not represent realistic radiographic acquisition.

The main experiment should prioritize domain realism and interpretability.

### Perspective / shear rationale

Strong geometric transformations may produce unrealistic bone geometry or distort fracture appearance.

They are unnecessary for the initial controlled experiment.

---

## 8.5 Critical baseline requirement

The "no augmentation" experiments must explicitly disable augmentation.

It is not sufficient to simply omit a custom augmentation configuration.

The actual YOLO training framework may have default augmentation behavior.

Therefore Phase 2E must inspect the installed training configuration and explicitly set unwanted augmentation parameters to zero/off for conditions A and B.

Likewise, any optional library-driven augmentation must be identified and controlled.

The baseline must truly represent:

```text
custom augmentation = OFF
```

---

# 9. Phase 2F: Experiment Freeze and Smoke Testing

## 9.1 A/B/C/D experiment matrix

The main thesis experiment is a 2 × 2 design:

| Experiment | Custom preprocessing | Controlled augmentation |
|---|---|---|
| A | No | No |
| B | Yes | No |
| C | No | Yes |
| D | Yes | Yes |

Interpretation:

### Experiment A
Baseline detection pipeline.

Tests YOLOv8 using the prepared detection dataset without custom X-ray preprocessing and without controlled augmentation.

### Experiment B
Preprocessing only.

Measures the effect of the custom grayscale + CLAHE preprocessing pipeline.

### Experiment C
Augmentation only.

Measures the effect of conservative controlled on-the-fly augmentation.

### Experiment D
Preprocessing + augmentation.

Measures their combined effect.

---

## 9.2 Variables that must remain fixed

Across A/B/C/D, keep fixed:

- raw dataset source;
- prepared split membership;
- class mapping;
- empty-label policy;
- bounding-box conversion;
- model architecture;
- pretrained initialization;
- image size;
- optimizer;
- learning rate;
- weight decay;
- epoch budget;
- batch size;
- checkpoint-selection rule;
- evaluation metrics;
- seed policy;
- test-set isolation policy.

The only planned experimental factors are:

```text
Factor 1: custom preprocessing
Factor 2: controlled augmentation
```

This enables meaningful causal interpretation.

---

## 9.3 Configuration files

The experiment definitions should be stored as explicit, version-controlled configuration files.

Possible structure:

```text
configs/
├── baseline.yaml
├── preprocessing.yaml
├── augmentation.yaml
└── preprocessing_augmentation.yaml
```

Common settings should ideally come from a shared base configuration to reduce accidental divergence.

The exact configuration architecture should be finalized during Phase 2F.

---

## 9.4 Environment freeze

Before official experiments, record the environment used for training.

At minimum:

- Python version;
- PyTorch version;
- Ultralytics version;
- CUDA version;
- GPU model;
- OpenCV version;
- NumPy version;
- relevant package versions;
- operating system if useful.

A dependency lock or reproducible environment description should be retained with the project.

---

## 9.5 Smoke tests

Before official experiments, run short technical smoke tests.

These are **not thesis results**.

Their purpose is to verify the pipeline.

Recommended smoke-test length:

```text
1–2 epochs
```

Smoke tests should confirm:

- prepared dataset loads correctly;
- all six classes are recognized;
- empty labels are accepted;
- bounding boxes appear correctly during training visualization;
- GPU training works;
- loss values are finite;
- metrics are generated;
- checkpoints are saved;
- inference runs;
- preprocessing images load correctly;
- augmentation behaves as intended;
- disabled augmentation is truly disabled in baseline conditions;
- transformed bounding boxes remain valid;
- experiment configuration files are actually being applied.

If possible, save visualized training batches for manual inspection.

Official Experiment A should not begin until smoke tests pass.

---

# 10. Standard Input Preparation vs Custom Preprocessing

The thesis should clearly distinguish two concepts.

## 10.1 Standard model input preparation

Operations required by the model or training framework, for example:

- letterbox / resize;
- tensor conversion;
- numerical normalization;
- batching.

These occur in all experiments and are not treated as the thesis's custom preprocessing variable.

## 10.2 Custom X-ray preprocessing

The experimental preprocessing condition is:

```text
grayscale standardization
+
CLAHE
```

This distinction prevents ambiguity when describing "preprocessing on/off."

---

# 11. Metrics and Evaluation Strategy

The main evaluation metrics should include:

- mAP50-95;
- mAP50;
- Precision;
- Recall;
- per-class Average Precision;
- training and validation losses where useful.

Additional error analysis may later include:

- false positives;
- false negatives;
- class-specific weaknesses;
- localization errors;
- behavior on empty-label/background images;
- difficult anatomical regions.

Exact final evaluation reporting belongs primarily to Phase 3, but metric definitions must be frozen in Phase 2C/F.

---

# 12. Documentation Requirements

Every Phase 2 decision should be documented close to implementation time.

The project should maintain a decision log containing:

- decision;
- reason;
- date;
- alternative options considered;
- expected methodological effect;
- relevant script/config version.

Important Phase 2 decisions include:

- prepared dataset directory structure;
- label serialization precision;
- manifest schema;
- visual QA sample policy;
- baseline model configuration;
- CLAHE settings;
- augmentation settings;
- seed policy;
- checkpoint-selection rule;
- metric definitions;
- software environment.

The purpose is to prevent undocumented configuration drift.

---

# 13. Phase 2 Guardrails

The following rules should be treated as non-negotiable unless deliberately reconsidered and documented.

### Raw-data rules

- Do not modify raw v3 files.
- Do not silently repair labels in place.
- Do not treat v4 as the main training dataset.
- Do not treat the redundant v4 copy as independent data.

### Split rules

- Preserve current train/validation/test membership during Phase 2.
- Do not use the test set for tuning or protocol decisions.

### Annotation rules

- Use the smallest axis-aligned box that contains each polygon.
- Do not add arbitrary bounding-box padding.
- Preserve class IDs.
- Preserve empty labels.
- Validate every generated annotation.

### Preprocessing rules

- Keep the main custom preprocessing simple and reproducible.
- Do not add extra filters without a documented reason.
- Do not change image geometry in the main CLAHE preprocessing condition.

### Augmentation rules

- Use on-the-fly augmentation.
- Explicitly disable augmentation in no-augmentation conditions.
- Avoid unrealistic medical-image composites.
- Keep transformations conservative.

### Experimental rules

- Change only preprocessing and augmentation in the primary A/B/C/D comparison.
- Keep all unrelated variables fixed.
- Record the full training environment.
- Run smoke tests before official experiments.

---

# 14. Phase 2 Completion Criteria

Phase 2 is complete when all of the following are true:

## Dataset preparation

- prepared v3 detection dataset exists;
- raw v3 remains unchanged;
- polygon annotations have been deterministically converted;
- empty labels are preserved;
- split membership is unchanged;
- six-class mapping is unchanged;
- generated files are traceable.

## Validation

- unit tests pass;
- all dataset-wide invariants pass;
- all boxes are within valid normalized coordinates;
- every polygon is fully contained by its generated box;
- occupancy statistics are generated;
- visual QA is completed;
- repeated conversion produces identical results.

## Training protocol

- baseline YOLOv8 configuration is finalized;
- training budget is fixed;
- optimizer and learning-rate policy are fixed;
- checkpoint selection is fixed;
- seed policy is fixed;
- evaluation metrics are fixed;
- test-set policy is documented.

## Preprocessing

- grayscale + CLAHE pipeline is finalized;
- CLAHE parameters are fixed;
- processed-image storage format is fixed;
- preprocessing output is visually reviewed;
- image geometry remains unchanged.

## Augmentation

- on-the-fly augmentation configuration is finalized;
- augmentation-off configuration is explicitly verified;
- unrealistic transformations are disabled;
- transformed training samples are visually reviewed.

## Experiment protocol

- A/B/C/D configuration files exist;
- environment versions are recorded;
- smoke tests pass for relevant configurations;
- no official results have been contaminated by test-set tuning.

At this point the project can move to:

# Phase 3: Controlled Model Experiments

---

# 15. Expected Phase 3 Transition

After Phase 2 is frozen, Phase 3 should proceed in the following logical order:

```text
Experiment A
    ↓
Experiment B
    ↓
Experiment C
    ↓
Experiment D
    ↓
validation comparison
    ↓
final model-selection decision
    ↓
locked final test evaluation
    ↓
error analysis
    ↓
thesis results and discussion
```

If multi-seed evaluation is adopted, the official experiment matrix should be repeated under the selected seed policy.

---

# 16. Known Limitations That Remain After Phase 2

Phase 2 does not solve every uncertainty in the source dataset.

The following limitations remain and should be documented in the thesis:

- patient/study identifiers are unavailable;
- patient-level split independence cannot be verified;
- clinical naming policy behind labels such as `positive` and `fracture` is not fully documented;
- radiologist review covered only 20 empty-label images;
- subtle fractures may still exist in some empty-label images;
- JPEG compression and image resolution may limit fracture visibility;
- polygon labels are approximate regions rather than precise fracture masks;
- polygon-to-box conversion necessarily includes some additional background.

These limitations do not currently block the planned thesis methodology.

---

# 17. Working Principle for Phase 2

Phase 2 should be implemented conservatively.

The priority order is:

```text
correctness
→ traceability
→ reproducibility
→ experimental fairness
→ simplicity
→ performance optimization
```

The project should avoid adding complexity unless it solves a clearly identified problem.

A simple, fully controlled experiment is more valuable for the thesis than a complicated pipeline whose results cannot be clearly interpreted.

---

# 18. Final Approved High-Level Design

The current approved Phase 2 design is:

```text
PHASE 2A
Raw v3
→ prepared detection dataset
→ exact minimum polygon-enclosing bounding boxes

PHASE 2B
Automated validation
+ unit tests
+ containment checks
+ occupancy analysis
+ visual QA
+ reproducibility checks

PHASE 2C
YOLOv8s baseline protocol
+ fixed training settings
+ validation-based checkpoint selection
+ strict test-set isolation

PHASE 2D
Custom preprocessing:
grayscale standardization
+ conservative CLAHE
+ three-channel output
+ lossless derived images

PHASE 2E
Controlled on-the-fly augmentation:
small rotation
+ small translation
+ small scale variation
+ conservative intensity variation
with flips, mosaic, MixUp, CutMix, shear,
perspective and copy-paste disabled

PHASE 2F
A/B/C/D configuration freeze
+ environment freeze
+ smoke tests
+ verification that experimental conditions
behave exactly as intended

THEN:
PHASE 3
Official controlled YOLOv8 experiments
```

---

## Important implementation note

This document defines the approved **high-level Phase 2 strategy**.

It should not be interpreted as permission to implement all subphases immediately without review.

Before each subphase begins:

1. inspect the current project state;
2. review the relevant section of this document;
3. finalize the low-level technical details;
4. document the decision;
5. implement;
6. test;
7. verify the result;
8. only then continue to the next subphase.

This staged process is intended to keep the thesis implementation planned, controlled, reproducible, and internally consistent.
