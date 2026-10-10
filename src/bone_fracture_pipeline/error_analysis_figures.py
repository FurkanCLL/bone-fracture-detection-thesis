"""Static research figures built only from Stage 2 numeric evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from bone_fracture_pipeline.error_analysis_quantitative import CATEGORIES, validate_numerical_tree
from bone_fracture_pipeline.prepare_dataset import CLASS_NAMES


BLUE = "#0072B2"
ORANGE = "#D55E00"
CATEGORY_LABELS = {"complete_miss": "No retained nearby candidate", "localization_failure": "Localization failure",
                   "low_confidence_detection": "Low-confidence detection", "classification_error": "Classification error",
                   "matching_conflict": "Matching conflict", "mixed_or_unresolved": "Mixed or unresolved"}


def _style_axis(ax, *, rate=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D8DDE2", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if rate:
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter(100))


def _save(fig, directory: Path, filename: str, note: str) -> Path:
    fig.text(0.01, 0.015, note, fontsize=9, color="#444444", va="bottom")
    fig.tight_layout(rect=(0, 0.085, 1, 0.95))
    path = directory / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white",
                metadata={"Software": "Bone fracture pipeline: Stage 2"})
    plt.close(fig)
    return path


def _iou_recall(result, directory):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
    rows = result["analysis_a_iou_sensitivity"]
    for ax, confidence in zip(axes, (0.05, 0.25)):
        for aware, color, marker, line in ((True, BLUE, "o", "-"), (False, ORANGE, "s", "--")):
            subset = sorted((row for row in rows if row["confidence"] == confidence and row["class_aware"] == aware), key=lambda row: row["iou"])
            ax.plot([row["iou"] for row in subset], [100 * row["recall"] for row in subset], color=color, marker=marker,
                    linestyle=line, linewidth=1.8, label="Class-aware" if aware else "Class-agnostic")
        ax.set(title=f"Confidence ≥ {confidence:.2f}", xlabel="Matching IoU threshold", xticks=(0.1, 0.3, 0.5))
        _style_axis(ax, rate=True)
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("Matched ground truth / all ground truth")
    fig.suptitle("Validation recall across IoU thresholds", fontsize=14)
    return _save(fig, directory, "iou_recall.png", "348 validation images; 204 GT boxes. IoU 0.10 is permissive approximate overlap, not clinically adequate localization.")


def _confidence_curves(result, directory):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    rows = result["analysis_b_confidence_sensitivity"]["operating_points"]
    for ax, metric in zip(axes, ("precision", "recall", "f1")):
        for aware, color, marker, line in ((True, BLUE, "o", "-"), (False, ORANGE, "s", "--")):
            subset = sorted((row for row in rows if row["class_aware"] == aware), key=lambda row: row["confidence"])
            ax.plot([row["confidence"] for row in subset], [100 * row[metric] for row in subset], color=color, marker=marker,
                    linestyle=line, linewidth=1.8, label="Class-aware" if aware else "Class-agnostic")
        ax.set_xscale("log")
        ax.set_xticks((0.01, 0.05, 0.1, 0.25, 0.5), (".01", ".05", ".10", ".25", ".50"))
        ax.minorticks_off()
        ax.set(title="F1" if metric == "f1" else metric.title(), xlabel="Confidence threshold (log scale)")
        _style_axis(ax, rate=True)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Validation fixed-threshold metrics at IoU ≥ 0.50", fontsize=14)
    return _save(fig, directory, "confidence_sensitivity.png", "Same retained post-NMS candidates at every threshold. Diagnostic comparisons only; no deployment threshold is selected.")


def _fn_categories(result, directory):
    summary = result["analysis_c_fn_taxonomy"]
    values = [summary["primary_categories"][category]["count"] for category in CATEGORIES]
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    positions = np.arange(len(CATEGORIES))
    ax.barh(positions, values, color=BLUE, height=0.62)
    ax.set_yticks(positions, [CATEGORY_LABELS[category] for category in CATEGORIES])
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(values, default=0) * 1.38, 1))
    for index, (category, count) in enumerate(zip(CATEGORIES, values)):
        share = summary["primary_categories"][category]["percentage_of_fn"]
        label = f"{count} ({share:.1f}%)" if share is not None else str(count)
        ax.text(count + 0.7, index, label, va="center", fontsize=10)
    ax.set_xlabel("False-negative annotations")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#D8DDE2", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.suptitle(f"Validation FN primary categories (n={summary['total_false_negatives']})", fontsize=14)
    return _save(fig, directory, "fn_categories.png", "Confidence ≥ 0.25; IoU ≥ 0.50; class-aware. Categories are exclusive; nonexclusive evidence flags remain in the JSON/JSONL.")


def _confusion(result, directory):
    summary = result["analysis_d_classes"]["class_confusion"]
    matrix = np.asarray(summary["matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(9.7, 6.7))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(int(matrix.max()), 1))
    labels = [name.replace(" ", "\n", 1).title() for name in CLASS_NAMES]
    ax.set_xticks(range(6), labels)
    ax.set_yticks(range(6), labels)
    ax.set(xlabel="Predicted class", ylabel="Ground-truth class")
    for row in range(6):
        for column in range(6):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=11,
                    color="white" if matrix[row, column] > matrix.max() * 0.55 else "#202020")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="Matched annotations")
    fig.suptitle(f"Geometry-matched class confusion (n={summary['geometric_matches']})", fontsize=14)
    return _save(fig, directory, "class_confusion.png", "Class-agnostic assignment; confidence ≥ 0.25 and IoU ≥ 0.50. Unmatched GT/predictions are outside this matrix.")


def _false_positives(result, directory):
    summary = result["analysis_e_false_positives"]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    positions = np.arange(6)
    width = 0.32
    for offset, group, color, hatch in ((-width / 2, "annotated", BLUE, None), (width / 2, "empty_label", ORANGE, "//")):
        values = summary["groups"][group]
        counts = [row["fp"] for row in values["by_predicted_class"]]
        label = f"Annotated images (n={values['images']})" if group == "annotated" else f"Empty-label images (n={values['images']})"
        ax.barh(positions + offset, counts, height=width, color=color, hatch=hatch, label=label)
        for position, count in zip(positions + offset, counts):
            ax.text(count + 0.4, position, str(count), va="center", fontsize=9)
    ax.set_yticks(positions, CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlabel("Annotation-based false-positive predictions")
    ax.set_xlim(0, max(row["fp"] for group in summary["groups"].values() for row in group["by_predicted_class"]) * 1.18 + 1)
    ax.grid(axis="x", color="#D8DDE2", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle(f"Validation FP distribution (total={summary['total_fp']})", fontsize=14)
    return _save(fig, directory, "false_positives.png", "Confidence ≥ 0.25; class-aware IoU ≥ 0.50. Empty labels do not establish medically confirmed negative X-rays.")


def _generalization(result, directory):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1.5]})
    reference = {"train": result["training_reference"], "valid": result["validation_reference"]}
    for offset, split, color, hatch in ((-0.18, "train", BLUE, None), (0.18, "valid", ORANGE, "//")):
        values = [100 * reference[split][key] for key in ("precision", "recall", "f1")]
        axes[0].bar(np.arange(3) + offset, values, width=0.34, color=color, hatch=hatch, label="Train" if split == "train" else "Validation")
        for position, value in zip(np.arange(3) + offset, values):
            axes[0].text(position, value + 1.2, f"{value:.1f}%", ha="center", fontsize=9)
    axes[0].set_xticks(range(3), ("Precision", "Recall", "F1"))
    axes[0].set_title("Overall reference metrics")
    _style_axis(axes[0], rate=True)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    train = result["analysis_f_generalization"]["train_class_rows"]
    valid = result["analysis_d_classes"]["class_rows"]
    for offset, rows, color, hatch in ((-0.18, train, BLUE, None), (0.18, valid, ORANGE, "//")):
        axes[1].barh(np.arange(6) + offset, [100 * row["recall"] for row in rows], height=0.34, color=color, hatch=hatch)
    axes[1].set_yticks(range(6), [f"{name}\nGT n={train[index]['support']}/{valid[index]['support']}" for index, name in enumerate(CLASS_NAMES)])
    axes[1].invert_yaxis()
    axes[1].set(xlim=(0, 100), xlabel="Matched GT / all class GT", title="Per-class recall; support is train/validation")
    axes[1].xaxis.set_major_formatter(PercentFormatter(100))
    axes[1].grid(axis="x", color="#D8DDE2", linewidth=0.6)
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Preserved D checkpoint: train versus validation", fontsize=14)
    return _save(fig, directory, "train_validation.png", "Same reference matcher: confidence ≥ 0.25, IoU ≥ 0.50, class-aware. Inference only; no augmentation or gradient updates.")


def generate_figures(result: dict[str, object], output_dir: Path) -> list[Path]:
    if set(result["split_counts"]) != {"train", "valid"}:
        raise ValueError("Figures require only the explicit train/validation diagnostic populations.")
    validate_numerical_tree(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 11.5,
                         "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 10,
                         "text.color": "#222222", "axes.labelcolor": "#222222"}):
        return [function(result, output_dir) for function in (
            _iou_recall, _confidence_curves, _fn_categories, _confusion, _false_positives, _generalization)]
