"""Thesis figures for train/validation dataset difficulty measurements."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SPLITS = ("train", "valid")
SPLIT_LABELS = {"train": "Train", "valid": "Validation"}
SPLIT_COLORS = {"train": "#0072B2", "valid": "#D55E00"}
THRESHOLD_COLOR = "#777777"


def _check_splits(*row_sets: list[dict[str, object]]) -> None:
    # Reject unexpected rows before creating figures so held-out data cannot enter a plot.
    for rows in row_sets:
        unexpected = {str(row["split"]) for row in rows} - set(SPLITS)
        if unexpected:
            raise ValueError(f"Figures accept train and valid rows only: {sorted(unexpected)}")


def _values(rows: list[dict[str, object]], split: str, field: str, scale: float = 1.0) -> np.ndarray:
    return np.asarray([float(row[field]) * scale for row in rows if row["split"] == split], dtype=float)


def _histogram(ax: plt.Axes, rows: list[dict[str, object]], field: str, xlabel: str,
               scale: float = 1.0, log_x: bool = False) -> None:
    groups = {split: _values(rows, split, field, scale) for split in SPLITS}
    combined = np.concatenate(tuple(groups.values()))
    if not len(combined):
        ax.text(0.5, 0.5, "No observations", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        return
    if not np.all(np.isfinite(combined)) or (log_x and np.any(combined <= 0)):
        raise ValueError(f"Invalid values for figure field: {field}")

    if log_x:
        low = 10 ** np.floor(np.log10(np.min(combined)))
        high = 10 ** np.ceil(np.log10(np.max(combined)))
        if low == high:
            high *= 10
        bins = np.geomspace(low, high, 32)
        ax.set_xscale("log")
    else:
        bins = np.histogram_bin_edges(combined, bins="auto")
        if len(bins) > 36:
            bins = np.linspace(bins[0], bins[-1], 36)

    for split in SPLITS:
        values = groups[split]
        if len(values):
            weights = np.full(len(values), 100.0 / len(values))
            ax.hist(values, bins=bins, weights=weights, histtype="step", linewidth=2,
                    color=SPLIT_COLORS[split], label=f"{SPLIT_LABELS[split]} (n={len(values)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Observations per split (%)")
    ax.grid(axis="y", alpha=0.2)


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _image_dimensions(image_rows: list[dict[str, object]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True)
    for ax, field, label, scale, title in (
        (axes[0, 0], "width_px", "Image width (pixels)", 1.0, "Width"),
        (axes[0, 1], "height_px", "Image height (pixels)", 1.0, "Height"),
        (axes[1, 0], "pixel_area", "Image area (megapixels)", 1e-6, "Pixel area"),
    ):
        _histogram(ax, image_rows, field, label, scale)
        ax.set_title(title)

    scatter = axes[1, 1]
    for split in SPLITS:
        widths = _values(image_rows, split, "width_px")
        heights = _values(image_rows, split, "height_px")
        scatter.scatter(widths, heights, s=17, alpha=0.42, color=SPLIT_COLORS[split],
                        linewidths=0, label=SPLIT_LABELS[split], rasterized=True)
    scatter.set(xlabel="Image width (pixels)", ylabel="Image height (pixels)",
                title="Resolution combinations")
    scatter.grid(alpha=0.2)
    scatter.legend(frameon=False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Source image dimensions: train and validation")
    return _save(fig, output_dir, "image_dimensions.png")


def _aspect_ratio(image_rows: list[dict[str, object]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    _histogram(axes[0], image_rows, "aspect_ratio", "Width / height")
    axes[0].axvline(1, color=THRESHOLD_COLOR, linestyle="--", linewidth=1, label="1:1")
    axes[0].set_title("Aspect-ratio distribution")
    axes[0].legend(frameon=False)

    order = ("portrait", "approximately_square", "landscape")
    found = {str(row["orientation"]) for row in image_rows}
    categories = [category for category in order if category in found]
    categories += sorted(found - set(categories))
    positions = np.arange(len(categories), dtype=float)
    if categories:
        for index, split in enumerate(SPLITS):
            split_rows = [row for row in image_rows if row["split"] == split]
            shares = [100 * sum(str(row["orientation"]) == category for row in split_rows)
                      / len(split_rows) if split_rows else 0 for category in categories]
            axes[1].barh(positions + (index - 0.5) * 0.36, shares, height=0.34,
                         color=SPLIT_COLORS[split], label=SPLIT_LABELS[split])
        axes[1].set_yticks(positions, labels=[category.replace("_", " ").capitalize() for category in categories])
        axes[1].invert_yaxis()
    axes[1].set(xlabel="Images per split (%)", title="Orientation groups", xlim=(0, 100))
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("Image aspect ratios: train and validation")
    return _save(fig, output_dir, "aspect_ratio_distribution.png")


def _bbox_area(annotation_rows: list[dict[str, object]], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    _histogram(ax, annotation_rows, "normalized_area", "Box area (% of image area)", 100, log_x=True)
    for threshold in (0.5, 1, 2, 5, 10):
        ax.axvline(threshold, color=THRESHOLD_COLOR, linestyle=":", linewidth=0.9, alpha=0.65)
    ax.set_xticks([0.01, 0.1, 0.5, 1, 2, 5, 10, 50],
                  labels=["0.01", "0.1", "0.5", "1", "2", "5", "10", "50"])
    ax.set_xlim(right=max(ax.get_xlim()[1], 10))
    ax.legend(frameon=False)
    ax.set_title("Source-annotated detection-box area")
    fig.text(0.5, -0.035, "Dotted lines mark descriptive cutoffs at 0.5%, 1%, 2%, 5%, and 10%.",
             ha="center", fontsize=9)
    return _save(fig, output_dir, "bbox_area_distribution.png")


def _effective_size(annotation_rows: list[dict[str, object]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)
    for ax, field, title in (
        (axes[0], "effective_width_640", "Effective box width"),
        (axes[1], "effective_height_640", "Effective box height"),
    ):
        _histogram(ax, annotation_rows, field, "Pixels after resize to 640", log_x=True)
        ax.set_title(title)
    axes[0].legend(frameon=False)

    cdf = axes[2]
    for split in SPLITS:
        values = np.sort(_values(annotation_rows, split, "effective_short_side_640"))
        if len(values):
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("Invalid effective shorter-side values")
            cdf.step(values, np.arange(1, len(values) + 1) * 100 / len(values),
                     where="post", color=SPLIT_COLORS[split], linewidth=2,
                     label=f"{SPLIT_LABELS[split]} (n={len(values)})")
    for threshold in (8, 16, 32, 64):
        cdf.axvline(threshold, color=THRESHOLD_COLOR, linestyle=":", linewidth=0.9, alpha=0.65)
    cdf.set(xlabel="Effective shorter side (pixels)", ylabel="Boxes at or below size (%)",
            title="Shorter-side cumulative distribution", ylim=(0, 100))
    cdf.set_xscale("log")
    cdf.set_xticks([8, 16, 32, 64, 128, 256], labels=["8", "16", "32", "64", "128", "256"])
    cdf.grid(axis="y", alpha=0.2)
    cdf.legend(frameon=False)
    fig.suptitle("Box size after aspect-ratio-preserving resize to a 640-pixel canvas")
    return _save(fig, output_dir, "effective_bbox_size_640.png")


def _class_scale(annotation_rows: list[dict[str, object]], class_rows: list[dict[str, object]],
                 output_dir: Path) -> Path:
    classes = {int(row["class_id"]): str(row["class_name"]) for row in class_rows}
    class_ids = sorted(classes)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True, constrained_layout=True)
    for ax, field, xlabel, scale, title in (
        (axes[0], "normalized_area", "Box area (% of image)", 100, "Area by source class"),
        (axes[1], "effective_short_side_640", "Effective shorter side (pixels)", 1,
         "Shorter side at 640"),
    ):
        for split_index, split in enumerate(SPLITS):
            first = True
            for index, class_id in enumerate(class_ids):
                values = np.asarray([float(row[field]) * scale for row in annotation_rows
                                     if row["split"] == split and int(row["class_id"]) == class_id])
                if not len(values):
                    continue
                lower, middle, upper = np.percentile(values, [25, 50, 75])
                ax.errorbar(middle, index + (split_index - 0.5) * 0.30,
                            xerr=[[middle - lower], [upper - middle]], fmt="o", capsize=2.5,
                            color=SPLIT_COLORS[split], markersize=5,
                            label=SPLIT_LABELS[split] if first else None)
                first = False
        ax.set(xlabel=xlabel, title=title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_xscale("log")
    axes[0].set_yticks(np.arange(len(class_ids)), labels=[classes[class_id] for class_id in class_ids])
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False)
    fig.suptitle("Class-wise box scale: median and interquartile range")
    return _save(fig, output_dir, "class_bbox_scale_comparison.png")


def _comparison(image_rows: list[dict[str, object]], annotation_rows: list[dict[str, object]],
                class_rows: list[dict[str, object]], output_dir: Path) -> Path:
    fig = plt.figure(figsize=(11.8, 9.3), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.6])
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
    groups = {split: [row for row in image_rows if row["split"] == split] for split in SPLITS}
    comparison_values = (
        ([100 * sum(bool(row["positive"]) for row in groups[split]) / len(groups[split])
          if groups[split] else 0 for split in SPLITS], "Positive images (%)", "Positive-image share"),
        ([float(np.median(_values(image_rows, split, "pixel_area", 1e-6)))
          if groups[split] else 0 for split in SPLITS], "Image area (megapixels)", "Median image area"),
        ([float(np.median(values)) if len(values) else 0 for values in
          (_values(annotation_rows, split, "normalized_area", 100) for split in SPLITS)],
         "Box area (% of image)", "Median box area"),
        ([float(np.median(values)) if len(values) else 0 for values in
          (_values(annotation_rows, split, "effective_short_side_640") for split in SPLITS)],
         "Effective shorter side (pixels)", "Median shorter side at 640"),
    )
    for ax, (values, xlabel, title) in zip(axes, comparison_values):
        ax.barh([0, 1], values, color=[SPLIT_COLORS[split] for split in SPLITS], height=0.58)
        ax.set_yticks([0, 1], labels=[SPLIT_LABELS[split] for split in SPLITS])
        ax.invert_yaxis()
        ax.set(xlabel=xlabel, title=title, xlim=(0, max(values) * 1.2 if max(values) else 1))
        ax.grid(axis="x", alpha=0.2)

    class_ax = fig.add_subplot(grid[2, :])
    classes = {int(row["class_id"]): str(row["class_name"]) for row in class_rows}
    class_ids = sorted(classes)
    positions = np.arange(len(class_ids), dtype=float)
    for index, split in enumerate(SPLITS):
        total = sum(int(row["annotations"]) for row in class_rows if row["split"] == split)
        count_by_class = {int(row["class_id"]): int(row["annotations"]) for row in class_rows
                          if row["split"] == split}
        shares = [100 * count_by_class.get(class_id, 0) / total if total else 0
                  for class_id in class_ids]
        class_ax.barh(positions + (index - 0.5) * 0.36, shares, height=0.34,
                      color=SPLIT_COLORS[split], label=SPLIT_LABELS[split])
    class_ax.set_yticks(positions, labels=[classes[class_id] for class_id in class_ids])
    class_ax.invert_yaxis()
    largest_share = max((patch.get_width() for patch in class_ax.patches), default=0)
    class_ax.set(xlabel="Annotations per split (%)", title="Source-class distribution",
                 xlim=(0, max(30, largest_share * 1.15)))
    class_ax.grid(axis="x", alpha=0.2)
    class_ax.legend(frameon=False)
    fig.suptitle("Train–validation comparison of dataset composition and box scale")
    return _save(fig, output_dir, "train_validation_comparison.png")


def generate_figures(image_rows: list[dict[str, object]],
                     annotation_rows: list[dict[str, object]],
                     class_rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    """Save six deterministic train/validation plots and return their paths."""
    _check_splits(image_rows, annotation_rows, class_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "figure.titlesize": 13,
                         "axes.spines.top": False, "axes.spines.right": False}):
        return [
            _image_dimensions(image_rows, output_dir),
            _aspect_ratio(image_rows, output_dir),
            _bbox_area(annotation_rows, output_dir),
            _effective_size(annotation_rows, output_dir),
            _class_scale(annotation_rows, class_rows, output_dir),
            _comparison(image_rows, annotation_rows, class_rows, output_dir),
        ]
