from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Sequence

import cv2
from PIL import Image, ImageDraw, ImageFont

from bone_fracture_pipeline.preprocessing_core import (
    PreprocessingError,
    ProcessedImageRecord,
    decode_without_orientation_transform,
    preprocess_image_array,
)


VISUAL_SAMPLES_PER_CATEGORY = 2


def select_visual_review(
    records: Sequence[ProcessedImageRecord],
    *,
    class_count: int,
    per_category: int = VISUAL_SAMPLES_PER_CATEGORY,
) -> list[tuple[ProcessedImageRecord, tuple[str, ...]]]:
    eligible = [record for record in records if record.split in {"train", "valid"}]
    if not eligible:
        raise PreprocessingError("Visual QA requires train or validation images.")
    if per_category <= 0:
        raise PreprocessingError("Visual QA category size must be positive.")

    stable_key = lambda record: (
        record.split,
        record.source_filename.casefold(),
        record.source_filename,
    )
    reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_key = {record.image_key: record for record in eligible}

    # Class representatives are typical-intensity examples rather than hand-picked favorable cases.
    for class_id in range(class_count):
        candidates = [record for record in eligible if class_id in record.class_ids]
        if not candidates:
            raise PreprocessingError(f"No train/validation visual candidate exists for class {class_id}.")
        class_median = median(record.original_mean for record in candidates)
        representative = min(
            candidates,
            key=lambda record: (abs(record.original_mean - class_median), stable_key(record)),
        )
        reasons[representative.image_key].add(f"class_{class_id}_representative")

    category_rankings = {
        "dark_image": sorted(eligible, key=lambda record: (record.original_mean, stable_key(record))),
        "bright_image": sorted(eligible, key=lambda record: (-record.original_mean, stable_key(record))),
        "low_contrast": sorted(eligible, key=lambda record: (record.original_std, stable_key(record))),
        "small_annotated_region": sorted(
            (record for record in eligible if record.smallest_box_area is not None),
            key=lambda record: (record.smallest_box_area, stable_key(record)),
        ),
        "multi_annotation": sorted(
            (record for record in eligible if len(record.annotations) > 1),
            key=lambda record: (-len(record.annotations), stable_key(record)),
        ),
        "unusual_aspect_ratio": sorted(
            eligible,
            key=lambda record: (-record.aspect_extremeness, stable_key(record)),
        ),
    }
    for reason, ranked in category_rankings.items():
        for record in ranked[:per_category]:
            reasons[record.image_key].add(reason)

    return [
        (by_key[key], tuple(sorted(reasons[key])))
        for key in sorted(reasons, key=lambda value: (value[0], value[1].casefold(), value[1]))
    ]


def render_visual_review(
    source: Path,
    processed_build: Path,
    review_build: Path,
    review_final: Path,
    selected: Sequence[tuple[ProcessedImageRecord, tuple[str, ...]]],
    class_names: Sequence[str],
    project_root: Path,
) -> list[dict[str, object]]:
    review_build.mkdir(parents=True)
    rendered: list[Path] = []
    rows: list[dict[str, object]] = []
    for index, (record, reasons) in enumerate(selected, start=1):
        source_image = source / record.split / "images" / record.source_filename
        processed_image = processed_build / record.split / "images" / record.processed_filename
        filename = f"{index:02d}_{record.split}_{Path(record.source_filename).stem}.png"
        destination = review_build / filename
        _render_review_image(source_image, processed_image, destination, record, reasons, class_names)
        rendered.append(destination)
        rows.append(
            {
                "review_order": index,
                "split": record.split,
                "source_filename": record.source_filename,
                "processed_filename": record.processed_filename,
                "review_image_path": _display_path(review_final / filename, project_root),
                "selection_reasons": ";".join(reasons),
                "class_ids": ";".join(str(value) for value in record.class_ids),
                "annotation_count": len(record.annotations),
                "original_mean": _round(record.original_mean),
                "processed_mean": _round(record.processed_mean),
                "original_std": _round(record.original_std),
                "processed_std": _round(record.processed_std),
            }
        )
    _render_contact_sheet(rendered, review_build / "contact_sheet.png")
    return rows


def _render_review_image(
    source_path: Path,
    processed_path: Path,
    output_path: Path,
    record: ProcessedImageRecord,
    reasons: Sequence[str],
    class_names: Sequence[str],
) -> None:
    source_array, _ = decode_without_orientation_transform(source_path)
    original_gray, _ = preprocess_image_array(source_array)
    processed_array = cv2.imread(str(processed_path), cv2.IMREAD_UNCHANGED)
    if processed_array is None:
        raise PreprocessingError(f"Could not decode visual-review input: {processed_path}")

    original = Image.fromarray(original_gray).convert("RGB")
    processed = Image.fromarray(processed_array[:, :, 0]).convert("RGB")
    max_panel_width, max_panel_height = 720, 720
    scale = min(max_panel_width / record.width, max_panel_height / record.height, 1.0)
    display_size = (max(1, round(record.width * scale)), max(1, round(record.height * scale)))
    if original.size != display_size:
        original = original.resize(display_size, Image.Resampling.LANCZOS)
        processed = processed.resize(display_size, Image.Resampling.LANCZOS)

    header_height = 84
    footer_height = 26
    gap = 12
    canvas = Image.new(
        "RGB",
        (display_size[0] * 2 + gap, display_size[1] + header_height + footer_height),
        "black",
    )
    canvas.paste(original, (0, header_height))
    canvas.paste(processed, (display_size[0] + gap, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 6), f"{record.split}: {record.source_filename}", fill="white", font=font)
    draw.text((8, 24), "reasons: " + ", ".join(reasons), fill="white", font=font)
    draw.text(
        (8, 42),
        f"mean {record.original_mean:.2f} -> {record.processed_mean:.2f}; "
        f"std {record.original_std:.2f} -> {record.processed_std:.2f}",
        fill="white",
        font=font,
    )
    draw.text((8, 62), "Original grayscale", fill=(190, 220, 255), font=font)
    draw.text((display_size[0] + gap + 8, 62), "CLAHE", fill=(190, 255, 200), font=font)

    line_width = max(2, round(min(display_size) / 250))
    for annotation in record.annotations:
        left = (annotation.x_center - annotation.width / 2) * display_size[0]
        top = (annotation.y_center - annotation.height / 2) * display_size[1] + header_height
        right = (annotation.x_center + annotation.width / 2) * display_size[0]
        bottom = (annotation.y_center + annotation.height / 2) * display_size[1] + header_height
        label = f"{annotation.class_id}: {class_names[annotation.class_id]}"
        for x_offset, color in ((0, (0, 255, 255)), (display_size[0] + gap, (255, 80, 220))):
            box = (left + x_offset, top, right + x_offset, bottom)
            draw.rectangle(box, outline=color, width=line_width)
            label_y = max(header_height, top - 13)
            draw.rectangle(
                (left + x_offset, label_y, left + x_offset + len(label) * 6 + 6, label_y + 13),
                fill="black",
            )
            draw.text((left + x_offset + 3, label_y), label, fill=color, font=font)
    draw.text(
        (8, header_height + display_size[1] + 6),
        "Same normalized boxes on both panels",
        fill="white",
        font=font,
    )
    canvas.save(output_path, format="PNG", optimize=True)


def _render_contact_sheet(paths: Sequence[Path], output: Path) -> None:
    if not paths:
        raise PreprocessingError("Visual QA selection produced no images.")
    thumb_width, thumb_height = 520, 260
    columns = 2
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), (24, 24, 24))
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            thumbnail = opened.convert("RGB")
            thumbnail.thumbnail((thumb_width - 10, thumb_height - 10), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_width + (thumb_width - thumbnail.width) // 2
        y = (index // columns) * thumb_height + (thumb_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (x, y))
    sheet.save(output, format="PNG", optimize=True)


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _round(value: float) -> float:
    return round(value, 12)
