from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")


@dataclass(frozen=True)
class Annotation:
    line_number: int
    class_id: int
    annotation_type: str
    x_center: float
    y_center: float
    width: float
    height: float
    polygon_points: tuple[tuple[float, float], ...] = ()

    @property
    def relative_area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class AnnotationIssue:
    line_number: int
    code: str
    message: str
    raw_line: str


@dataclass(frozen=True)
class ParsedLabel:
    annotations: tuple[Annotation, ...]
    issues: tuple[AnnotationIssue, ...]
    is_empty: bool


def parse_yolo_label(path: Path, valid_class_ids: set[int]) -> ParsedLabel:
    """Parses one YOLO label without changing or correcting its contents."""
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return ParsedLabel(annotations=(), issues=(), is_empty=True)

    annotations: list[Annotation] = []
    issues: list[AnnotationIssue] = []
    seen_rows: set[tuple[int, float, float, float, float]] = set()

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            issues.append(
                AnnotationIssue(line_number, "blank_line", "Blank line inside a non-empty label file.", raw_line)
            )
            continue

        values = stripped.split()
        is_box = len(values) == 5
        is_polygon = len(values) >= 7 and len(values) % 2 == 1
        if not is_box and not is_polygon:
            issues.append(
                AnnotationIssue(
                    line_number,
                    "incorrect_value_count",
                    f"Expected 5 box values or a class plus at least 3 coordinate pairs, but found {len(values)} values.",
                    raw_line,
                )
            )
            continue

        class_token, *coordinate_tokens = values
        if not INTEGER_PATTERN.fullmatch(class_token):
            issues.append(
                AnnotationIssue(line_number, "invalid_class_format", "Class ID must be an integer.", raw_line)
            )
            continue

        try:
            class_id = int(class_token)
            coordinates = tuple(float(value) for value in coordinate_tokens)
        except ValueError:
            issues.append(
                AnnotationIssue(line_number, "non_numeric_value", "Annotation coordinates must be numeric.", raw_line)
            )
            continue

        if not all(math.isfinite(value) for value in coordinates):
            issues.append(
                AnnotationIssue(line_number, "non_finite_value", "Annotation coordinates must be finite.", raw_line)
            )
            continue

        polygon_points: tuple[tuple[float, float], ...] = ()
        if is_box:
            x_center, y_center, width, height = coordinates
        else:
            polygon_points = tuple(zip(coordinates[::2], coordinates[1::2]))
            x_values = [point[0] for point in polygon_points]
            y_values = [point[1] for point in polygon_points]
            left, right = min(x_values), max(x_values)
            top, bottom = min(y_values), max(y_values)
            x_center = (left + right) / 2
            y_center = (top + bottom) / 2
            width = right - left
            height = bottom - top

        row_has_error = False
        if class_id not in valid_class_ids:
            issues.append(
                AnnotationIssue(
                    line_number,
                    "unexpected_class_id",
                    f"Class ID {class_id} is not declared by the dataset configuration.",
                    raw_line,
                )
            )
            row_has_error = True

        if is_polygon and not all(0.0 <= value <= 1.0 for value in coordinates):
            issues.append(
                AnnotationIssue(
                    line_number,
                    "polygon_coordinate_out_of_range",
                    "Polygon coordinates must be between 0 and 1.",
                    raw_line,
                )
            )
            row_has_error = True
        elif is_box and (not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0):
            issues.append(
                AnnotationIssue(
                    line_number,
                    "center_out_of_range",
                    "Box center coordinates must be between 0 and 1.",
                    raw_line,
                )
            )
            row_has_error = True

        if width <= 0.0 or height <= 0.0:
            issues.append(
                AnnotationIssue(
                    line_number,
                    "non_positive_size",
                    "Box width and height must be greater than zero.",
                    raw_line,
                )
            )
            row_has_error = True
        elif width > 1.0 or height > 1.0:
            issues.append(
                AnnotationIssue(
                    line_number,
                    "size_out_of_range",
                    "Box width and height must not exceed 1.",
                    raw_line,
                )
            )
            row_has_error = True

        if is_box and (
            x_center - width / 2 < 0.0
            or x_center + width / 2 > 1.0
            or y_center - height / 2 < 0.0
            or y_center + height / 2 > 1.0
        ):
            issues.append(
                AnnotationIssue(
                    line_number,
                    "box_outside_image",
                    "The box extends beyond the normalized image boundary.",
                    raw_line,
                )
            )
            row_has_error = True

        if is_polygon and len(set(polygon_points)) < 3:
            issues.append(
                AnnotationIssue(
                    line_number,
                    "degenerate_polygon",
                    "A polygon must contain at least three distinct points.",
                    raw_line,
                )
            )
            row_has_error = True

        row = (class_id, *coordinates)
        if row in seen_rows:
            issues.append(
                AnnotationIssue(line_number, "duplicate_annotation", "Identical annotation appears more than once.", raw_line)
            )
        seen_rows.add(row)

        # Invalid rows stay visible in the issue table but are excluded from statistics.
        if not row_has_error:
            annotations.append(
                Annotation(
                    line_number,
                    class_id,
                    "box" if is_box else "polygon",
                    x_center,
                    y_center,
                    width,
                    height,
                    polygon_points,
                )
            )

    return ParsedLabel(tuple(annotations), tuple(issues), is_empty=False)
