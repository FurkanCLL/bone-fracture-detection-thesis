from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bone_fracture_audit.yolo import Annotation


CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
PNG_COMPRESSION = 3
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PreprocessingError(RuntimeError):
    """Raised when the CLAHE dataset cannot be built or validated safely."""


@dataclass(frozen=True)
class ProcessedImageRecord:
    split: str
    source_filename: str
    processed_filename: str
    width: int
    height: int
    source_suffix: str
    source_dtype: str
    source_channels: int
    exif_orientation: int
    original_mean: float
    processed_mean: float
    original_std: float
    processed_std: float
    original_min: int
    original_max: int
    processed_min: int
    processed_max: int
    pixel_count: int
    original_sum: float
    original_square_sum: float
    processed_sum: float
    processed_square_sum: float
    original_zero_count: int
    original_full_count: int
    processed_zero_count: int
    processed_full_count: int
    source_image_sha256: str
    processed_image_sha256: str
    source_label_sha256: str
    processed_label_sha256: str
    annotations: tuple[Annotation, ...]

    @property
    def image_key(self) -> tuple[str, str]:
        return self.split, self.source_filename

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(sorted({annotation.class_id for annotation in self.annotations}))

    @property
    def smallest_box_area(self) -> float | None:
        if not self.annotations:
            return None
        return min(annotation.width * annotation.height for annotation in self.annotations)

    @property
    def aspect_extremeness(self) -> float:
        return abs(math.log(self.width / self.height))


# Applies only the approved intensity transformation and never changes image geometry.
def preprocess_image_array(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise PreprocessingError("Cannot preprocess an empty image array.")
    standardized = _standardize_to_uint8(image)
    if standardized.ndim == 2:
        grayscale = standardized
    elif standardized.ndim == 3 and standardized.shape[2] == 1:
        grayscale = standardized[:, :, 0]
    elif standardized.ndim == 3 and standardized.shape[2] == 3:
        grayscale = cv2.cvtColor(standardized, cv2.COLOR_BGR2GRAY)
    elif standardized.ndim == 3 and standardized.shape[2] == 4:
        grayscale = cv2.cvtColor(standardized, cv2.COLOR_BGRA2GRAY)
    else:
        raise PreprocessingError(f"Unsupported image shape: {image.shape!r}")

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    processed_grayscale = clahe.apply(grayscale)
    processed = np.repeat(processed_grayscale[:, :, np.newaxis], 3, axis=2)
    return grayscale, processed


def channels_are_identical(image: np.ndarray) -> bool:
    return bool(
        image.ndim == 3
        and image.shape[2] == 3
        and np.array_equal(image[:, :, 0], image[:, :, 1])
        and np.array_equal(image[:, :, 0], image[:, :, 2])
    )


def validate_processed_array(image: np.ndarray, expected_width: int, expected_height: int) -> None:
    if image.dtype != np.uint8:
        raise PreprocessingError(f"Processed image must be uint8, found {image.dtype}.")
    if image.shape != (expected_height, expected_width, 3):
        raise PreprocessingError(
            "Processed image geometry or channel count changed: "
            f"{image.shape!r} != {(expected_height, expected_width, 3)!r}."
        )
    if not channels_are_identical(image):
        raise PreprocessingError("Processed image channels are not exactly identical.")


def decode_without_orientation_transform(path: Path) -> tuple[np.ndarray, int]:
    try:
        with Image.open(path) as opened:
            encoded_size = opened.size
            orientation = int(opened.getexif().get(274, 1))
    except (OSError, TypeError, ValueError) as error:
        raise PreprocessingError(f"Could not inspect image metadata for {path}: {error}") from error
    if orientation != 1:
        raise PreprocessingError(
            f"Non-normal EXIF orientation {orientation} requires explicit review before processing: {path}"
        )

    # IMREAD_UNCHANGED retains the stored pixel matrix and does not apply EXIF orientation transforms.
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise PreprocessingError(f"Could not decode source image: {path}")
    height, width = image.shape[:2]
    if (width, height) != encoded_size:
        raise PreprocessingError(f"Decoder geometry differs from encoded geometry for {path}.")
    return image, orientation


def _standardize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.unsignedinteger):
        maximum = np.iinfo(image.dtype).max
        scaled = np.rint(image.astype(np.float64) * (255.0 / maximum))
        return scaled.astype(np.uint8)
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            raise PreprocessingError("Floating-point images must contain finite values in [0, 1].")
        return np.rint(image.astype(np.float64) * 255.0).astype(np.uint8)
    raise PreprocessingError(f"Unsupported source image dtype: {image.dtype}")
