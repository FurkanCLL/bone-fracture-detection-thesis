from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .audit import difference_hash, hamming_distance, sha256_file


ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{16,}$", re.IGNORECASE)
DEFAULT_DATASET_ROOT = Path("data/raw/bone-fracture-detection/bone fracture detection.v4-v4.yolov8")
DEFAULT_OUTPUT_DIR = Path("outputs/radiologist_empty_label_review")
DEFAULT_SEED = 20260914


@dataclass(frozen=True)
class ReviewCandidate:
    image_path: Path
    label_path: Path
    split: str
    source_key: str
    width: int
    height: int
    sha256: str
    dhash: str
    fingerprint: np.ndarray


def source_key(path: Path) -> str:
    return ROBOFLOW_SUFFIX.sub("", path.stem).casefold()


# Creates a small grayscale signature used only to spread the sample across different-looking X-rays.
def image_fingerprint(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        grayscale = ImageOps.autocontrast(ImageOps.grayscale(source))
        resized = ImageOps.fit(grayscale, (64, 64), method=Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0


# Reads empty labels from the allowed splits without opening the test directory.
def collect_empty_candidates(dataset_root: Path, splits: tuple[str, ...] = ("train", "valid")) -> list[ReviewCandidate]:
    candidates: list[ReviewCandidate] = []
    for split in splits:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        for label_path in sorted(label_dir.glob("*.txt"), key=lambda value: value.name.casefold()):
            if label_path.read_text(encoding="utf-8-sig").strip():
                continue
            image_path = image_dir / f"{label_path.stem}.jpg"
            if not image_path.is_file():
                continue
            with Image.open(image_path) as image:
                width, height = image.size
                perceptual_hash = difference_hash(image)
            candidates.append(
                ReviewCandidate(
                    image_path=image_path,
                    label_path=label_path,
                    split=split,
                    source_key=source_key(image_path),
                    width=width,
                    height=height,
                    sha256=sha256_file(image_path),
                    dhash=perceptual_hash,
                    fingerprint=image_fingerprint(image_path),
                )
            )
    return candidates


def fingerprint_difference(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first - second)))


# Uses a seeded farthest-first pass while rejecting repeated names, exact copies, and obvious visual variants.
def select_diverse_candidates(
    candidates: list[ReviewCandidate],
    split_counts: dict[str, int],
    seed: int,
) -> list[ReviewCandidate]:
    random_generator = random.Random(seed)
    shuffled = list(candidates)
    random_generator.shuffle(shuffled)
    tie_order = {candidate.image_path: index for index, candidate in enumerate(shuffled)}
    selected: list[ReviewCandidate] = []

    for split, count in split_counts.items():
        available = [candidate for candidate in shuffled if candidate.split == split]
        while sum(candidate.split == split for candidate in selected) < count:
            eligible: list[tuple[float, ReviewCandidate]] = []
            selected_hashes = {candidate.sha256 for candidate in selected}
            selected_sources = {candidate.source_key for candidate in selected}
            selected_paths = {candidate.image_path for candidate in selected}
            for candidate in available:
                if candidate.image_path in selected_paths or candidate.sha256 in selected_hashes or candidate.source_key in selected_sources:
                    continue
                differences = [fingerprint_difference(candidate.fingerprint, other.fingerprint) for other in selected]
                if any(
                    hamming_distance(candidate.dhash, other.dhash) <= 4 and difference <= 0.08
                    for other, difference in zip(selected, differences)
                ):
                    continue
                # The minimum distance spreads each new choice away from everything already selected.
                diversity = min(differences) if differences else 1.0
                eligible.append((diversity, candidate))

            if not eligible:
                raise ValueError(f"Could not select {count} sufficiently distinct empty-label images from {split}.")
            eligible.sort(key=lambda item: (-item[0], tie_order[item[1].image_path]))
            selected.append(eligible[0][1])
    return selected


# Creates a neutral response sheet without describing empty labels as healthy images.
def _write_review_readme(path: Path, seed: int, image_count: int) -> None:
    lines = [
        "# Empty-label X-ray review",
        "",
        "These images currently have no fracture annotation in the dataset. Please review each image without assuming it is healthy.",
        "",
        "For each image, choose one response:",
        "",
        "- no visible fracture",
        "- visible/suspected fracture",
        "- uncertain",
        "",
        f"The sample was selected reproducibly with seed `{seed}`. This small review is a sanity check and cannot validate every empty label in the dataset.",
        "",
        "| Image | Classification | Optional note |",
        "|---|---|---|",
    ]
    lines.extend(f"| empty_{index:02d}.jpg |  |  |" for index in range(1, image_count + 1))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Copies the chosen files byte-for-byte and records their source paths and hashes.
def prepare_review_set(
    dataset_root: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    train_count: int = 16,
    valid_count: int = 4,
) -> list[dict[str, str | int]]:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(dataset_root):
        raise ValueError("The review set must be written outside the immutable raw dataset.")

    candidates = collect_empty_candidates(dataset_root)
    selected = select_diverse_candidates(candidates, {"train": train_count, "valid": valid_count}, seed)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    rows: list[dict[str, str | int]] = []
    raw_root = dataset_root.parents[1]
    for index, candidate in enumerate(selected, start=1):
        review_name = f"empty_{index:02d}.jpg"
        review_path = output_dir / review_name
        # Copying bytes keeps the review image traceable to its raw source hash.
        shutil.copyfile(candidate.image_path, review_path)
        rows.append(
            {
                "review_image": review_name,
                "original_relative_path": candidate.image_path.relative_to(raw_root).as_posix(),
                "split": candidate.split,
                "original_filename": candidate.image_path.name,
                "width": candidate.width,
                "height": candidate.height,
                "source_key": candidate.source_key,
                "selection_seed": seed,
                "source_sha256": candidate.sha256,
                "review_sha256": sha256_file(review_path),
            }
        )

    with (output_dir / "review_index.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_review_readme(output_dir / "README.md", seed, len(rows))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a reproducible empty-label X-ray review set.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--valid-count", type=int, default=4)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    rows = prepare_review_set(
        arguments.dataset_root,
        arguments.output,
        arguments.seed,
        arguments.train_count,
        arguments.valid_count,
    )
    print(f"Created {len(rows)} review images in {arguments.output}")


if __name__ == "__main__":
    main()
