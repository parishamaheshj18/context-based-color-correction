"""Benchmark dataset format.

A dataset directory holds ground_truth.csv with columns:
  image            path relative to the directory
  r, g, b          ground-truth illuminant in the image's linear camera RGB
  chart_mask       optional mask image; nonzero pixels (the chart) are hidden during inference
  chart_patches    optional CSV of 24 ColorChecker patch camera-RGB rows for ΔE00
  skin_group       optional skin-tone group label, for the fairness breakdown
Converters for public benchmarks (Gehler-Shi, NUS 8-camera, Cube+) only need to write this file.
"""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from natural_color_charts.domain import LinearImage, RgbVector, SpectralArray


@dataclass(frozen=True, eq=False)
class BenchmarkSample:
    """One dataset row."""

    image_path: Path
    truth_rgb: RgbVector
    chart_mask_path: Path | None = None
    chart_patches_kx3: SpectralArray | None = None
    skin_group: str | None = None


def _optional_path(root: Path, value: str | None) -> Path | None:
    return root / value if value else None


def read_patch_table(path: Path) -> SpectralArray:
    """24×3 patch RGB values; a header row is skipped."""
    rows = [row for row in csv.reader(path.read_text().splitlines()) if row]
    numeric = [row for row in rows if _is_float(row[0])]
    return np.array([[float(v) for v in row[:3]] for row in numeric])


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def load_samples(dataset_dir: Path) -> list[BenchmarkSample]:
    """Parse ground_truth.csv."""
    samples = []
    with (dataset_dir / "ground_truth.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            patches_path = _optional_path(dataset_dir, row.get("chart_patches"))
            samples.append(
                BenchmarkSample(
                    image_path=dataset_dir / row["image"],
                    truth_rgb=np.array([float(row["r"]), float(row["g"]), float(row["b"])]),
                    chart_mask_path=_optional_path(dataset_dir, row.get("chart_mask")),
                    chart_patches_kx3=read_patch_table(patches_path) if patches_path else None,
                    skin_group=row.get("skin_group") or None,
                )
            )
    return samples


def hide_chart(image: LinearImage, mask_path: Path | None) -> LinearImage:
    """Black out the chart and exclude it so no stage can use it."""
    if mask_path is None:
        return image
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise ValueError(f"Could not read chart mask {mask_path}")
    mask = cv2.resize(mask_image, (image.width, image.height), interpolation=cv2.INTER_NEAREST) > 0
    rgb = image.rgb.copy()
    rgb[mask] = 0.0
    return dataclasses.replace(image, rgb=rgb, excluded_mask=mask)
