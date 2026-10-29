"""Synthetic test harness: render known scenes so estimators can be checked exactly."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from natural_color_charts.colorimetry import encode_srgb
from natural_color_charts.domain import (
    ImageMetadata,
    InputMode,
    LinearImage,
    ReferenceClass,
    ReferenceRegion,
    SpectralArray,
)
from natural_color_charts.io.writers import write_image8, write_tiff16
from natural_color_charts.spectral.rendering import sensor_responses


@dataclass(frozen=True, eq=False)
class SyntheticPatches:
    """Rendered patch colors together with the ground truth used to make them."""

    rgb_kx3: SpectralArray
    reflectances_kxn: SpectralArray
    brightness_k: SpectralArray


def render_synthetic_patches(
    sensitivity_3xn: SpectralArray,
    illuminant: SpectralArray,
    reflectances_kxn: SpectralArray,
    rng: np.random.Generator,
    noise_sigma: float = 0.0,
    brightness_range: tuple[float, float] = (1.0, 1.0),
) -> SyntheticPatches:
    """Apply the image formation model with random per-patch brightness and relative noise."""
    clean = sensor_responses(sensitivity_3xn, illuminant, reflectances_kxn)
    brightness = np.asarray(rng.uniform(*brightness_range, size=clean.shape[0]))
    noisy = clean * brightness[:, None] * (1.0 + rng.normal(0.0, noise_sigma, size=clean.shape))
    return SyntheticPatches(np.clip(noisy, 1e-9, None), reflectances_kxn, brightness)


def sample_prior_reflectances(
    mean_n: SpectralArray,
    basis_nxk: SpectralArray,
    variances_k: SpectralArray,
    count: int,
    rng: np.random.Generator,
) -> SpectralArray:
    """Draw reflectances R = μ + U·b with b ~ N(0, Σ), clipped to physical values."""
    coefficients = rng.normal(size=(count, variances_k.size)) * np.sqrt(variances_k)
    return np.clip(mean_n + coefficients @ basis_nxk.T, 0.001, 1.0)


@dataclass(frozen=True, eq=False)
class SyntheticScene:
    """An image made of flat patches plus one reference region per patch."""

    image: LinearImage
    regions: tuple[ReferenceRegion, ...]


def make_patch_image(
    patch_rgb_kx3: SpectralArray,
    labels: Sequence[ReferenceClass],
    patch_px: int = 32,
    background_rgb: Sequence[float] = (0.05, 0.05, 0.05),
    mode: InputMode = InputMode.RAW,
) -> SyntheticScene:
    """Lay patches out in a row on a background, with exact masks for each patch."""
    count = len(labels)
    height, width = patch_px * 3, patch_px * (2 * count + 1)
    rgb: NDArray[np.float32] = np.empty((height, width, 3), dtype=np.float32)
    rgb[:] = np.asarray(background_rgb, dtype=np.float32)
    regions = []
    for index, label in enumerate(labels):
        mask = np.zeros((height, width), dtype=bool)
        left = patch_px * (2 * index + 1)
        mask[patch_px : 2 * patch_px, left : left + patch_px] = True
        rgb[mask] = patch_rgb_kx3[index]
        regions.append(ReferenceRegion(label=label, mask=mask, detector_score=1.0))
    image = LinearImage(
        rgb=rgb,
        saturation_mask=np.zeros((height, width), dtype=bool),
        metadata=ImageMetadata(make="Synthetic", model="Camera"),
        mode=mode,
    )
    return SyntheticScene(image=image, regions=tuple(regions))


@dataclass(frozen=True, eq=False)
class SyntheticBenchmarkSpec:
    """Inputs for writing a synthetic benchmark in the dataset format of evaluation.dataset."""

    sensitivity_3xn: SpectralArray
    illuminants: Sequence[SpectralArray]
    reference_reflectances: Sequence[Sequence[tuple[ReferenceClass, SpectralArray]]]
    background_reflectances: Sequence[SpectralArray]
    chart_reflectances_kxn: SpectralArray
    noise_sigma: float = 0.002
    size_px: tuple[int, int] = (384, 512)


def _paint(
    reflectance_map: NDArray[np.float64], box: tuple[int, int, int, int], value: SpectralArray
) -> None:
    top, left, bottom, right = box
    reflectance_map[top:bottom, left:right] = value


def _render_reflectance_map(
    reflectance_map: NDArray[np.float64], spec: SyntheticBenchmarkSpec, illuminant: SpectralArray
) -> NDArray[np.float64]:
    height, width, bands = reflectance_map.shape
    flat = reflectance_map.reshape(-1, bands) * illuminant @ spec.sensitivity_3xn.T
    return flat.reshape(height, width, 3)


def write_synthetic_benchmark(
    directory: Path, spec: SyntheticBenchmarkSpec, rng: np.random.Generator
) -> Path:
    """Write images, region sidecars, chart masks and patches, and ground_truth.csv."""
    directory.mkdir(parents=True, exist_ok=True)
    height, width = spec.size_px
    rows = ["image,r,g,b,chart_mask,chart_patches"]
    for index, illuminant in enumerate(spec.illuminants):
        stem = f"scene_{index:03d}"
        reflectance_map = np.empty((height, width, spec.sensitivity_3xn.shape[1]))
        reflectance_map[:] = spec.background_reflectances[index]
        regions = []
        for slot, (label, reflectance) in enumerate(spec.reference_reflectances[index]):
            box = (40, 30 + slot * 110, 140, 130 + slot * 110)
            _paint(reflectance_map, box, reflectance)
            top, left, bottom, right = box
            regions.append(
                {
                    "label": label.value,
                    "box": [left / width, top / height, right / width, bottom / height],
                }
            )
        chart_mask = np.zeros((height, width), dtype=bool)
        cell = 22
        chart_top, chart_left = height - 4 * cell - 10, width - 6 * cell - 10
        for patch in range(24):
            row, column = divmod(patch, 6)
            box = (
                chart_top + row * cell,
                chart_left + column * cell,
                chart_top + (row + 1) * cell,
                chart_left + (column + 1) * cell,
            )
            _paint(reflectance_map, box, spec.chart_reflectances_kxn[patch])
            chart_mask[box[0] : box[2], box[1] : box[3]] = True
        linear = _render_reflectance_map(reflectance_map, spec, illuminant)
        linear *= 1.0 + rng.normal(0.0, spec.noise_sigma, size=linear.shape)
        scale = 0.8 / float(np.percentile(linear, 99.9))
        linear *= scale
        write_tiff16(directory / f"{stem}.tif", encode_srgb(linear.astype(np.float32)))
        (directory / f"{stem}.regions.json").write_text(json.dumps(regions))
        write_image8(directory / f"{stem}_chart.png", np.repeat(chart_mask[..., None], 3, axis=2))
        patches = sensor_responses(spec.sensitivity_3xn, illuminant, spec.chart_reflectances_kxn)
        np.savetxt(
            directory / f"{stem}_patches.csv", patches * scale, delimiter=",", header="r,g,b"
        )
        truth = spec.sensitivity_3xn @ illuminant
        truth = truth / truth[1]
        rows.append(
            f"{stem}.tif,{truth[0]:.6f},{truth[1]:.6f},{truth[2]:.6f},"
            f"{stem}_chart.png,{stem}_patches.csv"
        )
    (directory / "ground_truth.csv").write_text("\n".join(rows) + "\n")
    return directory
