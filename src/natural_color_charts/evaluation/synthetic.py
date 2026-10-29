"""Synthetic test harness: render known scenes so estimators can be checked exactly."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from natural_color_charts.domain import (
    ImageMetadata,
    InputMode,
    LinearImage,
    ReferenceClass,
    ReferenceRegion,
    SpectralArray,
)
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
