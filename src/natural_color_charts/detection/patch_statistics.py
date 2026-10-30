"""Stage 5: filter region pixels and compute robust patch colors and reliability weights."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.stats import trim_mean

from natural_color_charts.config import PatchConfig
from natural_color_charts.domain import (
    LinearImage,
    PatchMeasurement,
    ReferenceClass,
    ReferenceRegion,
    SceneContext,
)


def erode_mask(mask: NDArray[np.bool_], radius_px: int) -> NDArray[np.bool_]:
    """Shrink a mask so edge pixels mixed with neighbors are excluded."""
    if radius_px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return np.asarray(cv2.erode(mask.astype(np.uint8), kernel).astype(bool))


def specular_outliers(
    pixels: NDArray[np.float64], luminance_quantile: float, chroma_ratio: float
) -> NDArray[np.bool_]:
    """Very bright pixels that are much less saturated than the region: likely highlights."""
    luminance = pixels.mean(axis=1)
    saturation = 1.0 - pixels.min(axis=1) / np.maximum(pixels.max(axis=1), 1e-12)
    bright = luminance > np.quantile(luminance, luminance_quantile)
    return np.asarray(bright & (saturation < chroma_ratio * np.median(saturation)))


class PatchStatisticsCalculator:
    """Measures each region and assigns weight = vlm × size × exposure × class prior strength."""

    def __init__(self, config: PatchConfig) -> None:
        self._config = config

    def measure(
        self, image: LinearImage, regions: Sequence[ReferenceRegion], context: SceneContext
    ) -> list[PatchMeasurement]:
        """One measurement per region; unusable regions are kept with kept=False and a reason."""
        return [
            self._measure_region(image, region, index, context)
            for index, region in enumerate(regions)
        ]

    def _measure_region(
        self, image: LinearImage, region: ReferenceRegion, index: int, context: SceneContext
    ) -> PatchMeasurement:
        pixels = self.valid_pixels(image, region)
        if pixels.shape[0] < self._config.min_valid_pixels:
            return PatchMeasurement(
                label=region.label,
                rgb=np.zeros(3),
                pixel_count=int(pixels.shape[0]),
                weight=0.0,
                region_index=index,
                kept=False,
                rejection_reason=f"only {pixels.shape[0]} valid pixels",
                source=region.source,
                known_reflectance=region.known_reflectance,
            )
        rgb = self.robust_color(pixels)
        weight = self._weight(region, pixels.shape[0], rgb, context)
        return PatchMeasurement(
            label=region.label,
            rgb=rgb,
            pixel_count=int(pixels.shape[0]),
            weight=weight,
            region_index=index,
            source=region.source,
            known_reflectance=region.known_reflectance,
        )

    def _weight(
        self,
        region: ReferenceRegion,
        count: int,
        rgb: NDArray[np.float64],
        context: SceneContext,
    ) -> float:
        if not self._config.use_reliability_weights:
            return 1.0
        quality = self._size_factor(count) * self._exposure_factor(rgb)
        if region.source == "user":
            # The user vouches for the color, so trust comes from them (as the review agent
            # judged it), not the class prior.
            return float(self._config.user_anchor_weight * quality * region.candidate_confidence)
        return float(
            region.candidate_confidence * quality * self._class_strength(region.label, context)
        )

    def valid_pixels(self, image: LinearImage, region: ReferenceRegion) -> NDArray[np.float64]:
        """Region pixels after erosion, clipping, noise-floor, and specular rejection."""
        radius = round(self._config.erosion_fraction * max(image.width, image.height))
        mask = erode_mask(region.mask, radius) & image.valid_mask
        pixels = image.rgb[mask].astype(np.float64)
        pixels = pixels[pixels.min(axis=1) > self._config.noise_floor]
        if pixels.shape[0] < self._config.min_valid_pixels:
            return pixels
        speculars = specular_outliers(
            pixels, self._config.specular_luminance_quantile, self._config.specular_chroma_ratio
        )
        return pixels[~speculars]

    def robust_color(self, pixels: NDArray[np.float64]) -> NDArray[np.float64]:
        """Median or trimmed mean per channel in linear camera RGB."""
        if self._config.statistic == "median":
            return np.median(pixels, axis=0)
        return np.asarray(trim_mean(pixels, self._config.trim_fraction, axis=0))

    def _size_factor(self, count: int) -> float:
        return min(1.0, count / self._config.full_weight_pixels)

    def _exposure_factor(self, rgb: NDArray[np.float64]) -> float:
        # Dark patches are dominated by noise and black-level error.
        return min(1.0, float(rgb.mean()) / self._config.full_weight_luminance)

    def _class_strength(self, label: ReferenceClass, context: SceneContext) -> float:
        strength = self._config.class_prior_strength.get(label.value, 0.5)
        # Optical brighteners fluoresce under UV-rich daylight, making paper read bluer.
        if label is ReferenceClass.PAPER and context.environment == "outdoor":
            strength *= self._config.outdoor_paper_factor
        return strength
