"""Stage 9 rendering: apply the global transform and encode for display."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from natural_color_charts.colorimetry import (
    XYZ_TO_LINEAR_SRGB,
    apply_matrix,
    encode_srgb,
    luminance_from_linear_srgb,
)
from natural_color_charts.domain import CorrectionTransform, InputMode, LinearImage


@dataclass(frozen=True, eq=False)
class RenderedImage:
    """Corrected image in linear and sRGB-encoded form."""

    linear_srgb: NDArray[np.float32]
    encoded_srgb: NDArray[np.float32]
    exposure_gain: float


def exposure_gain(
    linear_srgb: NDArray[np.float32], valid: NDArray[np.bool_], percentile: float, target: float
) -> float:
    """Gain that puts the given luminance percentile of valid pixels at target."""
    luminance = luminance_from_linear_srgb(linear_srgb)[valid]
    level = float(np.percentile(luminance, percentile)) if luminance.size else 0.0
    return target / level if level > 0 else 1.0


class SrgbRenderer:
    """Renders camera RGB to sRGB with only global operations, so no content is invented."""

    def __init__(
        self, exposure_percentile: float, exposure_target: float, normalize_raster: bool
    ) -> None:
        self._percentile = exposure_percentile
        self._target = exposure_target
        self._normalize_raster = normalize_raster

    def render(
        self, image: LinearImage, transform: CorrectionTransform, gain: float | None = None
    ) -> RenderedImage:
        """Render; pass gain to reuse another render's exposure so only color differs."""
        camera_to_srgb = XYZ_TO_LINEAR_SRGB @ transform.camera_to_xyz
        linear = apply_matrix(image.rgb, camera_to_srgb)
        if gain is None:
            gain = self._auto_gain(image, linear)
        linear = np.clip(linear * np.float32(gain), 0.0, 1.0)
        # Saturated pixels lost their true color; rendering them white avoids colored highlights.
        linear[image.saturation_mask] = np.maximum(
            linear[image.saturation_mask], linear[image.saturation_mask].max(axis=1, keepdims=True)
        )
        return RenderedImage(
            linear_srgb=linear, encoded_srgb=encode_srgb(linear), exposure_gain=gain
        )

    def _auto_gain(self, image: LinearImage, linear: NDArray[np.float32]) -> float:
        if image.mode is InputMode.RAW or self._normalize_raster:
            return exposure_gain(linear, image.valid_mask, self._percentile, self._target)
        return 1.0
