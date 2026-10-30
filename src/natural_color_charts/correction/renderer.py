"""Stage 9 rendering: apply the global transform and encode for display."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from natural_color_charts.colorimetry import (
    XYZ_TO_LINEAR_SRGB,
    apply_matrix,
    decode_srgb,
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


def compress_highlights(linear: NDArray[np.float32], knee: float) -> NDArray[np.float32]:
    """Roll off pixels brighter than knee smoothly toward 1; darker pixels are unchanged.

    The whole pixel is scaled by one factor, so its hue is kept, unlike a per-channel clip that
    turns a bright blue sky cyan or white. The curve has slope 1 at the knee, so there is no edge.
    """
    peak = linear.max(axis=-1, keepdims=True)
    span = 1.0 - knee
    rolled = knee + span * (1.0 - np.exp(-(peak - knee) / span))
    scale = np.where(peak > knee, rolled / np.maximum(peak, 1e-12), 1.0)
    return np.asarray(linear * scale, dtype=np.float32)


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
        self,
        exposure_percentile: float,
        exposure_target: float,
        normalize_raster: bool,
        highlight_knee: float,
        saturation_threshold: float,
    ) -> None:
        self._percentile = exposure_percentile
        self._target = exposure_target
        self._normalize_raster = normalize_raster
        self._knee = highlight_knee
        self._clipped_linear = float(decode_srgb(np.array([saturation_threshold]))[0])

    def render(
        self, image: LinearImage, transform: CorrectionTransform, gain: float | None = None
    ) -> RenderedImage:
        """Render; pass gain to reuse another render's exposure so only color differs.

        Every version of a photo (as shot, baselines, corrected) goes through these same steps,
        so a before/after comparison differs only in color.
        """
        camera_to_srgb = XYZ_TO_LINEAR_SRGB @ transform.camera_to_xyz
        linear = apply_matrix(image.rgb, camera_to_srgb)
        if gain is None:
            gain = self._auto_gain(image, linear)
        linear = np.maximum(linear * np.float32(gain), 0.0)
        lost = self._color_lost(image)
        linear[lost] = linear[lost].max(axis=1, keepdims=True)
        linear = np.clip(compress_highlights(linear, self._knee), 0.0, 1.0)
        return RenderedImage(
            linear_srgb=linear, encoded_srgb=encode_srgb(linear), exposure_gain=gain
        )

    def _color_lost(self, image: LinearImage) -> NDArray[np.bool_]:
        """Pixels whose true color the camera lost; they render white instead of tinted.

        A clipped RAW sensor channel loses the color. In a display-referred file one channel at
        its maximum is usually just a vivid color (a blue sky); only all three at the maximum
        means the camera clipped to white.
        """
        if image.mode is InputMode.RAW:
            return image.saturation_mask
        return np.asarray(image.saturation_mask & (image.rgb.min(axis=2) >= self._clipped_linear))

    def _auto_gain(self, image: LinearImage, linear: NDArray[np.float32]) -> float:
        if image.mode is InputMode.RAW or self._normalize_raster:
            return exposure_gain(linear, image.valid_mask, self._percentile, self._target)
        return 1.0
