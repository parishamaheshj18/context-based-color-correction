"""Downscaled, display-ready previews for the VLM, the detector, and overlays."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from natural_color_charts.colorimetry import encode_srgb
from natural_color_charts.domain.image import LinearImage
from natural_color_charts.io.writers import to_uint8


def resize_to_long_edge(rgb: NDArray[np.float32], long_edge_px: int) -> NDArray[np.float32]:
    """Downscale so the longer side is at most long_edge_px."""
    height, width = rgb.shape[:2]
    scale = long_edge_px / max(height, width)
    if scale >= 1.0:
        return rgb
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), dtype=np.float32)


def render_preview(image: LinearImage, long_edge_px: int) -> NDArray[np.uint8]:
    """Render a recognizable 8-bit preview: as-shot white balance, auto exposure, sRGB curve."""
    rgb = resize_to_long_edge(image.rgb, long_edge_px)
    white = image.metadata.as_shot_white_rgb
    if white is not None:
        rgb = np.divide(rgb, np.asarray(white, dtype=np.float32), dtype=np.float32)
    # Previews only need to be recognizable, so a simple 99th-percentile exposure is enough.
    level = float(np.percentile(rgb.max(axis=2), 99)) or 1.0
    return to_uint8(encode_srgb(rgb / level))
