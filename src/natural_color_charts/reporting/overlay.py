"""Stage 10 reference overlay (no matplotlib, so it also runs in the browser)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from natural_color_charts.domain import ReferenceClass
from natural_color_charts.pipeline import PipelineResult

LABEL_COLORS: dict[ReferenceClass, tuple[int, int, int]] = {
    ReferenceClass.FOLIAGE: (40, 200, 60),
    ReferenceClass.SKIN: (240, 140, 60),
    ReferenceClass.CONCRETE: (150, 150, 150),
    ReferenceClass.PAPER: (60, 120, 255),
    ReferenceClass.SNOW: (0, 220, 220),
    ReferenceClass.CLOUD: (200, 80, 220),
    ReferenceClass.NEUTRAL_SURFACE: (255, 200, 40),
    ReferenceClass.KNOWN_COLOR: (255, 255, 255),
}
REJECTED_COLOR = (230, 30, 30)


def render_overlay(result: PipelineResult, alpha_max: float) -> NDArray[np.uint8]:
    """Preview with each reference mask tinted by label; opacity is proportional to weight."""
    overlay = result.preview.astype(np.float32)
    height, width = overlay.shape[:2]
    max_weight = max((p.weight for p in result.patches if p.kept), default=1.0) or 1.0
    for patch in result.patches:
        if not 0 <= patch.region_index < len(result.regions):
            continue
        mask = cv2.resize(
            result.regions[patch.region_index].mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        color = np.array(LABEL_COLORS[patch.label] if patch.kept else REJECTED_COLOR, np.float32)
        alpha = alpha_max * (patch.weight / max_weight if patch.kept else 0.5)
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
        _outline(overlay, mask, color)
    return np.asarray(np.clip(overlay, 0, 255), dtype=np.uint8)


def _outline(
    image: NDArray[np.float32], mask: NDArray[np.bool_], color: NDArray[np.float32]
) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, color.tolist(), 1)
