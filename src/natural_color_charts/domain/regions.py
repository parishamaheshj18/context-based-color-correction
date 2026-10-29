"""Reference regions (Stage 4) and their patch measurements (Stage 5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from natural_color_charts.domain.scene import BoxXyxy, ReferenceClass
from natural_color_charts.domain.spectral import RgbVector

__all__ = ["BoxXyxy", "Detection", "PatchMeasurement", "ReferenceRegion"]


@dataclass(frozen=True)
class Detection:
    """A labeled box in preview pixel coordinates."""

    label: ReferenceClass
    box_xyxy: BoxXyxy
    score: float
    # Set when the box comes from the scene analysis itself, which already judged the region.
    candidate_confidence: float | None = None
    description: str = ""


@dataclass(frozen=True, eq=False)
class ReferenceRegion:
    """A segmented reference region at full image resolution."""

    label: ReferenceClass
    mask: NDArray[np.bool_]
    detector_score: float
    box_xyxy: BoxXyxy | None = None
    candidate_confidence: float = 0.5
    description: str = ""


@dataclass(frozen=True, eq=False)
class PatchMeasurement:
    """Robust color of one region and how much the estimator should trust it."""

    label: ReferenceClass
    rgb: RgbVector
    pixel_count: int
    weight: float
    region_index: int = -1
    kept: bool = True
    rejection_reason: str = ""
