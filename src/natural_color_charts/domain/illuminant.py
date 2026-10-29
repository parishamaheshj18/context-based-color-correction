"""Illuminant estimate and correction transform objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from natural_color_charts.domain.spectral import Matrix3x3, RgbVector, SpectralArray


def normalize_white(rgb: RgbVector) -> RgbVector:
    """Scale a camera white so its green channel is 1."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return rgb / rgb[1]


@dataclass(frozen=True, eq=False)
class IlluminantEstimate:
    """Scene illuminant as a camera-RGB white point, optionally with a spectrum."""

    white_rgb: RgbVector
    method: str
    confidence: float
    family: str | None = None
    cct_k: float | None = None
    spectrum: SpectralArray | None = None
    cct_uncertainty_k: float | None = None
    angular_uncertainty_deg: float | None = None
    is_fallback: bool = False
    used_labels: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default=())


@dataclass(frozen=True, eq=False)
class CorrectionTransform:
    """Camera RGB to XYZ matrix plus the whites it was built for."""

    camera_to_xyz: Matrix3x3
    source_white_xyz: RgbVector
    target_white_xyz: RgbVector
    method: str
    adaptation_degree: float = 1.0
    notes: tuple[str, ...] = field(default=())
