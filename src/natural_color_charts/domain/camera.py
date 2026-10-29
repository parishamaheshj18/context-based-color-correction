"""Camera spectral sensitivity object."""

from __future__ import annotations

from dataclasses import dataclass

from natural_color_charts.domain.spectral import SpectralArray


@dataclass(frozen=True, eq=False)
class CameraSensitivity:
    """Per-channel spectral sensitivity sampled on the shared wavelength grid."""

    name: str
    sensitivity_3xn: SpectralArray
    selection_reason: str = ""
    is_approximate: bool = False
