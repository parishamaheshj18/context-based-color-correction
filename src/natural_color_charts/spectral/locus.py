"""Find the illuminant-family spectrum whose camera white best matches a white point."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from natural_color_charts.colorimetry import angular_error_deg
from natural_color_charts.domain import RgbVector, SpectralArray
from natural_color_charts.spectral.illuminants import IlluminantFamily
from natural_color_charts.spectral.rendering import white_response


@dataclass(frozen=True, eq=False)
class LocusMatch:
    """Closest member of an illuminant family to a camera white."""

    family: str
    cct_k: float
    spectrum: SpectralArray
    angular_error_deg: float


def match_family(
    white_rgb: RgbVector, sensitivity_3xn: SpectralArray, family: IlluminantFamily
) -> LocusMatch:
    """Search the family's CCT range (in mired) for the best angular match."""

    def error(mired_value: float) -> float:
        spectrum = family.spd(1e6 / mired_value)
        return angular_error_deg(white_response(sensitivity_3xn, spectrum), white_rgb)

    low, high = 1e6 / family.cct_max_k, 1e6 / family.cct_min_k
    grid = np.linspace(low, high, 41)
    best = grid[int(np.argmin([error(m) for m in grid]))]
    step = grid[1] - grid[0]
    result = minimize_scalar(
        error, bounds=(max(low, best - step), min(high, best + step)), method="bounded"
    )
    cct = 1e6 / float(result.x)
    return LocusMatch(family.name, cct, family.spd(cct), float(result.fun))


def match_locus(
    white_rgb: RgbVector, sensitivity_3xn: SpectralArray, families: Sequence[IlluminantFamily]
) -> LocusMatch:
    """Best match across several illuminant families."""
    matches = [match_family(white_rgb, sensitivity_3xn, family) for family in families]
    return min(matches, key=lambda match: match.angular_error_deg)
