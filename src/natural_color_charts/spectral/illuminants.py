"""Parametric illuminant families on the shared wavelength grid."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from colour.colorimetry import SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES

from natural_color_charts.domain import SpectralArray, WavelengthGrid, resample_spectrum

PLANCK_C2_M_K = 1.4388e-2
NORMALIZATION_WAVELENGTH_NM = 560.0


def normalize_spd(spd: SpectralArray, grid: WavelengthGrid) -> SpectralArray:
    """Scale an SPD to 1 at 560 nm, the conventional normalization point."""
    return spd / np.interp(NORMALIZATION_WAVELENGTH_NM, grid.wavelengths_nm, spd)


def daylight_chromaticity(cct_k: float) -> tuple[float, float]:
    """CIE daylight locus chromaticity (CIE 15:2004, valid for 4000–25000 K)."""
    t = float(np.clip(cct_k, 4000.0, 25000.0))
    if t <= 7000.0:
        x = -4.6070e9 / t**3 + 2.9678e6 / t**2 + 0.09911e3 / t + 0.244063
    else:
        x = -2.0064e9 / t**3 + 1.9018e6 / t**2 + 0.24748e3 / t + 0.237040
    return x, -3.0 * x**2 + 2.870 * x - 0.275


class DaylightBasis:
    """CIE S0, S1, S2 daylight basis functions resampled to a grid."""

    def __init__(self, grid: WavelengthGrid) -> None:
        self._grid = grid
        self._basis_3xn = np.stack(
            [
                resample_spectrum(sd.wavelengths, sd.values, grid)
                for sd in (
                    SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES[k] for k in ("S0", "S1", "S2")
                )
            ]
        )

    def spd(self, cct_k: float) -> SpectralArray:
        x, y = daylight_chromaticity(cct_k)
        m = 0.0241 + 0.2562 * x - 0.7341 * y
        m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / m
        m2 = (0.0300 - 31.4424 * x + 30.0717 * y) / m
        return normalize_spd(np.array([1.0, m1, m2]) @ self._basis_3xn, self._grid)


def blackbody_spd(temperature_k: float, grid: WavelengthGrid) -> SpectralArray:
    """Planckian radiator SPD; models tungsten and halogen light."""
    wavelengths_m = grid.wavelengths_nm * 1e-9
    radiance = wavelengths_m**-5 / np.expm1(PLANCK_C2_M_K / (wavelengths_m * temperature_k))
    return normalize_spd(radiance, grid)


class IlluminantFamily(Protocol):
    """A one-parameter family of illuminant spectra indexed by CCT."""

    name: str
    cct_min_k: float
    cct_max_k: float

    def spd(self, cct_k: float) -> SpectralArray: ...


class DaylightFamily:
    """CIE D-series daylight."""

    name = "daylight"

    def __init__(self, grid: WavelengthGrid, cct_min_k: float, cct_max_k: float) -> None:
        self._basis = DaylightBasis(grid)
        self.cct_min_k = cct_min_k
        self.cct_max_k = cct_max_k

    def spd(self, cct_k: float) -> SpectralArray:
        return self._basis.spd(cct_k)


class BlackbodyFamily:
    """Planckian radiators (tungsten, halogen, candle)."""

    name = "blackbody"

    def __init__(self, grid: WavelengthGrid, cct_min_k: float, cct_max_k: float) -> None:
        self._grid = grid
        self.cct_min_k = cct_min_k
        self.cct_max_k = cct_max_k

    def spd(self, cct_k: float) -> SpectralArray:
        return blackbody_spd(cct_k, self._grid)


def mired(cct_k: float) -> float:
    """Micro reciprocal degrees, a perceptually more uniform temperature scale."""
    return 1e6 / cct_k
