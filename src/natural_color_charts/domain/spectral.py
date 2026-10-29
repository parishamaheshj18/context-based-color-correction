"""Wavelength grid and spectral array types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SpectralArray = NDArray[np.float64]
RgbVector = NDArray[np.float64]
Matrix3x3 = NDArray[np.float64]


@dataclass(frozen=True)
class WavelengthGrid:
    """Shared, evenly spaced wavelength sampling used by every spectral quantity."""

    start_nm: float
    stop_nm: float
    step_nm: float

    @property
    def wavelengths_nm(self) -> SpectralArray:
        count = round((self.stop_nm - self.start_nm) / self.step_nm) + 1
        return np.linspace(self.start_nm, self.stop_nm, count)

    @property
    def size(self) -> int:
        return int(self.wavelengths_nm.size)


def resample_spectrum(
    source_wavelengths_nm: SpectralArray, source_values: SpectralArray, grid: WavelengthGrid
) -> SpectralArray:
    """Linearly interpolate a spectrum (or a stack of spectra on the last axis) onto the grid."""
    target = grid.wavelengths_nm
    values = np.atleast_2d(np.asarray(source_values, dtype=np.float64))
    order = np.argsort(source_wavelengths_nm)
    wavelengths = np.asarray(source_wavelengths_nm, dtype=np.float64)[order]
    resampled = np.stack([np.interp(target, wavelengths, row[order]) for row in values])
    return resampled if np.ndim(source_values) > 1 else resampled[0]
