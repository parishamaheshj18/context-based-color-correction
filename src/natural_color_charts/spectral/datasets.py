"""Reference spectral data from colour-science, resampled to the shared grid."""

from __future__ import annotations

import colour
import numpy as np
from colour.characterisation import MSDS_CAMERA_SENSITIVITIES

from natural_color_charts.domain import SpectralArray, WavelengthGrid, resample_spectrum

CIE_1931_OBSERVER = "CIE 1931 2 Degree Standard Observer"


def cie_1931_cmfs(grid: WavelengthGrid) -> SpectralArray:
    """CIE 1931 color matching functions as a 3×N array."""
    cmfs = colour.MSDS_CMFS[CIE_1931_OBSERVER]
    return resample_spectrum(cmfs.wavelengths, cmfs.values.T, grid)


def standard_illuminant_spd(name: str, grid: WavelengthGrid) -> SpectralArray:
    """A CIE standard illuminant SPD such as D65 or A."""
    sd = colour.SDS_ILLUMINANTS[name]
    return resample_spectrum(sd.wavelengths, sd.values, grid)


def chart_reflectances(name: str, grid: WavelengthGrid) -> tuple[list[str], SpectralArray]:
    """Patch names and K×N reflectances of a spectral color chart."""
    patches = colour.SDS_COLOURCHECKERS[name]
    names = list(patches)
    values = np.stack(
        [resample_spectrum(sd.wavelengths, sd.values, grid) for sd in patches.values()]
    )
    return names, values


def builtin_camera_sensitivities(grid: WavelengthGrid) -> dict[str, SpectralArray]:
    """Camera sensitivities bundled with colour-science, keyed by camera name."""
    cameras: dict[str, SpectralArray] = {}
    for label, msds in MSDS_CAMERA_SENSITIVITIES.items():
        name = label.replace(" (NPL)", "")
        cameras[name] = resample_spectrum(msds.wavelengths, msds.values.T, grid)
    return cameras
