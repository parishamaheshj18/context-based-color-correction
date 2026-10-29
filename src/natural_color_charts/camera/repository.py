"""Stage 2: camera spectral sensitivity repository."""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import numpy as np

from natural_color_charts.colorimetry import XYZ_TO_LINEAR_SRGB
from natural_color_charts.domain import (
    CameraSensitivity,
    SpectralArray,
    WavelengthGrid,
    resample_spectrum,
)
from natural_color_charts.spectral.datasets import builtin_camera_sensitivities, cie_1931_cmfs

logger = logging.getLogger(__name__)

CORPORATE_WORDS = frozenset({"corporation", "corp", "co", "ltd", "inc", "company", "imaging"})
SRGB_EQUIVALENT_NAME = "sRGB-equivalent (CIE 1931)"
# Jiang et al. (2013) camspec database samples 400–720 nm in 10 nm steps.
JIANG_WAVELENGTHS_NM = np.arange(400.0, 721.0, 10.0)


def normalize_camera_name(name: str) -> str:
    """Lowercase alphanumeric tokens without corporate suffixes or repeats.

    EXIF often repeats the make ("NIKON CORPORATION" + "NIKON D5100").
    """
    tokens = [t for t in re.findall(r"[a-z0-9]+", name.lower()) if t not in CORPORATE_WORDS]
    return " ".join(dict.fromkeys(tokens))


def srgb_equivalent_sensitivity(grid: WavelengthGrid) -> SpectralArray:
    """Sensitivities whose responses are linear sRGB, for display-referred inputs."""
    return XYZ_TO_LINEAR_SRGB @ cie_1931_cmfs(grid)


def read_sensitivity_csv(path: Path, grid: WavelengthGrid) -> SpectralArray:
    """Read a CSV with columns wavelength, red, green, blue (header row optional)."""
    with path.open(newline="") as handle:
        rows = [row for row in csv.reader(handle) if row and not row[0].startswith("#")]
    if not _is_number(rows[0][0]):
        rows = rows[1:]
    table = np.array([[float(value) for value in row[:4]] for row in rows])
    return resample_spectrum(table[:, 0], table[:, 1:4].T, grid)


def read_jiang_database(path: Path, grid: WavelengthGrid) -> dict[str, SpectralArray]:
    """Parse the camspec text format: a name line followed by R, G, and B value lines."""
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    cameras: dict[str, SpectralArray] = {}
    for start in range(0, len(lines) - 3, 4):
        channels = [np.array(lines[start + c].split(), dtype=float) for c in (1, 2, 3)]
        cameras[lines[start]] = resample_spectrum(JIANG_WAVELENGTHS_NM, np.stack(channels), grid)
    return cameras


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


class CameraSensitivityRepository:
    """Loads sensitivities from data files and colour-science, normalized to a peak of 1."""

    def __init__(self, grid: WavelengthGrid, directory: Path, include_builtin: bool) -> None:
        self._grid = grid
        self._curves: dict[str, SpectralArray] = {}
        if include_builtin:
            self._curves.update(builtin_camera_sensitivities(grid))
        self._curves.update(self._load_directory(directory))

    def _load_directory(self, directory: Path) -> dict[str, SpectralArray]:
        if not directory.is_dir():
            logger.info("Sensitivity directory %s not found", directory)
            return {}
        curves: dict[str, SpectralArray] = {}
        for path in sorted(directory.glob("*.csv")):
            curves[path.stem.replace("_", " ")] = read_sensitivity_csv(path, self._grid)
        for path in sorted(directory.glob("*.txt")):
            curves.update(read_jiang_database(path, self._grid))
        return curves

    def list_cameras(self) -> list[str]:
        """Names of all cameras with sensitivity data."""
        return sorted(self._curves)

    def get_sensitivity(self, name: str) -> CameraSensitivity:
        """Sensitivity for an exact repository name."""
        curves = self._curves[name]
        return CameraSensitivity(name=name, sensitivity_3xn=curves / np.max(curves))

    def srgb_equivalent(self) -> CameraSensitivity:
        """sRGB-equivalent sensitivity for display-referred images."""
        curves = srgb_equivalent_sensitivity(self._grid)
        return CameraSensitivity(SRGB_EQUIVALENT_NAME, curves / np.max(curves), is_approximate=True)
