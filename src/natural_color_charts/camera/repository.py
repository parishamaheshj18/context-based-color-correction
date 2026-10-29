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
    tokens = [t for t in re.findall(r"[a-z]+|[0-9]+", name.lower()) if t not in CORPORATE_WORDS]
    return " ".join(dict.fromkeys(tokens))


def compact_camera_name(name: str) -> str:
    """Normalized name without spaces, so "iPhone12ProMAX" equals "iPhone 12 Pro Max"."""
    return normalize_camera_name(name).replace(" ", "")


def model_numbers(name: str) -> frozenset[str]:
    """Digit groups in a camera name; cameras with different model numbers never match."""
    return frozenset(re.findall(r"[0-9]+", name))


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


def read_headerless_table(
    rows: list[list[float]], start_nm: float, step_nm: float, grid: WavelengthGrid
) -> SpectralArray:
    """R, G, B columns sampled every step_nm from start_nm, as in the Oh-lab smartphone database."""
    table = np.array([row[:3] for row in rows], dtype=float)
    wavelengths = start_nm + step_nm * np.arange(table.shape[0], dtype=np.float64)
    return resample_spectrum(wavelengths, table.T, grid)


def read_sensitivity_xlsx(path: Path) -> list[list[float]]:
    """Numeric rows of the first worksheet (needs openpyxl)."""
    try:
        import openpyxl
    except ImportError as error:
        raise ImportError("Reading .xlsx sensitivities needs: pip install openpyxl") from error
    sheet = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    return [
        [float(v) for v in row[:3]]
        for row in sheet.iter_rows(values_only=True)
        if row and all(isinstance(v, int | float) for v in row[:3])
    ]


def read_numeric_text(path: Path) -> list[list[float]]:
    """Whitespace-separated numeric rows; returns [] if any line is not numeric."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if not all(_is_number(field) for field in fields):
            return []
        rows.append([float(field) for field in fields])
    return rows


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

    def __init__(
        self,
        grid: WavelengthGrid,
        directory: Path,
        include_builtin: bool,
        headerless_start_nm: float = 400.0,
        headerless_step_nm: float = 10.0,
    ) -> None:
        self._grid = grid
        self._headerless = (headerless_start_nm, headerless_step_nm)
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
        start, step = self._headerless
        for path in sorted(directory.glob("*.xlsx")):
            if not path.name.startswith("~$"):
                curves[path.stem] = read_headerless_table(
                    read_sensitivity_xlsx(path), start, step, self._grid
                )
        for path in sorted(directory.glob("*.txt")):
            rows = read_numeric_text(path)
            if rows and len(rows[0]) == 3:
                curves[path.stem] = read_headerless_table(rows, start, step, self._grid)
            else:
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
