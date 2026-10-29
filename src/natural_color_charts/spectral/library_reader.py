"""Readers for measured reflectance libraries used to build priors (Stage 6)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from natural_color_charts.config import PriorConfig
from natural_color_charts.domain import (
    ReferenceClass,
    SpectralArray,
    WavelengthGrid,
    resample_spectrum,
)
from natural_color_charts.spectral.priors import ReflectancePrior, build_pca_prior

logger = logging.getLogger(__name__)


def _numeric_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    delimiter = "," if path.suffix.lower() == ".csv" else None
    for line in path.read_text(errors="replace").splitlines():
        fields = next(csv.reader([line])) if delimiter else line.split()
        try:
            rows.append([float(field) for field in fields if field.strip()])
        except ValueError:
            continue  # header and metadata lines
    return [row for row in rows if len(row) >= 2]


def read_spectra_file(path: Path) -> tuple[SpectralArray, SpectralArray]:
    """Wavelengths (nm) and M×W reflectances (0–1) from a CSV or ECOSTRESS-style text file.

    Columns: wavelength, then one column per sample. Wavelengths in µm and reflectance in
    percent (the ECOSTRESS convention) are detected and converted.
    """
    table = np.array(_numeric_rows(path))
    if table.size == 0:
        raise ValueError(f"No numeric spectral data in {path}")
    wavelengths = table[:, 0] * (1000.0 if table[:, 0].max() < 30 else 1.0)
    values = table[:, 1:].T
    if values.max() > 1.5:
        values = values / 100.0
    return wavelengths, values


def load_class_spectra(directory: Path, grid: WavelengthGrid) -> SpectralArray:
    """All spectra in a class directory, restricted to files covering the grid."""
    spectra: list[SpectralArray] = []
    for path in sorted(p for p in directory.iterdir() if p.suffix.lower() in {".csv", ".txt"}):
        wavelengths, values = read_spectra_file(path)
        if np.min(wavelengths) > grid.start_nm or np.max(wavelengths) < grid.stop_nm:
            logger.warning(
                "Skipping %s: does not cover %s-%s nm", path, grid.start_nm, grid.stop_nm
            )
            continue
        spectra.extend(resample_spectrum(wavelengths, values, grid))
    if not spectra:
        raise ValueError(f"No usable spectra in {directory}")
    return np.stack(spectra)


def build_priors_from_library(
    library_dir: Path, grid: WavelengthGrid, config: PriorConfig
) -> list[ReflectancePrior]:
    """PCA priors for every class subdirectory (e.g. library/foliage/*.csv)."""
    priors = []
    for label in ReferenceClass:
        class_dir = library_dir / label.value
        if not class_dir.is_dir():
            continue
        spectra = load_class_spectra(class_dir, grid)
        if spectra.shape[0] < config.min_samples_for_pca:
            logger.warning("%s has only %d spectra; skipping", label.value, spectra.shape[0])
            continue
        priors.append(build_pca_prior(label, spectra, config.components, f"library: {class_dir}"))
    return priors
