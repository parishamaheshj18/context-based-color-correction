"""Starter priors built from spectra bundled with colour-science.

These let the pipeline run before real spectral libraries (ECOSTRESS, skin datasets) are
downloaded. They rest on few samples; replace them with scripts/build_priors.py.
"""

from __future__ import annotations

import numpy as np
from colour.quality import SDS_TCS

from natural_color_charts.config import PriorConfig
from natural_color_charts.domain import (
    ReferenceClass,
    SpectralArray,
    WavelengthGrid,
    resample_spectrum,
)
from natural_color_charts.spectral.datasets import chart_reflectances
from natural_color_charts.spectral.priors import (
    ReflectancePrior,
    add_basis_vector,
    build_pca_prior,
    build_smooth_prior,
)

SPECTRAL_CHARTS = ("BabelColor Average", "ColorChecker N Ohta", "ISO 17321-1")
SKIN_PATCHES = {"dark skin", "light skin", "Caucasian", "Oriental", "South Asian", "African"}
FOLIAGE_PATCHES = {"foliage", "Summer Grass"}
# Measured near-neutral surfaces on the charts stand in for white/gray plastics and paints.
NEUTRAL_PATCHES = {
    "white 9.5 (.05 D)", "neutral 8 (.23 D)", "neutral 6.5 (.44 D)", "neutral 5 (.70 D)",
    "neutral 3.5 (1.05 D)", "White", "Gray-80", "Gray-70", "Gray-50", "Gray-35",
}  # fmt: skip
# Man-made neutrals vary more than paper (off-white plastics, warm grays), so the prior is wider.
NEUTRAL_SURFACE_SIGMA_FACTOR = 2.0
SOURCE = "bootstrap (colour-science charts; replace with measured libraries)"


def chart_patches(grid: WavelengthGrid, names: set[str]) -> SpectralArray:
    """Spectra of the named patches across all bundled spectral charts."""
    spectra: list[SpectralArray] = []
    for chart in (*SPECTRAL_CHARTS, "PMC"):
        patch_names, values = chart_reflectances(chart, grid)
        spectra.extend(values[i] for i, name in enumerate(patch_names) if name in names)
    return np.stack(spectra)


def cie_test_color_sample(name: str, grid: WavelengthGrid) -> SpectralArray:
    """A CIE 1995 test color sample, e.g. TCS13 (skin) or TCS14 (leaf green)."""
    # colour-science 0.4.7 groups the samples by edition; 0.4.6 keys them directly.
    samples = SDS_TCS.get("CIE 1995", SDS_TCS)
    sd = samples[name]
    return resample_spectrum(sd.wavelengths, sd.values, grid)


def neutral_mean(grid: WavelengthGrid, level: float, slope: float) -> SpectralArray:
    """Near-flat reflectance with a gentle linear tilt toward long wavelengths."""
    x = np.linspace(-0.5, 0.5, grid.size)
    return level * (1.0 + slope * x)


def brightener_bump(grid: WavelengthGrid) -> SpectralArray:
    """Excess blue reflectance of optical brighteners (fluorescence peaks near 440 nm)."""
    return np.exp(-0.5 * ((grid.wavelengths_nm - 440.0) / 20.0) ** 2)


def build_bootstrap_priors(
    grid: WavelengthGrid, config: PriorConfig
) -> dict[ReferenceClass, ReflectancePrior]:
    """One starter prior per reference class."""
    k, sigma = config.components, config.relative_shape_sigma
    skin = np.vstack([chart_patches(grid, SKIN_PATCHES), cie_test_color_sample("TCS13", grid)])
    foliage = np.vstack(
        [chart_patches(grid, FOLIAGE_PATCHES), cie_test_color_sample("TCS14", grid)]
    )
    paper = build_smooth_prior(
        ReferenceClass.PAPER, neutral_mean(grid, 0.85, 0.0), grid, k, sigma, SOURCE
    )
    return {
        ReferenceClass.SKIN: _measured_prior(ReferenceClass.SKIN, skin, grid, config),
        ReferenceClass.FOLIAGE: _measured_prior(ReferenceClass.FOLIAGE, foliage, grid, config),
        ReferenceClass.CONCRETE: build_smooth_prior(
            ReferenceClass.CONCRETE, neutral_mean(grid, 0.3, 0.15), grid, k, sigma, SOURCE
        ),
        ReferenceClass.PAPER: add_basis_vector(paper, brightener_bump(grid), 2.0 * sigma),
        ReferenceClass.SNOW: build_smooth_prior(
            ReferenceClass.SNOW, neutral_mean(grid, 0.9, -0.03), grid, k, sigma, SOURCE
        ),
        ReferenceClass.NEUTRAL_SURFACE: _neutral_surface_prior(grid, config),
        ReferenceClass.CLOUD: build_smooth_prior(
            ReferenceClass.CLOUD, neutral_mean(grid, 0.8, 0.0), grid, k, 1.5 * sigma, SOURCE
        ),
    }


def _neutral_surface_prior(grid: WavelengthGrid, config: PriorConfig) -> ReflectancePrior:
    # Only spectral shape matters (brightness is absorbed per patch), so anchors share one level.
    anchors = chart_patches(grid, NEUTRAL_PATCHES)
    anchors = 0.6 * anchors / anchors.mean(axis=1, keepdims=True)
    return build_smooth_prior(
        ReferenceClass.NEUTRAL_SURFACE,
        anchors.mean(axis=0),
        grid,
        config.components,
        NEUTRAL_SURFACE_SIGMA_FACTOR * config.relative_shape_sigma,
        SOURCE,
        anchors_mxn=anchors,
    )


def _measured_prior(
    label: ReferenceClass, spectra: SpectralArray, grid: WavelengthGrid, config: PriorConfig
) -> ReflectancePrior:
    if spectra.shape[0] >= config.min_samples_for_pca:
        return build_pca_prior(label, spectra, config.components, SOURCE)
    return build_smooth_prior(
        label,
        spectra.mean(axis=0),
        grid,
        config.components,
        config.relative_shape_sigma,
        SOURCE,
        anchors_mxn=spectra,
    )
