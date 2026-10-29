"""Stage 6: low-dimensional reflectance priors R = μ + U·b, b ~ N(0, diag(σ²))."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre

from natural_color_charts.domain import ReferenceClass, SpectralArray, WavelengthGrid

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class ReflectancePrior:
    """Gaussian subspace model of one material class's reflectance spectra."""

    label: ReferenceClass
    mean_n: SpectralArray
    basis_nxk: SpectralArray
    variances_k: SpectralArray
    explained_variance_ratio: float
    sample_count: int
    source: str

    @property
    def components(self) -> int:
        return int(self.variances_k.size)


def build_pca_prior(
    label: ReferenceClass, spectra_mxn: SpectralArray, components: int, source: str
) -> ReflectancePrior:
    """Principal-component prior from measured spectra."""
    mean = spectra_mxn.mean(axis=0)
    centered = spectra_mxn - mean
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    count = min(components, vh.shape[0], max(1, spectra_mxn.shape[0] - 1))
    variances = singular**2 / max(1, spectra_mxn.shape[0] - 1)
    total = variances.sum()
    return ReflectancePrior(
        label=label,
        mean_n=mean,
        basis_nxk=vh[:count].T,
        variances_k=np.maximum(variances[:count], 1e-10),
        explained_variance_ratio=float(variances[:count].sum() / total) if total > 0 else 1.0,
        sample_count=int(spectra_mxn.shape[0]),
        source=source,
    )


def smooth_basis(grid: WavelengthGrid, components: int) -> SpectralArray:
    """Orthonormal low-order Legendre polynomials over the grid (N×k): smooth spectral changes."""
    x = np.linspace(-1.0, 1.0, grid.size)
    columns = [legendre.legval(x, np.eye(components)[degree]) for degree in range(components)]
    basis, _ = np.linalg.qr(np.stack(columns, axis=1))
    return basis * np.sign(basis[-1] + 1e-12)


def build_smooth_prior(
    label: ReferenceClass,
    mean_n: SpectralArray,
    grid: WavelengthGrid,
    components: int,
    relative_sigma: float,
    source: str,
    anchors_mxn: SpectralArray | None = None,
) -> ReflectancePrior:
    """Prior with a smooth basis; variances come from anchor spectra if given, else a floor.

    relative_sigma is the per-wavelength standard deviation as a fraction of the mean level.
    For a unit-norm basis vector that equals a coefficient deviation of relative_sigma·‖μ‖.
    """
    basis = smooth_basis(grid, components)
    sigma = relative_sigma * float(np.linalg.norm(mean_n))
    variances = np.full(components, sigma**2)
    if anchors_mxn is not None and anchors_mxn.shape[0] > 1:
        projections = (anchors_mxn - mean_n) @ basis
        variances = np.maximum(projections.var(axis=0, ddof=1), sigma**2)
    return ReflectancePrior(label, mean_n, basis, variances, 1.0, _count(anchors_mxn), source)


def _count(anchors: SpectralArray | None) -> int:
    return 0 if anchors is None else int(anchors.shape[0])


def add_basis_vector(
    prior: ReflectancePrior, vector_n: SpectralArray, relative_sigma: float
) -> ReflectancePrior:
    """Extend a prior with one more (orthogonalized) direction of variation."""
    sigma = relative_sigma * float(np.linalg.norm(prior.mean_n))
    residual = vector_n - prior.basis_nxk @ (prior.basis_nxk.T @ vector_n)
    residual /= np.linalg.norm(residual)
    return ReflectancePrior(
        prior.label,
        prior.mean_n,
        np.column_stack([prior.basis_nxk, residual]),
        np.append(prior.variances_k, sigma**2),
        prior.explained_variance_ratio,
        prior.sample_count,
        prior.source,
    )


def save_prior(directory: Path, prior: ReflectancePrior) -> Path:
    """Write one prior as <label>.npz."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prior.label.value}.npz"
    np.savez(
        path,
        mean_n=prior.mean_n,
        basis_nxk=prior.basis_nxk,
        variances_k=prior.variances_k,
        explained_variance_ratio=prior.explained_variance_ratio,
        sample_count=prior.sample_count,
        source=prior.source,
    )
    return path


def load_prior(path: Path, label: ReferenceClass, grid: WavelengthGrid) -> ReflectancePrior:
    """Read a prior written by save_prior, checking it matches the grid."""
    with np.load(path) as data:
        prior = ReflectancePrior(
            label=label,
            mean_n=data["mean_n"],
            basis_nxk=data["basis_nxk"],
            variances_k=data["variances_k"],
            explained_variance_ratio=float(data["explained_variance_ratio"]),
            sample_count=int(data["sample_count"]),
            source=str(data["source"]),
        )
    if prior.mean_n.size != grid.size:
        raise ValueError(f"Prior {path} has {prior.mean_n.size} samples; grid has {grid.size}")
    return prior


def load_priors(
    directory: Path, grid: WavelengthGrid, fallback: Mapping[ReferenceClass, ReflectancePrior]
) -> dict[ReferenceClass, ReflectancePrior]:
    """Priors from disk where available, otherwise from the fallback set."""
    priors = dict(fallback)
    for label in ReferenceClass:
        path = directory / f"{label.value}.npz"
        if path.exists():
            priors[label] = load_prior(path, label, grid)
            logger.info("Loaded %s prior from %s", label.value, path)
    return priors
