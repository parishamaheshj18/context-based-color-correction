from pathlib import Path

import numpy as np

from natural_color_charts.domain import ReferenceClass, WavelengthGrid
from natural_color_charts.spectral.bootstrap_priors import build_bootstrap_priors
from natural_color_charts.spectral.priors import (
    build_pca_prior,
    load_priors,
    save_prior,
    smooth_basis,
)

GRID = WavelengthGrid(400, 700, 10)


def test_pca_prior_recovers_generating_subspace():
    rng = np.random.default_rng(0)
    basis = smooth_basis(GRID, 2)
    spectra = 0.4 + rng.normal(size=(200, 2)) * [0.3, 0.1] @ basis.T
    prior = build_pca_prior(ReferenceClass.FOLIAGE, spectra, 2, "test")
    projection = prior.basis_nxk @ prior.basis_nxk.T
    np.testing.assert_allclose(projection @ basis, basis, atol=1e-8)
    assert prior.explained_variance_ratio > 0.999
    np.testing.assert_allclose(np.sqrt(prior.variances_k), [0.3, 0.1], rtol=0.15)


def test_smooth_basis_is_orthonormal():
    basis = smooth_basis(GRID, 4)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)


def test_saved_priors_override_bootstrap(tmp_path: Path, config):
    bootstrap = build_bootstrap_priors(GRID, config.priors)
    assert set(bootstrap) == set(ReferenceClass)
    custom = build_pca_prior(
        ReferenceClass.SNOW, np.random.default_rng(1).uniform(0.8, 0.9, (10, GRID.size)), 2, "mine"
    )
    save_prior(tmp_path, custom)
    loaded = load_priors(tmp_path, GRID, bootstrap)
    assert loaded[ReferenceClass.SNOW].source == "mine"
    assert loaded[ReferenceClass.SKIN].source == bootstrap[ReferenceClass.SKIN].source
