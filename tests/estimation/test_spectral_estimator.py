"""Synthetic harness: known illuminant, camera, and prior-sampled reflectances."""

import numpy as np
import pytest

from natural_color_charts.colorimetry import angular_error_deg
from natural_color_charts.domain import (
    CameraSensitivity,
    IlluminantHypothesis,
    PatchMeasurement,
    ReferenceClass,
    SceneContext,
    WavelengthGrid,
)
from natural_color_charts.estimation.base import EstimationInput, EstimationUnavailableError
from natural_color_charts.estimation.spectral import (
    NeutralPatchEstimator,
    SpectralIlluminantEstimator,
    mired_prior_for,
)
from natural_color_charts.evaluation.synthetic import (
    render_synthetic_patches,
    sample_prior_reflectances,
)
from natural_color_charts.spectral.bootstrap_priors import build_bootstrap_priors
from natural_color_charts.spectral.datasets import builtin_camera_sensitivities
from natural_color_charts.spectral.illuminants import BlackbodyFamily, DaylightFamily
from natural_color_charts.spectral.rendering import white_response
from tests.conftest import make_image

GRID = WavelengthGrid(400, 700, 10)
NIKON = CameraSensitivity("Nikon 5100", builtin_camera_sensitivities(GRID)["Nikon 5100"])
FAMILIES = {
    "daylight": DaylightFamily(GRID, 4000, 25000),
    "blackbody": BlackbodyFamily(GRID, 1800, 6500),
}
C = ReferenceClass


@pytest.fixture(scope="module")
def priors(config):
    return build_bootstrap_priors(GRID, config.priors)


@pytest.fixture(scope="module")
def estimator(priors, config):
    return SpectralIlluminantEstimator(FAMILIES, priors, config.estimation)


def _scene(priors, labels, spectrum, seed=0, noise=0.003):
    rng = np.random.default_rng(seed)
    reflectances = np.vstack(
        [
            sample_prior_reflectances(
                priors[label].mean_n, priors[label].basis_nxk, priors[label].variances_k, 1, rng
            )
            for label in labels
        ]
    )
    rendered = render_synthetic_patches(
        NIKON.sensitivity_3xn, spectrum, reflectances, rng, noise, brightness_range=(0.3, 1.0)
    )
    return [
        PatchMeasurement(label, rgb, 5000, 0.8, index)
        for index, (label, rgb) in enumerate(zip(labels, rendered.rgb_kx3, strict=True))
    ]


def _request(patches, context=None):
    return EstimationInput(
        image=make_image(np.ones((2, 2, 3))),
        patches=patches,
        context=context or SceneContext(),
        sensitivity=NIKON,
    )


@pytest.mark.parametrize(
    ("family", "cct", "labels"),
    [
        ("blackbody", 2900.0, [C.CONCRETE, C.PAPER, C.SNOW]),
        ("daylight", 6500.0, [C.CONCRETE, C.CLOUD, C.FOLIAGE]),
        ("daylight", 5000.0, [C.FOLIAGE, C.SKIN, C.CONCRETE, C.PAPER]),
    ],
)
def test_recovers_illuminant_from_prior_sampled_patches(estimator, priors, family, cct, labels):
    truth = FAMILIES[family].spd(cct)
    outcome = estimator.estimate(_request(_scene(priors, labels, truth)))
    estimate = outcome.estimate
    assert angular_error_deg(estimate.white_rgb, white_response(NIKON.sensitivity_3xn, truth)) < 1.5
    assert estimate.family == family
    assert abs(estimate.cct_k - cct) / cct < 0.08
    assert 0 < estimate.confidence <= 1 and estimate.cct_uncertainty_k > 0


def test_leave_one_out_rejects_inconsistent_reference(estimator, priors):
    truth = FAMILIES["blackbody"].spd(3200)
    patches = _scene(priors, [C.CONCRETE, C.PAPER, C.SNOW, C.CLOUD], truth)
    tinted = PatchMeasurement(C.PAPER, patches[1].rgb * np.array([0.6, 1.0, 1.5]), 5000, 0.8, 9)
    outcome = estimator.estimate(_request([*patches, tinted]))
    assert (
        angular_error_deg(outcome.estimate.white_rgb, white_response(NIKON.sensitivity_3xn, truth))
        < 1.5
    )
    rejected = [p for p in outcome.patches if not p.kept]
    assert [p.region_index for p in rejected] == [9]
    assert "inconsistent" in rejected[0].rejection_reason


def test_context_prior_selects_family_and_pulls_estimate(estimator, priors):
    truth = FAMILIES["blackbody"].spd(3000)
    patches = _scene(priors, [C.FOLIAGE], truth, noise=0.0)
    hypothesis = IlluminantHypothesis("blackbody", (2700.0, 3300.0), 0.9)
    outcome = estimator.estimate(
        _request(patches, SceneContext(illuminant_hypotheses=(hypothesis,)))
    )
    assert outcome.estimate.family == "blackbody"
    assert 2500 < outcome.estimate.cct_k < 3600


def test_mired_prior_ignores_zero_confidence_hypotheses():
    context = SceneContext(
        illuminant_hypotheses=(IlluminantHypothesis("daylight", (4000, 6000), 0.0),)
    )
    assert mired_prior_for("daylight", context, 1.0) is None


def test_spectral_estimator_needs_sensitivity_and_patches(estimator, priors):
    with pytest.raises(EstimationUnavailableError):
        estimator.estimate(_request([]))
    patches = _scene(priors, [C.CONCRETE], FAMILIES["daylight"].spd(6000))
    request = EstimationInput(image=make_image(np.ones((2, 2, 3))), patches=patches)
    with pytest.raises(EstimationUnavailableError):
        estimator.estimate(request)


def test_neutral_patch_estimator_uses_only_neutral_classes(priors):
    truth = FAMILIES["blackbody"].spd(3000)
    patches = _scene(priors, [C.CONCRETE, C.SNOW, C.FOLIAGE], truth, noise=0.0)
    estimate = NeutralPatchEstimator(3.0).estimate(_request(patches)).estimate
    assert set(estimate.used_labels) == {"concrete", "snow"}
    assert angular_error_deg(estimate.white_rgb, white_response(NIKON.sensitivity_3xn, truth)) < 3.0
