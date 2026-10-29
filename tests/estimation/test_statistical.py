import numpy as np
import pytest

from natural_color_charts.colorimetry import angular_error_deg
from natural_color_charts.domain import IlluminantEstimate, ImageMetadata
from natural_color_charts.estimation.base import (
    EstimationInput,
    EstimationOutcome,
    EstimationUnavailableError,
    FallbackIlluminantEstimator,
)
from natural_color_charts.estimation.statistical import (
    AsShotEstimator,
    GrayEdgeEstimator,
    GrayWorldEstimator,
    ShadesOfGrayEstimator,
    WhitePatchEstimator,
)
from tests.conftest import make_cast_image, make_image

CAST = np.array([1.6, 1.0, 0.55])


@pytest.mark.parametrize(
    "estimator",
    [
        GrayWorldEstimator(0.2, 10_000),
        ShadesOfGrayEstimator(0.2, 10_000, power=6.0),
        WhitePatchEstimator(0.2, 10_000, percentile=99.5),
        GrayEdgeEstimator(0.2, 10_000, power=6.0, sigma=1.0),
    ],
)
def test_estimators_recover_cast_on_balanced_scene(estimator):
    image = make_cast_image(CAST)
    estimate = estimator.estimate(EstimationInput(image=image)).estimate
    assert angular_error_deg(estimate.white_rgb, CAST) < 3.0
    assert estimate.white_rgb[1] == pytest.approx(1.0)


def test_gray_world_ignores_saturated_pixels():
    image = make_cast_image(CAST)
    image.rgb[:8] = [1.0, 0.1, 0.1]
    image.saturation_mask[:8] = True
    estimate = GrayWorldEstimator(0.2, 10_000).estimate(EstimationInput(image=image)).estimate
    assert angular_error_deg(estimate.white_rgb, CAST) < 3.0


def test_as_shot_requires_metadata():
    image = make_image(np.ones((4, 4, 3)))
    with pytest.raises(EstimationUnavailableError):
        AsShotEstimator(0.2).estimate(EstimationInput(image=image))
    with_white = make_image(np.ones((4, 4, 3)), metadata=ImageMetadata(as_shot_white_rgb=CAST))
    white = AsShotEstimator(0.2).estimate(EstimationInput(image=with_white)).estimate.white_rgb
    np.testing.assert_allclose(white, CAST)


class _Unavailable:
    name = "unavailable"

    def estimate(self, request):
        raise EstimationUnavailableError("missing data")


class _Fixed:
    name = "fixed"

    def estimate(self, request):
        return EstimationOutcome(IlluminantEstimate(np.ones(3), self.name, 0.5))


def test_fallback_chain_uses_first_available_and_marks_fallback():
    chain = FallbackIlluminantEstimator([_Unavailable(), _Fixed()])
    estimate = chain.estimate(EstimationInput(image=make_image(np.ones((2, 2, 3))))).estimate
    assert estimate.method == "fixed" and estimate.is_fallback
    assert any("missing data" in note for note in estimate.notes)


def test_fallback_chain_raises_when_all_fail():
    with pytest.raises(EstimationUnavailableError):
        FallbackIlluminantEstimator([_Unavailable()]).estimate(
            EstimationInput(image=make_image(np.ones((2, 2, 3))))
        )
