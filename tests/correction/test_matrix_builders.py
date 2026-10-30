import numpy as np
import pytest

from natural_color_charts.colorimetry import D65_WHITE_XYZ, XYZ_TO_LINEAR_SRGB
from natural_color_charts.correction.adaptation import (
    adaptation_degree,
    apply_adaptation_degree,
    interpolate_white,
)
from natural_color_charts.correction.base import (
    CorrectionUnavailableError,
    FallbackCorrectionBuilder,
)
from natural_color_charts.correction.matrix_builders import (
    BradfordCorrectionBuilder,
    CameraMatrixCorrectionBuilder,
)
from natural_color_charts.correction.renderer import SrgbRenderer
from natural_color_charts.domain import (
    AdaptationMode,
    IlluminantEstimate,
    ImageMetadata,
    InputMode,
    UserIntent,
)
from tests.conftest import make_image

WHITE = np.array([1.8, 1.0, 0.6])
XYZ_TO_CAMERA = np.array([[0.9, 0.2, -0.1], [-0.3, 1.2, 0.1], [0.0, -0.2, 1.1]])


def _estimate() -> IlluminantEstimate:
    return IlluminantEstimate(white_rgb=WHITE, method="test", confidence=1.0)


def test_camera_matrix_builder_maps_estimated_white_to_d65():
    image = make_image(
        np.ones((2, 2, 3)), InputMode.RAW, ImageMetadata(xyz_to_camera_matrix=XYZ_TO_CAMERA)
    )
    transform = CameraMatrixCorrectionBuilder().build(_estimate(), image, None)
    np.testing.assert_allclose(transform.camera_to_xyz @ WHITE, D65_WHITE_XYZ, atol=1e-9)


def test_camera_matrix_builder_needs_matrix():
    with pytest.raises(CorrectionUnavailableError):
        CameraMatrixCorrectionBuilder().build(_estimate(), make_image(np.ones((2, 2, 3))), None)


@pytest.mark.parametrize("cat", ["Bradford", "CAT16"])
def test_bradford_builder_neutralizes_white_and_keeps_luminance(cat):
    transform = BradfordCorrectionBuilder(cat).build(
        _estimate(), make_image(np.ones((1, 1, 3))), None
    )
    srgb = XYZ_TO_LINEAR_SRGB @ transform.camera_to_xyz @ WHITE
    np.testing.assert_allclose(srgb / srgb[1], np.ones(3), atol=1e-6)


def test_fallback_correction_builder_skips_unavailable():
    chain = FallbackCorrectionBuilder(
        [CameraMatrixCorrectionBuilder(), BradfordCorrectionBuilder()]
    )
    assert chain.build(_estimate(), make_image(np.ones((1, 1, 3))), None).method == "bradford"


def test_adaptation_degree_from_intent():
    assert adaptation_degree(UserIntent(AdaptationMode.NEUTRAL, 0.3)) == 1.0
    assert adaptation_degree(UserIntent(AdaptationMode.AS_SHOT, 0.9)) == 0.0
    assert adaptation_degree(UserIntent(AdaptationMode.PRESERVE_AMBIANCE, 0.6)) == 0.6


def test_partial_adaptation_moves_white_between_source_and_target():
    full = BradfordCorrectionBuilder().build(_estimate(), make_image(np.ones((1, 1, 3))), None)
    none = apply_adaptation_degree(full, 0.0, "Bradford")
    rendered_white = none.camera_to_xyz @ WHITE
    np.testing.assert_allclose(rendered_white / rendered_white[1], full.source_white_xyz, atol=1e-9)
    half = interpolate_white(full.source_white_xyz, D65_WHITE_XYZ, 0.5, "Bradford")
    assert full.source_white_xyz[2] < half[2] < D65_WHITE_XYZ[2]


def test_renderer_outputs_neutral_gray_for_gray_patch():
    image = make_image(np.tile(WHITE * 0.2, (4, 4, 1)), InputMode.RAW)
    transform = BradfordCorrectionBuilder().build(_estimate(), image, None)
    rendered = SrgbRenderer(
        98.0, 0.5, normalize_raster=False, highlight_knee=0.9, saturation_threshold=0.98
    ).render(image, transform)
    pixel = rendered.linear_srgb[0, 0]
    np.testing.assert_allclose(pixel, [0.5, 0.5, 0.5], atol=1e-5)
    assert rendered.exposure_gain > 1
