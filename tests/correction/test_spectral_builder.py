import numpy as np
import pytest

from natural_color_charts.colorimetry import D65_WHITE_XYZ
from natural_color_charts.correction.base import CorrectionUnavailableError
from natural_color_charts.correction.matrix_fitting import (
    fit_white_preserving_matrix,
    mean_delta_e00,
)
from natural_color_charts.correction.spectral_builder import SpectralCorrectionMatrixBuilder
from natural_color_charts.domain import (
    CameraSensitivity,
    IlluminantEstimate,
    InputMode,
    WavelengthGrid,
)
from natural_color_charts.evaluation.synthetic import render_synthetic_patches
from natural_color_charts.spectral.datasets import (
    builtin_camera_sensitivities,
    chart_reflectances,
    cie_1931_cmfs,
    standard_illuminant_spd,
)
from natural_color_charts.spectral.illuminants import BlackbodyFamily, DaylightFamily
from natural_color_charts.spectral.rendering import sensor_responses, white_response
from tests.conftest import make_image

GRID = WavelengthGrid(400, 700, 10)
CMFS = cie_1931_cmfs(GRID)
D65 = standard_illuminant_spd("D65", GRID)
_, CHART = chart_reflectances("BabelColor Average", GRID)
NIKON = builtin_camera_sensitivities(GRID)["Nikon 5100"]
FAMILIES = [DaylightFamily(GRID, 4000, 25000), BlackbodyFamily(GRID, 1800, 6500)]


def _builder(refine: bool = False) -> SpectralCorrectionMatrixBuilder:
    return SpectralCorrectionMatrixBuilder(CHART, CMFS, D65, FAMILIES, refine, allow_raster=False)


def _reference_xyz() -> np.ndarray:
    return sensor_responses(CMFS, D65, CHART) / white_response(CMFS, D65)[1]


def test_white_preserving_fit_satisfies_constraint():
    rng = np.random.default_rng(1)
    camera, target = rng.uniform(size=(24, 3)), rng.uniform(size=(24, 3))
    white, white_xyz = np.array([0.6, 1.0, 0.8]), np.array([0.95, 1.0, 1.09])
    matrix = fit_white_preserving_matrix(camera, target, white, white_xyz)
    np.testing.assert_allclose(matrix @ white, white_xyz, atol=1e-12)


@pytest.mark.parametrize("illuminant_cct", [2800.0, 4200.0])
def test_spectral_matrix_corrects_chart_under_tungsten_like_light(illuminant_cct):
    spectrum = BlackbodyFamily(GRID, 1800, 6500).spd(illuminant_cct)
    patches = render_synthetic_patches(NIKON, spectrum, CHART, np.random.default_rng(0))
    camera_white = white_response(NIKON, spectrum)
    estimate = IlluminantEstimate(camera_white / camera_white[1], "truth", 1.0, spectrum=spectrum)
    image = make_image(np.ones((2, 2, 3)), InputMode.RAW)
    transform = _builder().build(estimate, image, CameraSensitivity("Nikon 5100", NIKON))
    np.testing.assert_allclose(
        transform.camera_to_xyz @ estimate.white_rgb, D65_WHITE_XYZ, atol=1e-3
    )
    corrected = patches.rgb_kx3 / camera_white[1] @ transform.camera_to_xyz.T
    assert mean_delta_e00(corrected, _reference_xyz(), transform.target_white_xyz) < 3.0


def test_delta_e_refinement_does_not_increase_error():
    spectrum = BlackbodyFamily(GRID, 1800, 6500).spd(3000)
    camera = sensor_responses(NIKON, spectrum, CHART) / white_response(NIKON, spectrum)[1]
    white = white_response(NIKON, spectrum) / white_response(NIKON, spectrum)[1]
    estimate = IlluminantEstimate(white, "truth", 1.0, spectrum=spectrum)
    image = make_image(np.ones((2, 2, 3)), InputMode.RAW)
    sensitivity = CameraSensitivity("Nikon 5100", NIKON)
    errors = []
    for refine in (False, True):
        matrix = _builder(refine).build(estimate, image, sensitivity).camera_to_xyz
        errors.append(mean_delta_e00(camera @ matrix.T, _reference_xyz(), D65_WHITE_XYZ))
    assert errors[1] <= errors[0] + 1e-9


def test_spectrum_inferred_from_white_when_missing():
    spectrum = DaylightFamily(GRID, 4000, 25000).spd(5500)
    white = white_response(NIKON, spectrum)
    estimate = IlluminantEstimate(white / white[1], "gray_world", 0.2)
    image = make_image(np.ones((2, 2, 3)), InputMode.RAW)
    transform = _builder().build(estimate, image, CameraSensitivity("Nikon 5100", NIKON))
    assert any("daylight" in note for note in transform.notes)


def test_spectral_builder_requires_sensitivity_and_skips_raster():
    estimate = IlluminantEstimate(np.ones(3), "x", 1.0)
    with pytest.raises(CorrectionUnavailableError):
        _builder().build(estimate, make_image(np.ones((2, 2, 3)), InputMode.RAW), None)
    with pytest.raises(CorrectionUnavailableError):
        _builder().build(estimate, make_image(np.ones((2, 2, 3))), CameraSensitivity("sRGB", CMFS))
