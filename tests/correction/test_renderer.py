"""Rendering: highlights roll off with their hue kept; only RAW sensor clipping turns white."""

import dataclasses

import numpy as np

from natural_color_charts.colorimetry import D65_WHITE_XYZ, LINEAR_SRGB_TO_XYZ, decode_srgb
from natural_color_charts.correction.renderer import SrgbRenderer, compress_highlights
from natural_color_charts.domain import CorrectionTransform, InputMode
from tests.conftest import make_image

IDENTITY = CorrectionTransform(LINEAR_SRGB_TO_XYZ, D65_WHITE_XYZ, D65_WHITE_XYZ, "as_shot")
RENDERER = SrgbRenderer(
    98.0, 0.85, normalize_raster=False, highlight_knee=0.9, saturation_threshold=0.98
)


def test_pixels_below_the_knee_are_unchanged():
    pixels = np.array([[0.2, 0.5, 0.89], [0.0, 0.1, 0.3]], np.float32)
    np.testing.assert_array_equal(compress_highlights(pixels, 0.9), pixels)


def test_bright_pixels_roll_off_below_one_and_keep_their_hue():
    pixels = np.array([[0.5, 0.8, 1.0], [1.0, 1.4, 2.0], [0.3, 0.6, 0.95]], np.float32)
    rolled = compress_highlights(pixels, 0.9)
    assert np.all(rolled.max(axis=1) < 1.0)
    np.testing.assert_allclose(
        rolled / rolled.max(axis=1, keepdims=True),
        pixels / pixels.max(axis=1, keepdims=True),
        rtol=1e-6,
    )
    assert rolled[1].max() > rolled[0].max() > rolled[2].max()  # brighter stays brighter


def _sky(mode):
    # A vivid sky in a phone photo: blue at the file's maximum, red and green well below it.
    image = make_image(np.tile(decode_srgb(np.array([198, 231, 255]) / 255), (4, 4, 1)), mode)
    return dataclasses.replace(image, saturation_mask=np.ones((4, 4), bool))


def test_a_vivid_sky_in_a_phone_photo_stays_blue():
    pixel = RENDERER.render(_sky(InputMode.RASTER), IDENTITY).encoded_srgb[0, 0] * 255
    assert pixel[2] - pixel[0] > 40  # still clearly blue, not painted white


def test_a_blown_white_sky_in_a_phone_photo_stays_white_under_a_warm_correction():
    image = make_image(np.ones((4, 4, 3), np.float32), InputMode.RASTER)
    image = dataclasses.replace(image, saturation_mask=np.ones((4, 4), bool))
    warm = dataclasses.replace(
        IDENTITY, camera_to_xyz=LINEAR_SRGB_TO_XYZ @ np.diag([1.2, 1.0, 0.8])
    )
    pixel = RENDERER.render(image, warm).linear_srgb[0, 0]
    assert np.ptp(pixel) < 1e-6  # the camera clipped it to white, so it is not tinted cream


def test_clipped_raw_pixels_render_neutral():
    pixel = RENDERER.render(_sky(InputMode.RAW), IDENTITY, gain=1.0).linear_srgb[0, 0]
    assert np.ptp(pixel) < 1e-6
