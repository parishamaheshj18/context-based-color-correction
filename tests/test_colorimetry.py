import numpy as np

from natural_color_charts.colorimetry import (
    D65_WHITE_XYZ,
    XYZ_TO_LINEAR_SRGB,
    angular_error_deg,
    decode_srgb,
    encode_srgb,
)


def test_srgb_round_trip():
    values = np.linspace(0, 1, 101, dtype=np.float32)
    np.testing.assert_allclose(decode_srgb(encode_srgb(values)), values, atol=1e-5)


def test_d65_maps_to_srgb_white():
    np.testing.assert_allclose(XYZ_TO_LINEAR_SRGB @ D65_WHITE_XYZ, np.ones(3), atol=1e-6)


def test_angular_error_is_scale_invariant():
    assert angular_error_deg(np.array([1, 2, 3]), np.array([2, 4, 6])) < 1e-6
    assert abs(angular_error_deg(np.array([1, 0, 0]), np.array([0, 1, 0])) - 90) < 1e-9
