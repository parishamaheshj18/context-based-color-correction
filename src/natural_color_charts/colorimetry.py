"""Pure color-science helpers shared across stages."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from natural_color_charts.domain.spectral import Matrix3x3, RgbVector

# IEC 61966-2-1 primaries with a D65 white.
XYZ_TO_LINEAR_SRGB: Matrix3x3 = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)
LINEAR_SRGB_TO_XYZ: Matrix3x3 = np.linalg.inv(XYZ_TO_LINEAR_SRGB)
D65_WHITE_XYZ: RgbVector = LINEAR_SRGB_TO_XYZ @ np.ones(3)


def decode_srgb(encoded: NDArray[np.floating]) -> NDArray[np.float32]:
    """Invert the sRGB transfer function."""
    encoded = np.asarray(encoded, dtype=np.float32)
    linear = np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
    return linear.astype(np.float32)


def encode_srgb(linear: NDArray[np.floating]) -> NDArray[np.float32]:
    """Apply the sRGB transfer function to values clipped to [0, 1]."""
    linear = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    return encoded.astype(np.float32)


def angular_error_deg(a: RgbVector, b: RgbVector) -> float:
    """Angle between two RGB vectors, the standard color-constancy error metric."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def apply_matrix(image_rgb: NDArray[np.floating], matrix: Matrix3x3) -> NDArray[np.float32]:
    """Apply a 3x3 matrix to every pixel of an H×W×3 image."""
    flat = np.asarray(image_rgb, dtype=np.float32).reshape(-1, 3)
    return (flat @ np.asarray(matrix, dtype=np.float32).T).reshape(np.shape(image_rgb))


def luminance_from_linear_srgb(image_rgb: NDArray[np.floating]) -> NDArray[np.float32]:
    """Relative luminance Y of linear sRGB pixels."""
    weights = LINEAR_SRGB_TO_XYZ[1].astype(np.float32)
    return np.asarray(image_rgb, dtype=np.float32) @ weights


def xyz_to_lab(xyz: NDArray[np.floating], white_xyz: RgbVector) -> NDArray[np.float64]:
    """CIE 1976 L*a*b* of XYZ (last axis) relative to white_xyz."""
    ratio = np.asarray(xyz, dtype=np.float64) / np.asarray(white_xyz, dtype=np.float64)
    delta = 6 / 29
    f = np.where(ratio > delta**3, np.cbrt(ratio), ratio / (3 * delta**2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def xyz_to_xy(xyz: RgbVector) -> NDArray[np.float64]:
    """CIE 1931 chromaticity coordinates of an XYZ triplet."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz[:2] / xyz.sum()
