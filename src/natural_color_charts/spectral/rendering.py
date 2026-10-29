"""Image formation model: spectra to camera RGB and to XYZ."""

from __future__ import annotations

import numpy as np

from natural_color_charts.domain import SpectralArray


def sensor_responses(
    sensitivity_3xn: SpectralArray, illuminant: SpectralArray, reflectances_kxn: SpectralArray
) -> SpectralArray:
    """Responses ρ_k = S · diag(L) · R_k for K reflectances, as a K×3 array."""
    return np.atleast_2d(reflectances_kxn) * illuminant @ sensitivity_3xn.T


def white_response(sensitivity_3xn: SpectralArray, illuminant: SpectralArray) -> SpectralArray:
    """Response to a perfect white reflector, i.e. the illuminant as seen by the sensor."""
    return sensitivity_3xn @ illuminant


def chromaticity(rgb: SpectralArray) -> SpectralArray:
    """Normalize RGB rows to sum to one, removing per-patch brightness."""
    rgb = np.atleast_2d(rgb)
    return rgb / rgb.sum(axis=-1, keepdims=True)
