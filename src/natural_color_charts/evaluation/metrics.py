"""Stage 11 metrics: illuminant angular errors and chart color differences."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from natural_color_charts.colorimetry import D65_WHITE_XYZ, angular_error_deg
from natural_color_charts.correction.adaptation import chromatic_adaptation_matrix
from natural_color_charts.correction.matrix_fitting import mean_delta_e00
from natural_color_charts.domain import Matrix3x3, RgbVector, SpectralArray, WavelengthGrid
from natural_color_charts.spectral.datasets import (
    chart_reflectances,
    cie_1931_cmfs,
    standard_illuminant_spd,
)
from natural_color_charts.spectral.rendering import sensor_responses, white_response


def recovery_angular_error(estimate_rgb: RgbVector, truth_rgb: RgbVector) -> float:
    """Angle between estimated and ground-truth illuminant (degrees)."""
    return angular_error_deg(estimate_rgb, truth_rgb)


def reproduction_angular_error(estimate_rgb: RgbVector, truth_rgb: RgbVector) -> float:
    """Angle between the corrected white and true white (Finlayson & Zakizadeh 2014)."""
    corrected_white = np.asarray(truth_rgb, dtype=float) / np.asarray(estimate_rgb, dtype=float)
    return angular_error_deg(corrected_white, np.ones(3))


def summarize_errors(errors: Sequence[float]) -> dict[str, float]:
    """Mean, median, trimean, best and worst 25%, and maximum, as in color-constancy papers."""
    values = np.sort(np.asarray(errors, dtype=float))
    if values.size == 0:
        return {}
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    quarter = max(1, values.size // 4)
    return {
        "mean": float(values.mean()),
        "median": float(q2),
        "trimean": float((q1 + 2 * q2 + q3) / 4),
        "best25": float(values[:quarter].mean()),
        "worst25": float(values[-quarter:].mean()),
        "max": float(values[-1]),
        "count": float(values.size),
    }


def reference_chart_xyz(grid: WavelengthGrid, chart: str) -> SpectralArray:
    """XYZ of chart patches under D65, adapted to the sRGB D65 white the renderer uses."""
    cmfs, d65 = cie_1931_cmfs(grid), standard_illuminant_spd("D65", grid)
    _, reflectances = chart_reflectances(chart, grid)
    white = white_response(cmfs, d65)
    adapt = chromatic_adaptation_matrix(white / white[1], D65_WHITE_XYZ, "Bradford")
    return sensor_responses(cmfs, d65, reflectances) / white[1] @ adapt.T


def chart_delta_e00(
    camera_to_xyz: Matrix3x3,
    patches_camera_kx3: SpectralArray,
    reference_xyz_kx3: SpectralArray,
    white_patch_index: int,
) -> float:
    """Mean ΔE00 of corrected chart patches, with exposure matched on the white patch."""
    xyz = patches_camera_kx3 @ camera_to_xyz.T
    xyz *= reference_xyz_kx3[white_patch_index, 1] / xyz[white_patch_index, 1]
    return mean_delta_e00(xyz, reference_xyz_kx3, D65_WHITE_XYZ)
