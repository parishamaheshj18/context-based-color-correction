"""Pure matrix-fitting math for Stage 8."""

from __future__ import annotations

import colour
import numpy as np
from scipy.optimize import minimize

from natural_color_charts.colorimetry import xyz_to_xy
from natural_color_charts.domain import Matrix3x3, RgbVector, SpectralArray


def fit_white_preserving_matrix(
    camera_kx3: SpectralArray, xyz_kx3: SpectralArray, camera_white: RgbVector, xyz_white: RgbVector
) -> Matrix3x3:
    """Least-squares M with M·camera ≈ xyz, constrained so M·camera_white == xyz_white exactly.

    Closed-form equality-constrained least squares, solved independently for each output row.
    """
    camera = np.asarray(camera_kx3, dtype=np.float64).T
    target = np.asarray(xyz_kx3, dtype=np.float64).T
    gram_inv = np.linalg.inv(camera @ camera.T)
    unconstrained = target @ camera.T @ gram_inv
    direction = gram_inv @ camera_white
    violation = xyz_white - unconstrained @ camera_white
    return unconstrained + np.outer(violation, direction) / (camera_white @ direction)


def mean_delta_e00(
    xyz_a_kx3: SpectralArray, xyz_b_kx3: SpectralArray, white_xyz: RgbVector
) -> float:
    """Mean CIEDE2000 difference between two sets of XYZ colors under one white."""
    white_xy = xyz_to_xy(white_xyz)
    lab_a = colour.XYZ_to_Lab(xyz_a_kx3, white_xy)
    lab_b = colour.XYZ_to_Lab(xyz_b_kx3, white_xy)
    return float(np.mean(colour.delta_E(lab_a, lab_b, method="CIE 2000")))


def refine_matrix_for_delta_e(
    matrix: Matrix3x3,
    camera_kx3: SpectralArray,
    xyz_kx3: SpectralArray,
    camera_white: RgbVector,
    xyz_white: RgbVector,
) -> Matrix3x3:
    """Minimize mean ΔE00 while keeping the white mapping exact.

    Updates are restricted to rows orthogonal to camera_white, so M·camera_white is unchanged.
    """
    _, _, vh = np.linalg.svd(camera_white[None, :])
    null_basis_2x3 = vh[1:]

    def compose(params: SpectralArray) -> Matrix3x3:
        return matrix + params.reshape(3, 2) @ null_basis_2x3

    def objective(params: SpectralArray) -> float:
        return mean_delta_e00(camera_kx3 @ compose(params).T, xyz_kx3, xyz_white)

    result = minimize(objective, np.zeros(6), method="Nelder-Mead", options={"maxiter": 4000})
    return compose(result.x)
