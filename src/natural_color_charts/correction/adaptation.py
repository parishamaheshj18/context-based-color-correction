"""Stage 9: adaptation degree and partial chromatic adaptation."""

from __future__ import annotations

import dataclasses

import numpy as np
from colour.adaptation import CHROMATIC_ADAPTATION_TRANSFORMS

from natural_color_charts.domain import (
    AdaptationMode,
    CorrectionTransform,
    Matrix3x3,
    RgbVector,
    UserIntent,
)


def adaptation_degree(intent: UserIntent) -> float:
    """Map user intent to D in [0, 1]: 1 is fully neutral, 0 keeps the scene illuminant."""
    if intent.adaptation is AdaptationMode.NEUTRAL:
        return 1.0
    if intent.adaptation is AdaptationMode.AS_SHOT:
        return 0.0
    return float(np.clip(intent.strength, 0.0, 1.0))


def cone_matrix(transform: str) -> Matrix3x3:
    """XYZ-to-cone-response matrix of a named chromatic adaptation transform."""
    try:
        return np.asarray(CHROMATIC_ADAPTATION_TRANSFORMS[transform], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"Unknown chromatic adaptation transform: {transform}") from error


def chromatic_adaptation_matrix(
    source_white_xyz: RgbVector, target_white_xyz: RgbVector, transform: str
) -> Matrix3x3:
    """Von Kries-style matrix mapping source_white_xyz exactly onto target_white_xyz."""
    cones = cone_matrix(transform)
    gains = (cones @ target_white_xyz) / (cones @ source_white_xyz)
    return np.asarray(np.linalg.inv(cones) @ np.diag(gains) @ cones)


def interpolate_white(
    source_white_xyz: RgbVector, target_white_xyz: RgbVector, degree: float, transform: str
) -> RgbVector:
    """White between source (degree 0) and target (degree 1), interpolated in cone space."""
    cones = cone_matrix(transform)
    source = cones @ (source_white_xyz / source_white_xyz[1])
    target = cones @ (target_white_xyz / target_white_xyz[1])
    white = np.linalg.inv(cones) @ ((1.0 - degree) * source + degree * target)
    return np.asarray(white / white[1])


def apply_adaptation_degree(
    transform: CorrectionTransform, degree: float, cat_name: str
) -> CorrectionTransform:
    """Re-target a fully corrected transform so whites render at the partially adapted white."""
    if degree >= 1.0:
        return dataclasses.replace(transform, adaptation_degree=1.0)
    partial_white = interpolate_white(
        transform.source_white_xyz, transform.target_white_xyz, degree, cat_name
    )
    retarget = chromatic_adaptation_matrix(transform.target_white_xyz, partial_white, cat_name)
    return dataclasses.replace(
        transform,
        camera_to_xyz=retarget @ transform.camera_to_xyz,
        target_white_xyz=partial_white,
        adaptation_degree=degree,
    )
