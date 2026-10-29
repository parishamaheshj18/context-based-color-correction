"""Correction builders that do not need spectral sensitivity data (Stage 8 fallbacks)."""

from __future__ import annotations

import numpy as np

from natural_color_charts.colorimetry import D65_WHITE_XYZ, LINEAR_SRGB_TO_XYZ
from natural_color_charts.correction.adaptation import chromatic_adaptation_matrix
from natural_color_charts.correction.base import CorrectionUnavailableError
from natural_color_charts.domain import (
    CameraSensitivity,
    CorrectionTransform,
    IlluminantEstimate,
    InputMode,
    LinearImage,
    RgbVector,
    normalize_white,
)


def normalize_luminance(xyz: RgbVector) -> RgbVector:
    """Scale an XYZ white so Y is 1."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz / xyz[1]


class CameraMatrixCorrectionBuilder:
    """Uses the RAW file's XYZ-to-camera matrix after white-balancing to the estimate."""

    name = "camera_matrix"

    def build(
        self,
        estimate: IlluminantEstimate,
        image: LinearImage,
        sensitivity: CameraSensitivity | None,
    ) -> CorrectionTransform:
        xyz_to_camera = image.metadata.xyz_to_camera_matrix
        if image.mode is not InputMode.RAW or xyz_to_camera is None:
            raise CorrectionUnavailableError("no camera color matrix in metadata")
        camera_to_xyz = np.linalg.inv(xyz_to_camera)
        white = normalize_white(estimate.white_rgb)
        # The matrix is calibrated for D65: map the estimated white to the D65 camera white first.
        d65_camera_white = xyz_to_camera @ D65_WHITE_XYZ
        white_balance = np.diag(d65_camera_white / white)
        return CorrectionTransform(
            camera_to_xyz=camera_to_xyz @ white_balance,
            source_white_xyz=normalize_luminance(camera_to_xyz @ white),
            target_white_xyz=D65_WHITE_XYZ,
            method=self.name,
            notes=("single D65 camera matrix; dual-illuminant interpolation not applied",),
        )


class BradfordCorrectionBuilder:
    """Treats the sensor space as linear sRGB and adapts the estimated white to D65."""

    name = "bradford"

    def __init__(self, transform: str = "Bradford") -> None:
        self._transform = transform

    def build(
        self,
        estimate: IlluminantEstimate,
        image: LinearImage,
        sensitivity: CameraSensitivity | None,
    ) -> CorrectionTransform:
        source_white = normalize_luminance(LINEAR_SRGB_TO_XYZ @ normalize_white(estimate.white_rgb))
        adaptation = chromatic_adaptation_matrix(source_white, D65_WHITE_XYZ, self._transform)
        notes = () if image.mode is InputMode.RASTER else ("RAW treated as linear sRGB",)
        return CorrectionTransform(
            camera_to_xyz=adaptation @ LINEAR_SRGB_TO_XYZ,
            source_white_xyz=source_white,
            target_white_xyz=D65_WHITE_XYZ,
            method=self.name,
            notes=notes,
        )
