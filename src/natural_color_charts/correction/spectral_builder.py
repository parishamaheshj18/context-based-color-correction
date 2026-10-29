"""Stage 8: spectrally derived correction matrix from a virtual color chart."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from natural_color_charts.colorimetry import D65_WHITE_XYZ
from natural_color_charts.correction.adaptation import chromatic_adaptation_matrix
from natural_color_charts.correction.base import CorrectionUnavailableError
from natural_color_charts.correction.matrix_builders import normalize_luminance
from natural_color_charts.correction.matrix_fitting import (
    fit_white_preserving_matrix,
    refine_matrix_for_delta_e,
)
from natural_color_charts.domain import (
    CameraSensitivity,
    CorrectionTransform,
    IlluminantEstimate,
    InputMode,
    LinearImage,
    SpectralArray,
    normalize_white,
)
from natural_color_charts.spectral.illuminants import IlluminantFamily
from natural_color_charts.spectral.locus import match_locus
from natural_color_charts.spectral.rendering import sensor_responses, white_response


class SpectralCorrectionMatrixBuilder:
    """Fits camera RGB to XYZ on chart reflectances rendered under the estimated illuminant."""

    name = "spectral"

    def __init__(
        self,
        training_reflectances_kxn: SpectralArray,
        cmfs_3xn: SpectralArray,
        target_spd: SpectralArray,
        families: Sequence[IlluminantFamily],
        refine_delta_e: bool,
        allow_raster: bool,
    ) -> None:
        self._reflectances = training_reflectances_kxn
        self._cmfs = cmfs_3xn
        self._families = list(families)
        self._refine = refine_delta_e
        self._allow_raster = allow_raster
        white_xyz = white_response(cmfs_3xn, target_spd)
        target_xyz = sensor_responses(cmfs_3xn, target_spd, training_reflectances_kxn)
        # The renderer assumes the sRGB D65 white; adapting to it keeps neutrals exactly neutral.
        to_display_white = chromatic_adaptation_matrix(
            white_xyz / white_xyz[1], D65_WHITE_XYZ, "Bradford"
        )
        self._target_white = D65_WHITE_XYZ
        self._target_xyz = target_xyz / white_xyz[1] @ to_display_white.T

    def build(
        self,
        estimate: IlluminantEstimate,
        image: LinearImage,
        sensitivity: CameraSensitivity | None,
    ) -> CorrectionTransform:
        if sensitivity is None:
            raise CorrectionUnavailableError("no camera sensitivity data")
        if image.mode is InputMode.RASTER and not self._allow_raster:
            raise CorrectionUnavailableError("spectral matrix disabled for raster input")
        spectrum, notes = self._illuminant_spectrum(estimate, sensitivity)
        camera = sensor_responses(sensitivity.sensitivity_3xn, spectrum, self._reflectances)
        # Scale so the illuminant's own camera white has green 1, matching the estimate's scale.
        camera /= white_response(sensitivity.sensitivity_3xn, spectrum)[1]
        white = normalize_white(estimate.white_rgb)
        matrix = fit_white_preserving_matrix(camera, self._target_xyz, white, self._target_white)
        if self._refine:
            matrix = refine_matrix_for_delta_e(
                matrix, camera, self._target_xyz, white, self._target_white
            )
        return CorrectionTransform(
            camera_to_xyz=matrix,
            source_white_xyz=normalize_luminance(white_response(self._cmfs, spectrum)),
            target_white_xyz=self._target_white,
            method=self.name,
            notes=notes,
        )

    def _illuminant_spectrum(
        self, estimate: IlluminantEstimate, sensitivity: CameraSensitivity
    ) -> tuple[SpectralArray, tuple[str, ...]]:
        if estimate.spectrum is not None:
            return np.asarray(estimate.spectrum), ()
        match = match_locus(estimate.white_rgb, sensitivity.sensitivity_3xn, self._families)
        note = (
            f"spectrum inferred from white point: {match.family} {match.cct_k:.0f} K "
            f"(locus distance {match.angular_error_deg:.2f}°)"
        )
        return match.spectrum, (note,)
