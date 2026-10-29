"""Image-level domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from natural_color_charts.domain.spectral import Matrix3x3, RgbVector

PreviewImage = NDArray[np.uint8]


class InputMode(StrEnum):
    """How the pixel values relate to the sensor."""

    RAW = "raw"
    RASTER = "raster"  # JPEG/HEIC/PNG: linear sRGB stands in for the sensor space


@dataclass(frozen=True)
class ImageMetadata:
    """Capture metadata relevant to color processing."""

    make: str | None = None
    model: str | None = None
    exposure_time_s: float | None = None
    iso: float | None = None
    f_number: float | None = None
    capture_time: str | None = None
    as_shot_white_rgb: RgbVector | None = None
    xyz_to_camera_matrix: Matrix3x3 | None = None

    @property
    def camera_name(self) -> str | None:
        parts = [part for part in (self.make, self.model) if part]
        return " ".join(parts) if parts else None


@dataclass(frozen=True, eq=False)
class LinearImage:
    """Linear camera RGB in [0, 1] with a mask of saturated pixels."""

    rgb: NDArray[np.float32]
    saturation_mask: NDArray[np.bool_]
    metadata: ImageMetadata
    mode: InputMode
    source_path: Path | None = None
    excluded_mask: NDArray[np.bool_] | None = field(default=None)

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        """Pixels usable for statistics: not saturated and not explicitly excluded."""
        valid = ~self.saturation_mask
        if self.excluded_mask is not None:
            valid &= ~self.excluded_mask
        return valid
