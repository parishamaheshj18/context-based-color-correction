"""Ground truth from a hand-annotated 24-patch ColorChecker in a photo."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from natural_color_charts.domain import LinearImage, RgbVector, SpectralArray, normalize_white

PointXy = tuple[float, float]
CHART_ROWS, CHART_COLUMNS = 4, 6
# Neutral 8, 6.5, 5 and 3.5 (patches 20-23); white can clip and black is noisy.
GRAY_PATCH_INDICES = (19, 20, 21, 22)


@dataclass(frozen=True)
class ChartAnnotation:
    """Patch-center corners in normalized image coordinates plus the chart's exclusion box.

    Corners are the centers of patch 1 (dark skin), 6 (bluish green), 19 (white), 24 (black).
    """

    corners: tuple[PointXy, PointXy, PointXy, PointXy]
    mask_box_xyxy: tuple[float, float, float, float]

    def patch_centers(self) -> NDArray[np.float64]:
        """24×2 normalized (x, y) centers by bilinear interpolation, in chart order."""
        top_left, top_right, bottom_left, bottom_right = (np.array(c) for c in self.corners)
        centers = []
        for row in range(CHART_ROWS):
            v = row / (CHART_ROWS - 1)
            for column in range(CHART_COLUMNS):
                u = column / (CHART_COLUMNS - 1)
                top = (1 - u) * top_left + u * top_right
                bottom = (1 - u) * bottom_left + u * bottom_right
                centers.append((1 - v) * top + v * bottom)
        return np.array(centers)

    def sample_radius(self, width: int, height: int) -> int:
        """A quarter of the patch spacing, so samples stay inside each patch."""
        top_left, top_right = np.array(self.corners[0]), np.array(self.corners[1])
        spacing = np.hypot(*((top_right - top_left) * [width, height])) / (CHART_COLUMNS - 1)
        return max(1, int(spacing / 4))

    def mask(self, width: int, height: int) -> NDArray[np.bool_]:
        """Pixels to hide from the pipeline."""
        x0, y0, x1, y1 = self.mask_box_xyxy
        mask = np.zeros((height, width), dtype=bool)
        mask[round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)] = True
        return mask


def sample_chart(image: LinearImage, annotation: ChartAnnotation) -> SpectralArray:
    """Median linear camera RGB of each of the 24 patches."""
    radius = annotation.sample_radius(image.width, image.height)
    samples = []
    for x, y in annotation.patch_centers():
        cx, cy = round(x * image.width), round(y * image.height)
        window = image.rgb[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
        samples.append(np.median(window.reshape(-1, 3), axis=0))
    return np.array(samples, dtype=np.float64)


def chart_white(
    patches_kx3: SpectralArray, indices: Sequence[int] = GRAY_PATCH_INDICES
) -> RgbVector:
    """Ground-truth illuminant: mean chromaticity of the gray patches, green normalized to 1."""
    grays = patches_kx3[list(indices)]
    return normalize_white((grays / grays.sum(axis=1, keepdims=True)).mean(axis=0))
