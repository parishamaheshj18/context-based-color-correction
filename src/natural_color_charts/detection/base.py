"""Stage 4 interfaces: detection, segmentation, and mask refinement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from natural_color_charts.domain import Detection, PreviewImage, ReferenceClass, SceneContext

__all__ = ["PreviewImage"]

PreviewMask = NDArray[np.bool_]


@dataclass(frozen=True, eq=False)
class DetectionRequest:
    """A display preview, the labels to look for, and where the image came from."""

    preview: PreviewImage
    labels: Sequence[ReferenceClass]
    source_path: Path | None = None
    context: SceneContext | None = None


class ReferenceDetector(Protocol):
    """Finds labeled boxes; boxes use normalized [0, 1] xyxy coordinates."""

    def detect(self, request: DetectionRequest) -> list[Detection]: ...


class Segmenter(Protocol):
    """Turns detections into preview-resolution masks, one per detection."""

    def segment(
        self, preview: PreviewImage, detections: Sequence[Detection]
    ) -> list[PreviewMask]: ...


class MaskRefiner(Protocol):
    """Removes pixels that do not belong to the reference material."""

    def refine(
        self, preview: PreviewImage, label: ReferenceClass, mask: PreviewMask
    ) -> PreviewMask: ...


def box_to_pixels(
    box_xyxy: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """Convert a normalized box to clamped integer pixel bounds."""
    x0, y0, x1, y1 = box_xyxy
    left, right = sorted((round(x0 * width), round(x1 * width)))
    top, bottom = sorted((round(y0 * height), round(y1 * height)))
    return max(0, left), max(0, top), min(width, right), min(height, bottom)
