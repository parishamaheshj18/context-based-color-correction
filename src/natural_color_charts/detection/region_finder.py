"""Stage 4: find reference regions for the usable candidates in the scene context."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from natural_color_charts.detection.base import (
    DetectionRequest,
    MaskRefiner,
    PreviewImage,
    PreviewMask,
    ReferenceDetector,
    Segmenter,
)
from natural_color_charts.domain import LinearImage, ReferenceRegion, SceneContext

logger = logging.getLogger(__name__)


def upscale_mask(mask: PreviewMask, width: int, height: int) -> NDArray[np.bool_]:
    """Nearest-neighbor resize of a preview mask to full resolution."""
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return np.asarray(resized.astype(bool))


class ReferenceRegionFinder:
    """Detect, segment, and refine on the preview, then map masks to the full image."""

    def __init__(
        self,
        detector: ReferenceDetector,
        segmenter: Segmenter,
        refiners: Sequence[MaskRefiner],
    ) -> None:
        self._detector = detector
        self._segmenter = segmenter
        self._refiners = list(refiners)

    def find(
        self, image: LinearImage, preview: PreviewImage, context: SceneContext
    ) -> list[ReferenceRegion]:
        """Regions for every usable reference candidate in the context."""
        usable = [c for c in context.reference_candidates if c.usable]
        if not usable:
            return []
        request = DetectionRequest(preview, [c.label for c in usable], image.source_path)
        detections = self._detector.detect(request)
        logger.info("Detected %d reference boxes", len(detections))
        masks = self._segmenter.segment(preview, detections)
        regions = []
        for detection, mask in zip(detections, masks, strict=True):
            for refiner in self._refiners:
                mask = refiner.refine(preview, detection.label, mask)
            candidate = context.candidate_for(detection.label)
            regions.append(
                ReferenceRegion(
                    label=detection.label,
                    mask=upscale_mask(mask, image.width, image.height),
                    detector_score=detection.score,
                    box_xyxy=detection.box_xyxy,
                    candidate_confidence=candidate.confidence if candidate else 0.5,
                    description=candidate.description if candidate else "",
                )
            )
        return regions
