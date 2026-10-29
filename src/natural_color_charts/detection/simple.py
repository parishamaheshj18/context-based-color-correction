"""Detectors and segmenters that need no model weights."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

import cv2
import numpy as np

from natural_color_charts.detection.base import (
    DetectionRequest,
    PreviewImage,
    PreviewMask,
    box_to_pixels,
)
from natural_color_charts.domain import Detection, ReferenceClass

logger = logging.getLogger(__name__)


class NullDetector:
    """Finds nothing, so estimation falls back to image statistics."""

    def detect(self, request: DetectionRequest) -> list[Detection]:
        return []


class ManualRegionDetector:
    """Reads boxes from a sidecar JSON next to the image, e.g. photo.regions.json.

    Format: [{"label": "paper", "box": [x0, y0, x1, y1]}] with coordinates normalized to [0, 1].
    """

    def __init__(self, sidecar_suffix: str) -> None:
        self._suffix = sidecar_suffix

    def detect(self, request: DetectionRequest) -> list[Detection]:
        if request.source_path is None:
            return []
        sidecar = request.source_path.with_name(request.source_path.stem + self._suffix)
        if not sidecar.exists():
            logger.info("No region sidecar at %s", sidecar)
            return []
        entries = json.loads(sidecar.read_text())
        wanted = set(request.labels)
        detections = [
            Detection(
                label=ReferenceClass(entry["label"]),
                box_xyxy=(
                    float(entry["box"][0]),
                    float(entry["box"][1]),
                    float(entry["box"][2]),
                    float(entry["box"][3]),
                ),
                score=float(entry.get("score", 1.0)),
            )
            for entry in entries
        ]
        return [d for d in detections if d.label in wanted]


class ContextBoxDetector:
    """Uses the boxes the scene analysis (VLM) attached to each usable candidate."""

    def detect(self, request: DetectionRequest) -> list[Detection]:
        if request.context is None:
            return []
        wanted = set(request.labels)
        return [
            Detection(
                label=candidate.label,
                box_xyxy=box,
                score=candidate.confidence,
                candidate_confidence=candidate.confidence,
                description=candidate.description,
            )
            for candidate in request.context.reference_candidates
            if candidate.usable and candidate.label in wanted
            for box in candidate.boxes
        ]


class ChromaCoherenceRefiner:
    """Keeps mask pixels whose rg chromaticity is near the mask's median.

    Boxes include some background; the material usually dominates, so its chromaticity is
    the median and the outliers are other objects.
    """

    def __init__(self, max_distance: float) -> None:
        self._max_distance = max_distance

    def refine(
        self, preview: PreviewImage, label: ReferenceClass, mask: PreviewMask
    ) -> PreviewMask:
        if self._max_distance <= 0 or not mask.any():
            return mask
        rgb = preview.astype(np.float32) + 1.0
        rg = rgb[..., :2] / rgb.sum(axis=2, keepdims=True)
        distance = np.linalg.norm(rg - np.median(rg[mask], axis=0), axis=2)
        return np.asarray(mask & (distance <= self._max_distance))


class BoxSegmenter:
    """Uses the inner part of each box as its mask; patch statistics reject the outliers."""

    def __init__(self, inset_fraction: float) -> None:
        self._inset = inset_fraction

    def segment(self, preview: PreviewImage, detections: Sequence[Detection]) -> list[PreviewMask]:
        height, width = preview.shape[:2]
        masks = []
        for detection in detections:
            left, top, right, bottom = box_to_pixels(detection.box_xyxy, width, height)
            inset_x = int((right - left) * self._inset)
            inset_y = int((bottom - top) * self._inset)
            mask = np.zeros((height, width), dtype=bool)
            mask[top + inset_y : bottom - inset_y, left + inset_x : right - inset_x] = True
            masks.append(mask)
        return masks


class SkinPixelRefiner:
    """Keeps skin-chroma pixels inside skin regions, dropping eyes, hair, and most of the lips.

    The YCrCb chroma box (Chai & Ngan 1999) is defined on chroma only, so it covers the full
    range of skin lightness rather than favoring lighter skin.
    """

    def __init__(self, cr_range: tuple[int, int], cb_range: tuple[int, int]) -> None:
        self._cr_range = cr_range
        self._cb_range = cb_range

    def refine(
        self, preview: PreviewImage, label: ReferenceClass, mask: PreviewMask
    ) -> PreviewMask:
        if label is not ReferenceClass.SKIN:
            return mask
        ycrcb = cv2.cvtColor(preview, cv2.COLOR_RGB2YCrCb)
        cr, cb = ycrcb[..., 1], ycrcb[..., 2]
        chroma_ok = (
            (cr >= self._cr_range[0])
            & (cr <= self._cr_range[1])
            & (cb >= self._cb_range[0])
            & (cb <= self._cb_range[1])
        )
        return np.asarray(mask & chroma_ok)
