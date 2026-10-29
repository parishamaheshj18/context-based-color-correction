"""Correction builder interface and fallback chain."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from natural_color_charts.domain import (
    CameraSensitivity,
    CorrectionTransform,
    IlluminantEstimate,
    LinearImage,
)

logger = logging.getLogger(__name__)


class CorrectionUnavailableError(RuntimeError):
    """Raised when a builder lacks the data it needs, so the next one should be tried."""


class CorrectionBuilder(Protocol):
    """Builds a camera-RGB-to-XYZ matrix that maps the estimated white to the target white."""

    name: str

    def build(
        self,
        estimate: IlluminantEstimate,
        image: LinearImage,
        sensitivity: CameraSensitivity | None,
    ) -> CorrectionTransform: ...


class FallbackCorrectionBuilder:
    """Tries builders in order and returns the first transform that can be built."""

    name = "fallback_chain"

    def __init__(self, builders: Sequence[CorrectionBuilder]) -> None:
        if not builders:
            raise ValueError("At least one builder is required")
        self._builders = list(builders)

    def build(
        self,
        estimate: IlluminantEstimate,
        image: LinearImage,
        sensitivity: CameraSensitivity | None,
    ) -> CorrectionTransform:
        reasons: list[str] = []
        for builder in self._builders:
            try:
                return builder.build(estimate, image, sensitivity)
            except CorrectionUnavailableError as error:
                logger.info("Correction builder %s unavailable: %s", builder.name, error)
                reasons.append(f"{builder.name}: {error}")
        raise CorrectionUnavailableError("; ".join(reasons))
