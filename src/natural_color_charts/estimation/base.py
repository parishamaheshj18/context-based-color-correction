"""Illuminant estimator interface and the fallback chain."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from natural_color_charts.domain import (
    CameraSensitivity,
    IlluminantEstimate,
    LinearImage,
    PatchMeasurement,
    SceneContext,
)

logger = logging.getLogger(__name__)


class EstimationUnavailableError(RuntimeError):
    """Raised when an estimator lacks the inputs it needs, so the next one should be tried."""


@dataclass(frozen=True, eq=False)
class EstimationInput:
    """Everything an estimator may use."""

    image: LinearImage
    patches: Sequence[PatchMeasurement] = ()
    context: SceneContext = field(default_factory=SceneContext)
    sensitivity: CameraSensitivity | None = None


@dataclass(frozen=True, eq=False)
class EstimationOutcome:
    """The estimate plus the patches as the estimator finally judged them."""

    estimate: IlluminantEstimate
    patches: tuple[PatchMeasurement, ...] = ()


class IlluminantEstimator(Protocol):
    """Estimates the scene illuminant."""

    name: str

    def estimate(self, request: EstimationInput) -> EstimationOutcome: ...


class FallbackIlluminantEstimator:
    """Tries estimators in order and returns the first that succeeds."""

    name = "fallback_chain"

    def __init__(self, estimators: Sequence[IlluminantEstimator]) -> None:
        if not estimators:
            raise ValueError("At least one estimator is required")
        self._estimators = list(estimators)

    def estimate(self, request: EstimationInput) -> EstimationOutcome:
        skipped: list[str] = []
        for position, estimator in enumerate(self._estimators):
            try:
                outcome = estimator.estimate(request)
            except EstimationUnavailableError as error:
                logger.info("Estimator %s unavailable: %s", estimator.name, error)
                skipped.append(f"{estimator.name}: {error}")
                continue
            return _mark_fallback(outcome, skipped, is_fallback=position > 0)
        raise EstimationUnavailableError("; ".join(skipped))


def _mark_fallback(
    outcome: EstimationOutcome, skipped: list[str], is_fallback: bool
) -> EstimationOutcome:
    estimate = outcome.estimate
    marked = dataclasses.replace(
        estimate,
        is_fallback=estimate.is_fallback or is_fallback,
        notes=estimate.notes + tuple(f"skipped {reason}" for reason in skipped),
    )
    return dataclasses.replace(outcome, estimate=marked)
