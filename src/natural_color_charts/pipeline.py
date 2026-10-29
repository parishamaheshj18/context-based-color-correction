"""Orchestrates the pipeline stages; concrete stages are injected by pipeline_factory."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from natural_color_charts.correction.adaptation import adaptation_degree, apply_adaptation_degree
from natural_color_charts.correction.base import CorrectionBuilder
from natural_color_charts.correction.renderer import RenderedImage, SrgbRenderer
from natural_color_charts.domain import (
    CameraSensitivity,
    CorrectionTransform,
    IlluminantEstimate,
    LinearImage,
    SceneContext,
)
from natural_color_charts.estimation.base import EstimationInput, IlluminantEstimator
from natural_color_charts.io.loaders import ImageLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class PipelineResult:
    """All intermediate and final products of one run."""

    image: LinearImage
    sensitivity: CameraSensitivity | None
    context: SceneContext
    estimate: IlluminantEstimate
    transform: CorrectionTransform
    rendered: RenderedImage


class ColorCorrectionPipeline:
    """Runs the stages in order for one image."""

    def __init__(
        self,
        loader: ImageLoader,
        estimator: IlluminantEstimator,
        correction_builder: CorrectionBuilder,
        renderer: SrgbRenderer,
        adaptation_transform: str,
    ) -> None:
        self._loader = loader
        self._estimator = estimator
        self._correction_builder = correction_builder
        self._renderer = renderer
        self._adaptation_transform = adaptation_transform

    def run(self, path: Path) -> PipelineResult:
        """Correct the image at path."""
        return self.run_image(self._loader.load(path))

    def run_image(self, image: LinearImage) -> PipelineResult:
        """Correct an already loaded image."""
        context = SceneContext()
        outcome = self._estimator.estimate(EstimationInput(image=image, context=context))
        transform = self._correction_builder.build(outcome.estimate, image, None)
        degree = adaptation_degree(context.user_intent)
        transform = apply_adaptation_degree(transform, degree, self._adaptation_transform)
        return PipelineResult(
            image=image,
            sensitivity=None,
            context=context,
            estimate=outcome.estimate,
            transform=transform,
            rendered=self._renderer.render(image, transform),
        )
