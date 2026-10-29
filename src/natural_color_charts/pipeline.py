"""Orchestrates the pipeline stages; concrete stages are injected by pipeline_factory."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from natural_color_charts.camera.selector import CameraSensitivitySelector
from natural_color_charts.context.base import SceneContextProvider
from natural_color_charts.correction.adaptation import adaptation_degree, apply_adaptation_degree
from natural_color_charts.correction.base import CorrectionBuilder
from natural_color_charts.correction.renderer import RenderedImage, SrgbRenderer
from natural_color_charts.detection.patch_statistics import PatchStatisticsCalculator
from natural_color_charts.detection.region_finder import ReferenceRegionFinder
from natural_color_charts.domain import (
    CameraSensitivity,
    CorrectionTransform,
    IlluminantEstimate,
    LinearImage,
    PatchMeasurement,
    PreviewImage,
    ReferenceRegion,
    SceneContext,
    WavelengthGrid,
)
from natural_color_charts.estimation.base import EstimationInput, IlluminantEstimator
from natural_color_charts.io.loaders import ImageLoader
from natural_color_charts.io.preview import render_preview

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class PipelineResult:
    """All intermediate and final products of one run."""

    image: LinearImage
    preview: PreviewImage
    sensitivity: CameraSensitivity | None
    context: SceneContext
    regions: tuple[ReferenceRegion, ...]
    patches: tuple[PatchMeasurement, ...]
    estimate: IlluminantEstimate
    transform: CorrectionTransform
    rendered: RenderedImage
    grid: WavelengthGrid


@dataclass(frozen=True, eq=False)
class PipelineStages:
    """The injected stage implementations."""

    loader: ImageLoader
    sensitivity_selector: CameraSensitivitySelector
    context_provider: SceneContextProvider
    region_finder: ReferenceRegionFinder
    patch_calculator: PatchStatisticsCalculator
    estimator: IlluminantEstimator
    correction_builder: CorrectionBuilder
    renderer: SrgbRenderer


class ColorCorrectionPipeline:
    """Runs the stages in order for one image."""

    def __init__(
        self,
        stages: PipelineStages,
        grid: WavelengthGrid,
        preview_long_edge_px: int,
        adaptation_transform: str,
    ) -> None:
        self._stages = stages
        self._grid = grid
        self._preview_px = preview_long_edge_px
        self._adaptation_transform = adaptation_transform

    def run(
        self, path: Path, camera: str | None = None, intent: str | None = None
    ) -> PipelineResult:
        """Correct the image at path, optionally forcing a camera and stating an intent."""
        return self.run_image(self._stages.loader.load(path), camera, intent)

    def run_image(
        self, image: LinearImage, camera: str | None = None, intent: str | None = None
    ) -> PipelineResult:
        """Correct an already loaded image."""
        stages = self._stages
        sensitivity = stages.sensitivity_selector.select(image, camera)
        preview = render_preview(image, self._preview_px)
        context = stages.context_provider.analyze(preview, intent)
        regions = tuple(stages.region_finder.find(image, preview, context))
        patches = tuple(stages.patch_calculator.measure(image, regions, context))
        outcome = stages.estimator.estimate(
            EstimationInput(image=image, patches=patches, context=context, sensitivity=sensitivity)
        )
        transform = stages.correction_builder.build(outcome.estimate, image, sensitivity)
        degree = adaptation_degree(context.user_intent)
        transform = apply_adaptation_degree(transform, degree, self._adaptation_transform)
        logger.info(
            "Estimated white %s via %s", outcome.estimate.white_rgb, outcome.estimate.method
        )
        return PipelineResult(
            image=image,
            preview=preview,
            sensitivity=sensitivity,
            context=context,
            regions=regions,
            patches=outcome.patches or patches,
            estimate=outcome.estimate,
            transform=transform,
            rendered=stages.renderer.render(image, transform),
            grid=self._grid,
        )
