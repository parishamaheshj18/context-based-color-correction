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
        preview = self.render_preview(image)
        context = self._stages.context_provider.analyze(preview, intent)
        return self.run_with_context(image, preview, context, camera)

    def render_preview(self, image: LinearImage) -> PreviewImage:
        """The display preview used for scene analysis, detection, and overlays."""
        return render_preview(image, self._preview_px)

    def measure_references(
        self, image: LinearImage, preview: PreviewImage, context: SceneContext
    ) -> tuple[tuple[ReferenceRegion, ...], tuple[PatchMeasurement, ...]]:
        """Stages 4 and 5: reference regions and their patch measurements."""
        regions = tuple(self._stages.region_finder.find(image, preview, context))
        return regions, tuple(self._stages.patch_calculator.measure(image, regions, context))

    def run_with_context(
        self,
        image: LinearImage,
        preview: PreviewImage,
        context: SceneContext,
        camera: str | None = None,
    ) -> PipelineResult:
        """Stages 2 and 4-9 with a given scene context, e.g. one proposed by a model in training."""
        stages = self._stages
        sensitivity = stages.sensitivity_selector.select(image, camera)
        regions, patches = self.measure_references(image, preview, context)
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

    def estimate_illuminant(
        self,
        image: LinearImage,
        patches: tuple[PatchMeasurement, ...],
        context: SceneContext,
        camera: str | None = None,
    ) -> IlluminantEstimate:
        """Stage 7 alone, for experiments that vary which references are used."""
        sensitivity = self._stages.sensitivity_selector.select(image, camera)
        request = EstimationInput(
            image=image, patches=patches, context=context, sensitivity=sensitivity
        )
        return self._stages.estimator.estimate(request).estimate

    def analyze_scene(self, preview: PreviewImage, intent: str | None) -> SceneContext:
        """Stage 3 alone."""
        return self._stages.context_provider.analyze(preview, intent)
