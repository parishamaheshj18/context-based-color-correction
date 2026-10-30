"""Orchestrates the pipeline stages; concrete stages are injected by pipeline_factory."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from natural_color_charts.camera.selector import CameraSensitivitySelector
from natural_color_charts.colorimetry import D65_WHITE_XYZ, LINEAR_SRGB_TO_XYZ
from natural_color_charts.context.base import SceneContextProvider
from natural_color_charts.correction.adaptation import adaptation_degree, apply_adaptation_degree
from natural_color_charts.correction.base import CorrectionBuilder
from natural_color_charts.correction.renderer import RenderedImage, SrgbRenderer
from natural_color_charts.detection.anchors import AnchorRegionBuilder
from natural_color_charts.detection.patch_statistics import PatchStatisticsCalculator
from natural_color_charts.detection.region_finder import ReferenceRegionFinder
from natural_color_charts.domain import (
    CameraSensitivity,
    CorrectionTransform,
    IlluminantEstimate,
    InputMode,
    LinearImage,
    PatchMeasurement,
    PreviewImage,
    ReferenceRegion,
    SceneContext,
    UserColorAnchor,
    WavelengthGrid,
    normalize_white,
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
    neutral_transform: CorrectionTransform | None = None  # before intent's partial adaptation
    as_shot: RenderedImage | None = None  # the camera's own colors, rendered like the correction


@dataclass(frozen=True, eq=False)
class PipelineStages:
    """The injected stage implementations."""

    loader: ImageLoader
    sensitivity_selector: CameraSensitivitySelector
    context_provider: SceneContextProvider
    region_finder: ReferenceRegionFinder
    anchor_builder: AnchorRegionBuilder
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
        self,
        image: LinearImage,
        preview: PreviewImage,
        context: SceneContext,
        anchors: Sequence[UserColorAnchor] = (),
    ) -> tuple[tuple[ReferenceRegion, ...], tuple[PatchMeasurement, ...]]:
        """Stages 4 and 5: AI-found regions, the user's color matches, and their measurements."""
        found = self._stages.region_finder.find(image, preview, context)
        regions = tuple(found) + tuple(self._stages.anchor_builder.build(image, preview, anchors))
        return regions, tuple(self._stages.patch_calculator.measure(image, regions, context))

    def run_with_context(
        self,
        image: LinearImage,
        preview: PreviewImage,
        context: SceneContext,
        camera: str | None = None,
        anchors: Sequence[UserColorAnchor] = (),
    ) -> PipelineResult:
        """Stages 2 and 4-9 with a given scene context (and optional user color matches)."""
        stages = self._stages
        sensitivity = stages.sensitivity_selector.select(image, camera)
        regions, patches = self.measure_references(image, preview, context, anchors)
        outcome = stages.estimator.estimate(
            EstimationInput(image=image, patches=patches, context=context, sensitivity=sensitivity)
        )
        neutral = stages.correction_builder.build(outcome.estimate, image, sensitivity)
        degree = adaptation_degree(context.user_intent)
        transform = apply_adaptation_degree(neutral, degree, self._adaptation_transform)
        logger.info(
            "Estimated white %s via %s", outcome.estimate.white_rgb, outcome.estimate.method
        )
        rendered = stages.renderer.render(image, transform)
        as_shot = stages.renderer.render(
            image, self.as_shot_transform(image, sensitivity), rendered.exposure_gain
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
            rendered=rendered,
            grid=self._grid,
            neutral_transform=neutral,
            as_shot=as_shot,
        )

    def as_shot_transform(
        self, image: LinearImage, sensitivity: CameraSensitivity | None
    ) -> CorrectionTransform:
        """The camera's own white balance as a transform, so "as shot" renders like a correction."""
        if image.mode is InputMode.RASTER:
            # A display-referred file is already balanced, and its linear values are linear sRGB.
            return CorrectionTransform(LINEAR_SRGB_TO_XYZ, D65_WHITE_XYZ, D65_WHITE_XYZ, "as_shot")
        white = image.metadata.as_shot_white_rgb
        estimate = IlluminantEstimate(
            white_rgb=normalize_white(white if white is not None else np.ones(3)),
            method="as_shot",
            confidence=1.0,
        )
        return self._stages.correction_builder.build(estimate, image, sensitivity)

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
