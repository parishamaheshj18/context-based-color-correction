"""Composition root: the only place concrete stage classes are created."""

from __future__ import annotations

from natural_color_charts.config import AppConfig, EstimationConfig
from natural_color_charts.correction.base import CorrectionBuilder, FallbackCorrectionBuilder
from natural_color_charts.correction.matrix_builders import (
    BradfordCorrectionBuilder,
    CameraMatrixCorrectionBuilder,
)
from natural_color_charts.correction.renderer import SrgbRenderer
from natural_color_charts.estimation.base import FallbackIlluminantEstimator, IlluminantEstimator
from natural_color_charts.estimation.statistical import (
    AsShotEstimator,
    GrayEdgeEstimator,
    GrayWorldEstimator,
    ShadesOfGrayEstimator,
    WhitePatchEstimator,
)
from natural_color_charts.io.loaders import (
    RasterImageLoader,
    RawImageLoader,
    SuffixDispatchingLoader,
)
from natural_color_charts.pipeline import ColorCorrectionPipeline

STATISTICAL_ESTIMATORS = ("gray_world", "shades_of_gray", "white_patch", "gray_edge", "as_shot")


def build_loader(config: AppConfig) -> SuffixDispatchingLoader:
    """RAW first, then display-referred formats."""
    threshold = config.io.saturation_threshold
    return SuffixDispatchingLoader(
        [RawImageLoader(threshold, config.io.raw_half_size), RasterImageLoader(threshold)]
    )


def build_statistical_estimator(name: str, config: EstimationConfig) -> IlluminantEstimator:
    """Create one of the classic baseline estimators by name."""
    confidence, max_pixels = config.fallback_confidence, config.statistics_max_pixels
    builders: dict[str, IlluminantEstimator] = {
        "gray_world": GrayWorldEstimator(confidence, max_pixels),
        "shades_of_gray": ShadesOfGrayEstimator(
            confidence, max_pixels, config.shades_of_gray_power
        ),
        "white_patch": WhitePatchEstimator(confidence, max_pixels, config.white_patch_percentile),
        "gray_edge": GrayEdgeEstimator(
            confidence, max_pixels, config.gray_edge_power, config.gray_edge_sigma
        ),
        "as_shot": AsShotEstimator(confidence),
    }
    if name not in builders:
        raise ValueError(f"Unknown estimator: {name}")
    return builders[name]


def build_estimator(config: AppConfig) -> IlluminantEstimator:
    """Chain the configured estimators, skipping those not yet available."""
    names = [n for n in config.estimation.fallback_chain if n in STATISTICAL_ESTIMATORS]
    return FallbackIlluminantEstimator(
        [build_statistical_estimator(name, config.estimation) for name in names or ["gray_world"]]
    )


def build_correction_builder(config: AppConfig) -> CorrectionBuilder:
    """Camera matrix when the RAW file has one, otherwise Bradford in linear sRGB."""
    return FallbackCorrectionBuilder(
        [
            CameraMatrixCorrectionBuilder(),
            BradfordCorrectionBuilder(config.correction.adaptation_transform),
        ]
    )


def build_pipeline(config: AppConfig) -> ColorCorrectionPipeline:
    """Assemble the full pipeline from configuration."""
    renderer = SrgbRenderer(
        config.render.exposure_percentile,
        config.render.exposure_target,
        config.render.normalize_raster_exposure,
    )
    return ColorCorrectionPipeline(
        loader=build_loader(config),
        estimator=build_estimator(config),
        correction_builder=build_correction_builder(config),
        renderer=renderer,
        adaptation_transform=config.correction.adaptation_transform,
    )
