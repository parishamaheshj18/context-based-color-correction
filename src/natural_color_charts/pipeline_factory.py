"""Composition root: the only place concrete stage classes are created."""

from __future__ import annotations

from natural_color_charts.camera.repository import CameraSensitivityRepository
from natural_color_charts.camera.selector import CameraSensitivitySelector
from natural_color_charts.config import AppConfig, EstimationConfig
from natural_color_charts.correction.base import CorrectionBuilder, FallbackCorrectionBuilder
from natural_color_charts.correction.matrix_builders import (
    BradfordCorrectionBuilder,
    CameraMatrixCorrectionBuilder,
)
from natural_color_charts.correction.renderer import SrgbRenderer
from natural_color_charts.correction.spectral_builder import SpectralCorrectionMatrixBuilder
from natural_color_charts.domain import WavelengthGrid
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
from natural_color_charts.spectral.datasets import (
    chart_reflectances,
    cie_1931_cmfs,
    standard_illuminant_spd,
)
from natural_color_charts.spectral.illuminants import (
    BlackbodyFamily,
    DaylightFamily,
    IlluminantFamily,
)

STATISTICAL_ESTIMATORS = ("gray_world", "shades_of_gray", "white_patch", "gray_edge", "as_shot")


def build_grid(config: AppConfig) -> WavelengthGrid:
    """Shared wavelength grid."""
    spectral = config.spectral
    return WavelengthGrid(
        spectral.wavelength_start_nm, spectral.wavelength_stop_nm, spectral.wavelength_step_nm
    )


def build_families(config: AppConfig, names: list[str] | None = None) -> list[IlluminantFamily]:
    """Illuminant families from config, optionally restricted to the given names."""
    grid = build_grid(config)
    constructors: dict[str, type[DaylightFamily] | type[BlackbodyFamily]] = {
        "daylight": DaylightFamily,
        "blackbody": BlackbodyFamily,
    }
    selected = names or list(config.estimation.families)
    families: list[IlluminantFamily] = []
    for name in selected:
        bounds = config.estimation.families[name]
        families.append(constructors[name](grid, bounds.cct_min_k, bounds.cct_max_k))
    return families


def build_repository(config: AppConfig) -> CameraSensitivityRepository:
    """Sensitivity repository over the configured data directory."""
    return CameraSensitivityRepository(
        build_grid(config), config.camera.sensitivity_dir, config.camera.include_builtin
    )


def build_selector(config: AppConfig) -> CameraSensitivitySelector:
    """Camera sensitivity selector."""
    return CameraSensitivitySelector(
        build_repository(config),
        config.camera.fuzzy_match_cutoff,
        config.camera.use_same_make_fallback,
    )


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
    """Spectral matrix when sensitivities exist, then the RAW camera matrix, then Bradford."""
    grid = build_grid(config)
    _, reflectances = chart_reflectances(config.correction.training_set, grid)
    spectral = SpectralCorrectionMatrixBuilder(
        training_reflectances_kxn=reflectances,
        cmfs_3xn=cie_1931_cmfs(grid),
        target_spd=standard_illuminant_spd(config.correction.target_illuminant, grid),
        families=build_families(config),
        refine_delta_e=config.correction.refine_delta_e,
        allow_raster=config.correction.raster_method == "spectral",
    )
    return FallbackCorrectionBuilder(
        [
            spectral,
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
        sensitivity_selector=build_selector(config),
        estimator=build_estimator(config),
        correction_builder=build_correction_builder(config),
        renderer=renderer,
        adaptation_transform=config.correction.adaptation_transform,
    )
