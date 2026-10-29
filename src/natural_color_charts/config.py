"""Typed configuration loaded from YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
CONFIG_ENV_VAR = "NCC_CONFIG"


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SpectralConfig(_Section):
    wavelength_start_nm: float
    wavelength_stop_nm: float
    wavelength_step_nm: float


class IoConfig(_Section):
    raw_half_size: bool
    saturation_threshold: float
    preview_long_edge_px: int


class CameraConfig(_Section):
    sensitivity_dir: Path
    include_builtin: bool
    fuzzy_match_cutoff: float
    use_same_make_fallback: bool


class OllamaConfig(_Section):
    host: str
    model: str
    timeout_s: float


class AnthropicConfig(_Section):
    model: str
    max_tokens: int


class ContextConfig(_Section):
    client: Literal["none", "ollama", "anthropic"]
    max_retries: int
    preview_long_edge_px: int
    default_families: list[str]
    default_candidate_confidence: float
    default_ambiance_strength: float
    ollama: OllamaConfig
    anthropic: AnthropicConfig


class DetectionConfig(_Section):
    detector: Literal["none", "manual", "transformers"]
    segmenter: Literal["box", "sam"]
    device: str
    detector_model: str
    segmenter_model: str
    score_threshold: float
    max_regions_per_label: int
    skin_cr_range: tuple[int, int]
    skin_cb_range: tuple[int, int]
    prompts: dict[str, list[str]]


class PatchConfig(_Section):
    erosion_fraction: float
    noise_floor: float
    specular_luminance_quantile: float
    specular_chroma_ratio: float
    statistic: Literal["trimmed_mean", "median"]
    trim_fraction: float
    min_valid_pixels: int
    full_weight_pixels: int
    full_weight_luminance: float
    outdoor_paper_factor: float
    class_prior_strength: dict[str, float]


class PriorConfig(_Section):
    directory: Path
    bootstrap_if_missing: bool
    components: int
    min_samples_for_pca: int
    neutral_smooth_sigma: float


class FamilyRange(_Section):
    cct_min_k: float
    cct_max_k: float


class EstimationConfig(_Section):
    fallback_chain: list[str]
    families: dict[str, FamilyRange]
    grid_step_mired: float
    chromaticity_sigma: float
    context_prior_weight: float
    robust_loss: str
    robust_scale: float
    loo_min_references: int
    loo_outlier_angle_deg: float
    confidence_residual_scale_deg: float
    fallback_confidence: float
    shades_of_gray_power: float
    gray_edge_power: float
    gray_edge_sigma: float
    white_patch_percentile: float
    statistics_max_pixels: int


class CorrectionConfig(_Section):
    training_set: str
    target_illuminant: str
    refine_delta_e: bool
    raster_method: Literal["bradford", "spectral"]
    adaptation_transform: str


class RenderConfig(_Section):
    exposure_percentile: float
    exposure_target: float
    normalize_raster_exposure: bool


class OutputConfig(_Section):
    directory: Path
    jpeg_quality: int
    overlay_alpha_max: float


class EvaluationConfig(_Section):
    chart_mask_value: float
    white_patch_index: int


class AppConfig(_Section):
    """Root configuration for the whole pipeline."""

    spectral: SpectralConfig
    io: IoConfig
    camera: CameraConfig
    context: ContextConfig
    detection: DetectionConfig
    patches: PatchConfig
    priors: PriorConfig
    estimation: EstimationConfig
    correction: CorrectionConfig
    render: RenderConfig
    output: OutputConfig
    evaluation: EvaluationConfig


def resolve_config_path(path: Path | None = None) -> Path:
    """Return the explicit path, the NCC_CONFIG path, or the bundled default."""
    if path is not None:
        return path
    env_path = os.environ.get(CONFIG_ENV_VAR)
    return Path(env_path) if env_path else DEFAULT_CONFIG_PATH


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load the YAML config and apply nested overrides such as {"context": {"client": "none"}}."""
    raw = yaml.safe_load(resolve_config_path(path).read_text())
    merged = merge_nested(raw, overrides or {})
    return AppConfig.model_validate(merged)


def merge_nested(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into a copy of base."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged
