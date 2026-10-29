"""Builds the static showcase site's data: before/after renders, AI context, and chart scores.

The chart in a test photo is hidden from every stage (VLM, detector, estimator) and used only
afterwards, to measure the error and to show its grays rendering neutral.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image

from natural_color_charts.colorimetry import encode_srgb
from natural_color_charts.config import AppConfig, load_config
from natural_color_charts.correction.adaptation import apply_adaptation_degree
from natural_color_charts.domain import (
    NEUTRAL_CLASSES,
    CorrectionTransform,
    IlluminantEstimate,
    InputMode,
    LinearImage,
    normalize_white,
)
from natural_color_charts.estimation.base import EstimationInput
from natural_color_charts.estimation.statistical import GrayWorldEstimator
from natural_color_charts.evaluation.chart import ChartAnnotation, chart_white, sample_chart
from natural_color_charts.evaluation.metrics import (
    chart_delta_e00,
    recovery_angular_error,
    reference_chart_xyz,
)
from natural_color_charts.io.preview import downscale_image
from natural_color_charts.io.writers import to_uint8
from natural_color_charts.pipeline import PipelineResult
from natural_color_charts.pipeline_factory import (
    build_correction_builder,
    build_grid,
    build_loader,
    build_pipeline,
    build_renderer,
    build_selector,
)
from natural_color_charts.reporting.overlay import render_overlay
from natural_color_charts.reporting.report import to_jsonable
from natural_color_charts.spectral.illuminants import DaylightBasis

logger = logging.getLogger(__name__)


class ShowcaseError(RuntimeError):
    """A case could not be built faithfully."""


@dataclass(frozen=True)
class ShowcaseCase:
    """One test photo on the showcase page."""

    case_id: str
    title: str
    caption: str
    image: Path
    intent: str | None = None
    chart: ChartAnnotation | None = None


@dataclass(frozen=True)
class ShowcaseManifest:
    """showcase/cases.yaml: shared settings plus the cases."""

    cases: tuple[ShowcaseCase, ...]
    config_overrides: dict[str, Any]
    processing_long_edge_px: int
    display_long_edge_px: int


def load_manifest(path: Path) -> ShowcaseManifest:
    """Parse the YAML manifest; image paths are relative to the working directory."""
    raw = yaml.safe_load(path.read_text())
    cases = tuple(
        ShowcaseCase(
            case_id=entry["id"],
            title=entry["title"],
            caption=entry.get("caption", ""),
            image=Path(entry["image"]),
            intent=entry.get("intent"),
            chart=_chart(entry.get("chart")),
        )
        for entry in raw["cases"]
    )
    return ShowcaseManifest(
        cases=cases,
        config_overrides=raw.get("config_overrides", {}),
        processing_long_edge_px=int(raw.get("processing_long_edge_px", 2048)),
        display_long_edge_px=int(raw.get("display_long_edge_px", 1600)),
    )


def _chart(entry: dict[str, Any] | None) -> ChartAnnotation | None:
    if not entry:
        return None
    corners = tuple((float(x), float(y)) for x, y in entry["corners"])
    box = tuple(float(v) for v in entry["mask_box"])
    return ChartAnnotation(corners=corners, mask_box_xyxy=box)  # type: ignore[arg-type]


def camera_label(make: str | None, model: str | None) -> str | None:
    """'Canon EOS Rebel T8i' rather than 'Canon Canon EOS Rebel T8i'."""
    if model and make and model.lower().startswith(make.lower()):
        return model
    return " ".join(part for part in (make, model) if part) or None


def hide_annotated_chart(image: LinearImage, chart: ChartAnnotation) -> LinearImage:
    """Black out and exclude the chart so no stage can use it."""
    mask = chart.mask(image.width, image.height)
    rgb = image.rgb.copy()
    rgb[mask] = 0.0
    return dataclasses.replace(image, rgb=rgb, excluded_mask=mask)


def save_jpeg(path: Path, image_uint8: NDArray[np.uint8], long_edge_px: int) -> str:
    """Write a display-size JPEG and return its file name."""
    picture = Image.fromarray(image_uint8)
    picture.thumbnail((long_edge_px, long_edge_px), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    picture.save(path, quality=88, optimize=True, progressive=True)
    return path.name


class ShowcaseBuilder:
    """Runs each case and writes images plus a JSON record for the page."""

    def __init__(self, manifest: ShowcaseManifest, site_dir: Path, config: AppConfig) -> None:
        self._manifest = manifest
        self._site_dir = site_dir
        self._config = config
        self._pipeline = build_pipeline(config)
        self._loader = build_loader(config)
        self._selector = build_selector(config)
        self._builder = build_correction_builder(config)
        self._renderer = build_renderer(config)
        self._gray_world = GrayWorldEstimator(1.0, config.estimation.statistics_max_pixels)
        self._reference_xyz = reference_chart_xyz(
            build_grid(config), config.correction.training_set
        )

    @classmethod
    def from_manifest(cls, manifest_path: Path, site_dir: Path) -> ShowcaseBuilder:
        manifest = load_manifest(manifest_path)
        return cls(manifest, site_dir, load_config(overrides=manifest.config_overrides))

    def build(self, only: set[str] | None = None) -> Path:
        """Process the cases (all, or just `only`) and merge them into data/cases.json."""
        data_path = self._site_dir / "data" / "cases.json"
        existing = json.loads(data_path.read_text())["cases"] if data_path.exists() else []
        records = {record["id"]: record for record in existing}
        for case in self._manifest.cases:
            if only and case.case_id not in only:
                continue
            logger.info("Building case %s", case.case_id)
            records[case.case_id] = self.build_case(case)
        ordered = [records[c.case_id] for c in self._manifest.cases if c.case_id in records]
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            json.dumps({"cases": ordered, "pipeline": self._pipeline_info()}, indent=1)
        )
        return data_path

    def build_case(self, case: ShowcaseCase) -> dict[str, Any]:
        started = time.perf_counter()
        image = downscale_image(
            self._loader.load(case.image), self._manifest.processing_long_edge_px
        )
        hidden = hide_annotated_chart(image, case.chart) if case.chart else image
        result = self._pipeline.run_image(hidden, intent=case.intent)
        if self._config.context.client != "none" and result.context.source != "vlm":
            # Publishing the fallback context would present "AI reasoning" that never happened.
            raise ShowcaseError(f"{case.case_id}: the VLM gave no usable answer; see the log above")
        renders = self._renders(image, hidden, result)
        record: dict[str, Any] = {
            "id": case.case_id,
            "title": case.title,
            "caption": case.caption,
            "intent": case.intent,
            "source_file": case.image.name,
            "camera": camera_label(image.metadata.make, image.metadata.model),
            "input_mode": image.mode.value,
            "size": [image.width, image.height],
            "images": self._save_images(case, result, renders),
            "ai": self._ai_section(result),
            "physics": self._physics_section(result),
            "seconds": round(time.perf_counter() - started, 1),
        }
        if case.chart:
            record["ground_truth"] = self._score(image, case.chart, renders)
        return record

    def _renders(
        self, image: LinearImage, hidden: LinearImage, result: PipelineResult
    ) -> dict[str, tuple[CorrectionTransform | None, NDArray[np.float32]]]:
        # The final transform is applied to the full photo, chart included.
        ours = self._renderer.render(image, result.transform)
        gain = ours.exposure_gain
        sensitivity = self._selector.select(image)
        gray_estimate = self._gray_world.estimate(EstimationInput(image=hidden)).estimate
        gray = self._transform_for(gray_estimate, image, sensitivity)
        renders: dict[str, tuple[CorrectionTransform | None, NDArray[np.float32]]] = {
            "ours": (result.transform, ours.encoded_srgb),
            "gray_world": (gray, self._renderer.render(image, gray, gain).encoded_srgb),
        }
        if image.mode is InputMode.RASTER:
            renders["before"] = (None, encode_srgb(image.rgb))
        else:
            white = image.metadata.as_shot_white_rgb
            as_shot = IlluminantEstimate(
                white_rgb=normalize_white(white if white is not None else np.ones(3)),
                method="as_shot",
                confidence=1.0,
            )
            transform = self._transform_for(as_shot, image, sensitivity)
            renders["before"] = (
                transform,
                self._renderer.render(image, transform, gain).encoded_srgb,
            )
        return renders

    def _transform_for(
        self, estimate: IlluminantEstimate, image: LinearImage, sensitivity: Any
    ) -> CorrectionTransform:
        transform = self._builder.build(estimate, image, sensitivity)
        return apply_adaptation_degree(transform, 1.0, self._config.correction.adaptation_transform)

    def _save_images(
        self,
        case: ShowcaseCase,
        result: PipelineResult,
        renders: dict[str, tuple[CorrectionTransform | None, NDArray[np.float32]]],
    ) -> dict[str, str]:
        directory = self._site_dir / "assets" / "cases" / case.case_id
        size = self._manifest.display_long_edge_px
        names = {
            key: save_jpeg(directory / f"{key}.jpg", to_uint8(encoded), size)
            for key, (_, encoded) in renders.items()
        }
        names["overlay"] = save_jpeg(
            directory / "overlay.jpg",
            render_overlay(result, self._config.output.overlay_alpha_max),
            size,
        )
        names["thumb"] = save_jpeg(directory / "thumb.jpg", to_uint8(renders["ours"][1]), 480)
        return {key: f"assets/cases/{case.case_id}/{name}" for key, name in names.items()}

    def _ai_section(self, result: PipelineResult) -> dict[str, Any]:
        context = result.context
        return {
            "source": context.source,
            "model": self._model_name(),
            "environment": context.environment,
            "time_of_day": context.time_of_day,
            "weather": context.weather,
            "light_sources": list(context.visible_light_sources),
            "hypotheses": to_jsonable(context.illuminant_hypotheses),
            "candidates": to_jsonable(context.reference_candidates),
            "intent": to_jsonable(context.user_intent),
            "rationale": context.rationale,
            "detector": {
                "transformers": self._config.detection.detector_model,
                "vlm": f"{self._model_name()} boxes",
            }.get(self._config.detection.detector, self._config.detection.detector),
            "segmenter": self._config.detection.segmenter_model
            if self._config.detection.segmenter == "sam"
            else self._config.detection.segmenter,
            "regions": [self._region_entry(result, patch) for patch in result.patches],
        }

    @staticmethod
    def _unused_reason(result: PipelineResult, patch: Any) -> str:
        if patch.rejection_reason:
            return str(patch.rejection_reason)
        if patch.kept and patch.label not in NEUTRAL_CLASSES and result.sensitivity is None:
            # Non-neutral materials are only informative through the spectral model.
            return "needs this camera's spectral curves"
        return "not used by the estimator"

    @staticmethod
    def _region_entry(result: PipelineResult, patch: Any) -> dict[str, Any]:
        region = (
            result.regions[patch.region_index]
            if 0 <= patch.region_index < len(result.regions)
            else None
        )
        used = patch.kept and patch.label.value in result.estimate.used_labels
        return {
            "label": patch.label.value,
            "description": region.description if region else "",
            "box_xyxy": to_jsonable(region.box_xyxy) if region else None,
            "detector_score": round(region.detector_score, 3) if region else None,
            "pixels": patch.pixel_count,
            "weight": round(patch.weight, 3),
            "kept": patch.kept,
            "reason": "" if used else ShowcaseBuilder._unused_reason(result, patch),
            "used": used,
        }

    def _physics_section(self, result: PipelineResult) -> dict[str, Any]:
        estimate = result.estimate
        section: dict[str, Any] = {
            "estimator": estimate.method,
            "is_fallback": estimate.is_fallback,
            "family": estimate.family,
            "cct_k": estimate.cct_k,
            "cct_uncertainty_k": estimate.cct_uncertainty_k,
            "confidence": round(estimate.confidence, 3),
            "white_rgb": to_jsonable(estimate.white_rgb),
            "notes": list(estimate.notes),
            "correction": result.transform.method,
            "adaptation_degree": result.transform.adaptation_degree,
            "camera_curves": result.sensitivity.name if result.sensitivity else None,
            "camera_curves_reason": result.sensitivity.selection_reason
            if result.sensitivity
            else None,
        }
        if estimate.spectrum is not None:
            spectrum = np.asarray(estimate.spectrum)
            d65 = DaylightBasis(result.grid).spd(6504.0)
            section["spectrum"] = {
                "wavelengths_nm": to_jsonable(result.grid.wavelengths_nm),
                "relative_power": to_jsonable(spectrum / np.max(spectrum)),
                "d65_relative_power": to_jsonable(d65 / np.max(d65)),
            }
        return section

    def _score(
        self,
        image: LinearImage,
        chart: ChartAnnotation,
        renders: dict[str, tuple[CorrectionTransform | None, NDArray[np.float32]]],
    ) -> dict[str, Any]:
        patches = sample_chart(image, chart)
        truth = chart_white(patches)
        scores: dict[str, Any] = {"truth_white_rgb": to_jsonable(truth)}
        for key, (transform, _) in renders.items():
            if transform is None:
                continue
            white = np.linalg.solve(transform.camera_to_xyz, transform.target_white_xyz)
            scores[key] = {
                "angular_error_deg": round(recovery_angular_error(white, truth), 2),
                "delta_e00": round(
                    chart_delta_e00(
                        transform.camera_to_xyz,
                        patches,
                        self._reference_xyz,
                        self._config.evaluation.white_patch_index,
                    ),
                    2,
                ),
            }
        return scores

    def _model_name(self) -> str | None:
        context = self._config.context
        if context.client == "anthropic":
            return context.anthropic.model
        if context.client == "ollama":
            return context.ollama.model
        return None

    def _pipeline_info(self) -> dict[str, Any]:
        return {
            "vlm": self._model_name(),
            "detector": self._config.detection.detector,
            "segmenter": self._config.detection.segmenter_model,
            "processing_long_edge_px": self._manifest.processing_long_edge_px,
        }
