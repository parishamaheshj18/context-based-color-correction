"""Stage 11: run the pipeline, its ablations, and baselines over a benchmark dataset."""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from natural_color_charts.colorimetry import D65_WHITE_XYZ
from natural_color_charts.config import AppConfig, merge_nested
from natural_color_charts.correction.base import CorrectionBuilder
from natural_color_charts.correction.matrix_fitting import fit_white_preserving_matrix
from natural_color_charts.domain import IlluminantEstimate, LinearImage, Matrix3x3
from natural_color_charts.estimation.base import EstimationInput, EstimationUnavailableError
from natural_color_charts.evaluation.ablations import ablation_overrides
from natural_color_charts.evaluation.dataset import BenchmarkSample, hide_chart, load_samples
from natural_color_charts.evaluation.metrics import (
    chart_delta_e00,
    recovery_angular_error,
    reference_chart_xyz,
    reproduction_angular_error,
    summarize_errors,
)
from natural_color_charts.pipeline_factory import (
    build_correction_builder,
    build_grid,
    build_loader,
    build_named_estimator,
    build_pipeline,
    build_selector,
)

logger = logging.getLogger(__name__)

CHART_UPPER_BOUND = "chart_fit_upper_bound"
EVALUATION_INTENT = "neutral"

_MethodOutput = tuple[IlluminantEstimate, Matrix3x3]
_Method = Callable[[LinearImage], _MethodOutput]


@dataclass(frozen=True)
class SampleResult:
    """Errors of one method on one image."""

    image: str
    method: str
    recovery_deg: float
    reproduction_deg: float
    delta_e00: float | None
    confidence: float
    is_fallback: bool
    estimator: str
    skin_group: str | None


class BenchmarkRunner:
    """Evaluates pipeline variants and baseline estimators against chart ground truth."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._loader = build_loader(config)
        self._reference_xyz = reference_chart_xyz(
            build_grid(config), config.correction.training_set
        )

    def run(
        self, dataset_dir: Path, ablations: Sequence[str], baselines: Sequence[str]
    ) -> list[SampleResult]:
        samples = load_samples(dataset_dir)
        methods = self._pipeline_methods(ablations) + self._baseline_methods(baselines)
        results: list[SampleResult] = []
        for sample in samples:
            image = hide_chart(self._loader.load(sample.image_path), sample.chart_mask_path)
            for name, method in methods:
                try:
                    estimate, matrix = method(image)
                except EstimationUnavailableError as error:
                    logger.warning("%s unavailable on %s: %s", name, sample.image_path.name, error)
                    continue
                results.append(self._score(sample, name, estimate, matrix))
            results.extend(self._upper_bound(sample, image))
        return results

    def _pipeline_methods(self, ablations: Sequence[str]) -> list[tuple[str, _Method]]:
        methods: list[tuple[str, _Method]] = []
        for name in ablations:
            config = self._config.model_validate(
                merge_nested(self._config.model_dump(), ablation_overrides(name))
            )
            pipeline = build_pipeline(config)

            def run(image: LinearImage, pipeline: Any = pipeline) -> _MethodOutput:
                result = pipeline.run_image(image, intent=EVALUATION_INTENT)
                return result.estimate, result.transform.camera_to_xyz

            methods.append((f"pipeline:{name}", run))
        return methods

    def _baseline_methods(self, baselines: Sequence[str]) -> list[tuple[str, _Method]]:
        selector = build_selector(self._config)
        builder = build_correction_builder(self._config)
        methods: list[tuple[str, _Method]] = []
        for name in baselines:
            estimator = build_named_estimator(name, self._config)

            def run(
                image: LinearImage, estimator: Any = estimator, builder: CorrectionBuilder = builder
            ) -> _MethodOutput:
                sensitivity = selector.select(image)
                outcome = estimator.estimate(EstimationInput(image=image, sensitivity=sensitivity))
                matrix = builder.build(outcome.estimate, image, sensitivity).camera_to_xyz
                return outcome.estimate, matrix

            methods.append((f"baseline:{name}", run))
        return methods

    def _score(
        self,
        sample: BenchmarkSample,
        method: str,
        estimate: IlluminantEstimate,
        matrix: Matrix3x3,
    ) -> SampleResult:
        delta_e = None
        if sample.chart_patches_kx3 is not None:
            delta_e = chart_delta_e00(
                matrix,
                sample.chart_patches_kx3,
                self._reference_xyz,
                self._config.evaluation.white_patch_index,
            )
        return SampleResult(
            image=sample.image_path.name,
            method=method,
            recovery_deg=recovery_angular_error(estimate.white_rgb, sample.truth_rgb),
            reproduction_deg=reproduction_angular_error(estimate.white_rgb, sample.truth_rgb),
            delta_e00=delta_e,
            confidence=estimate.confidence,
            is_fallback=estimate.is_fallback,
            estimator=estimate.method,
            skin_group=sample.skin_group,
        )

    def _upper_bound(self, sample: BenchmarkSample, image: LinearImage) -> list[SampleResult]:
        """A matrix fitted to the chart itself: the best any global 3×3 correction can do."""
        patches = sample.chart_patches_kx3
        if patches is None:
            return []
        # The constraint white must carry the chart's exposure, or it fights the patch fit.
        white_index = self._config.evaluation.white_patch_index
        exposure = patches[white_index, 1] / (
            sample.truth_rgb[1] * self._reference_xyz[white_index, 1]
        )
        matrix = fit_white_preserving_matrix(
            patches, self._reference_xyz, sample.truth_rgb * exposure, D65_WHITE_XYZ
        )
        estimate = IlluminantEstimate(sample.truth_rgb, CHART_UPPER_BOUND, 1.0)
        return [self._score(sample, CHART_UPPER_BOUND, estimate, matrix)]


def summarize_results(results: Sequence[SampleResult]) -> dict[str, Any]:
    """Per-method error statistics, plus a per-skin-group breakdown when groups are labeled."""
    by_method: dict[str, list[SampleResult]] = defaultdict(list)
    for result in results:
        by_method[result.method].append(result)
    return {method: _method_summary(rows) for method, rows in by_method.items()}


def _method_summary(rows: Sequence[SampleResult]) -> dict[str, Any]:
    delta_es = [r.delta_e00 for r in rows if r.delta_e00 is not None]
    summary: dict[str, Any] = {
        "recovery_deg": summarize_errors([r.recovery_deg for r in rows]),
        "reproduction_deg": summarize_errors([r.reproduction_deg for r in rows]),
        "delta_e00_mean": float(np.mean(delta_es)) if delta_es else None,
        "fallback_rate": float(np.mean([r.is_fallback for r in rows])),
    }
    groups = sorted({r.skin_group for r in rows if r.skin_group})
    if groups:
        summary["recovery_deg_by_skin_group"] = {
            group: summarize_errors([r.recovery_deg for r in rows if r.skin_group == group])
            for group in groups
        }
    return summary


def write_results(directory: Path, results: Sequence[SampleResult]) -> tuple[Path, Path]:
    """Write per-image results.csv and summary.json."""
    directory.mkdir(parents=True, exist_ok=True)
    rows_path, summary_path = directory / "results.csv", directory / "summary.json"
    fields = [f.name for f in dataclasses.fields(SampleResult)]
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dataclasses.asdict(r) for r in results)
    summary_path.write_text(json.dumps(summarize_results(results), indent=2))
    return rows_path, summary_path


def format_summary_table(summary: dict[str, Any]) -> str:
    """Plain-text table of the headline numbers."""
    lines = [f"{'method':40s} {'mean':>6s} {'median':>6s} {'tri':>6s} {'w25':>6s} {'ΔE00':>6s}"]
    for method, stats in summary.items():
        recovery = stats["recovery_deg"]
        delta_e = stats["delta_e00_mean"]
        lines.append(
            f"{method:40s} {recovery['mean']:6.2f} {recovery['median']:6.2f} "
            f"{recovery['trimean']:6.2f} {recovery['worst25']:6.2f} "
            f"{delta_e if delta_e is not None else float('nan'):6.2f}"
        )
    return "\n".join(lines)
