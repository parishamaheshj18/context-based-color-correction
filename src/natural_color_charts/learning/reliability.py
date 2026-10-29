"""Stage 12: data and rewards for learning which references to trust.

1. Automatic labels: a reference is helpful if removing it raises the angular error against
   ground truth, harmful if removing it lowers the error.
2. Supervised records: the Stage 3 prompt with a target JSON whose usable flags follow the
   labels, for LoRA fine-tuning a small VLM.
3. Reward: −(angular error) of the full pipeline run with a model-proposed scene context,
   for GRPO.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from natural_color_charts.context.analyzer import SCENE_PROMPT
from natural_color_charts.context.schema import SceneAnalysisResponse
from natural_color_charts.domain import (
    LinearImage,
    PatchMeasurement,
    PreviewImage,
    ReferenceCandidate,
    ReferenceClass,
    SceneContext,
)
from natural_color_charts.evaluation.dataset import BenchmarkSample, hide_chart
from natural_color_charts.evaluation.metrics import recovery_angular_error
from natural_color_charts.io.loaders import ImageLoader
from natural_color_charts.pipeline import ColorCorrectionPipeline
from natural_color_charts.reporting.report import to_jsonable

logger = logging.getLogger(__name__)

INVALID_RESPONSE_REWARD = -45.0


class Helpfulness(StrEnum):
    """Effect of one reference on the illuminant estimate."""

    HELPFUL = "helpful"
    NEUTRAL = "neutral"
    HARMFUL = "harmful"


@dataclass(frozen=True)
class ReferenceLabel:
    """Leave-one-out effect of one reference region."""

    image: str
    region_index: int
    label: ReferenceClass
    box_xyxy: tuple[float, float, float, float] | None
    error_with_deg: float
    error_without_deg: float
    helpfulness: Helpfulness


def classify_effect(error_with: float, error_without: float, margin_deg: float) -> Helpfulness:
    """Helpful if dropping the reference makes the estimate worse by more than the margin."""
    if error_without - error_with > margin_deg:
        return Helpfulness.HELPFUL
    if error_with - error_without > margin_deg:
        return Helpfulness.HARMFUL
    return Helpfulness.NEUTRAL


def _without(patches: Sequence[PatchMeasurement], index: int) -> tuple[PatchMeasurement, ...]:
    return tuple(
        dataclasses.replace(p, kept=False, rejection_reason="left out") if i == index else p
        for i, p in enumerate(patches)
    )


class ReferenceLabeler:
    """Runs leave-one-reference-out estimates against chart ground truth."""

    def __init__(
        self, pipeline: ColorCorrectionPipeline, loader: ImageLoader, margin_deg: float
    ) -> None:
        self._pipeline = pipeline
        self._loader = loader
        self._margin = margin_deg

    def prepare(self, sample: BenchmarkSample) -> tuple[LinearImage, PreviewImage, SceneContext]:
        """Load the image with the chart hidden and run zero-shot scene analysis."""
        image = hide_chart(self._loader.load(sample.image_path), sample.chart_mask_path)
        preview = self._pipeline.render_preview(image)
        return image, preview, self._pipeline.analyze_scene(preview, "neutral")

    def label(self, sample: BenchmarkSample) -> tuple[SceneContext, list[ReferenceLabel]]:
        """Labels for every kept reference in one image."""
        image, preview, context = self.prepare(sample)
        regions, patches = self._pipeline.measure_references(image, preview, context)
        error_with = self._error(image, patches, context, sample)
        labels = []
        for index, patch in enumerate(patches):
            if not patch.kept:
                continue
            error_without = self._error(image, _without(patches, index), context, sample)
            region = regions[patch.region_index]
            labels.append(
                ReferenceLabel(
                    image=sample.image_path.name,
                    region_index=patch.region_index,
                    label=patch.label,
                    box_xyxy=region.box_xyxy,
                    error_with_deg=error_with,
                    error_without_deg=error_without,
                    helpfulness=classify_effect(error_with, error_without, self._margin),
                )
            )
        return context, labels

    def _error(
        self,
        image: LinearImage,
        patches: tuple[PatchMeasurement, ...],
        context: SceneContext,
        sample: BenchmarkSample,
    ) -> float:
        estimate = self._pipeline.estimate_illuminant(image, patches, context)
        return recovery_angular_error(estimate.white_rgb, sample.truth_rgb)


def supervised_target(context: SceneContext, labels: Iterable[ReferenceLabel]) -> dict[str, Any]:
    """Stage 3 JSON whose candidate usability follows the leave-one-out labels.

    A class is usable unless all of its labeled regions were harmful.
    """
    by_class: dict[ReferenceClass, list[Helpfulness]] = {}
    for item in labels:
        by_class.setdefault(item.label, []).append(item.helpfulness)
    candidates = []
    for label, effects in by_class.items():
        usable = any(effect is not Helpfulness.HARMFUL for effect in effects)
        original = context.candidate_for(label) or ReferenceCandidate(label)
        candidates.append(
            {
                "label": label.value,
                "description": original.description,
                "usable": usable,
                "confidence": round(effects.count(Helpfulness.HELPFUL) / len(effects), 3)
                if usable
                else 0.0,
                "concerns": list(original.concerns)
                + ([] if usable else ["disagrees with the other references"]),
            }
        )
    return {
        "environment": context.environment,
        "time_of_day": context.time_of_day,
        "weather": context.weather,
        "visible_light_sources": list(context.visible_light_sources),
        "illuminant_hypotheses": [to_jsonable(h) for h in context.illuminant_hypotheses],
        "reference_candidates": candidates,
        "user_intent": {"adaptation": "neutral", "strength": 1.0},
        "rationale": context.rationale or "Usability derived from leave-one-out angular error.",
    }


def write_supervised_records(
    path: Path, records: Iterable[tuple[BenchmarkSample, dict[str, Any]]]
) -> Path:
    """JSONL of {image, prompt, target} for supervised fine-tuning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for sample, target in records:
            record = {
                "image": str(sample.image_path),
                "prompt": SCENE_PROMPT.format(intent="neutral"),
                "target": json.dumps(target),
            }
            handle.write(json.dumps(record) + "\n")
    return path


def context_reward(
    completion: str,
    pipeline: ColorCorrectionPipeline,
    image: LinearImage,
    preview: PreviewImage,
    sample: BenchmarkSample,
) -> float:
    """GRPO reward: negative recovery angular error of the pipeline using the proposed context."""
    try:
        context = SceneAnalysisResponse.model_validate_json(completion).to_domain()
    except ValidationError:
        return INVALID_RESPONSE_REWARD
    _, patches = pipeline.measure_references(image, preview, context)
    estimate = pipeline.estimate_illuminant(image, patches, context)
    return -recovery_angular_error(estimate.white_rgb, sample.truth_rgb)
