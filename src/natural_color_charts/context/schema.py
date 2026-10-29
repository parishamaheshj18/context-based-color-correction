"""Structured VLM response for scene analysis, validated with Pydantic."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from natural_color_charts.domain import (
    SCENE_CLASSES,
    AdaptationMode,
    BoxXyxy,
    IlluminantHypothesis,
    ReferenceCandidate,
    ReferenceClass,
    SceneContext,
    UserIntent,
)

FamilyName = Literal["daylight", "blackbody"]
MAX_BOXES_PER_CANDIDATE = 3
MIN_BOX_SIDE = 0.01


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HypothesisResponse(_Strict):
    family: FamilyName
    cct_range_k: tuple[float, float]
    confidence: float = Field(ge=0.0, le=1.0)


class CandidateResponse(_Strict):
    label: ReferenceClass

    @field_validator("label")
    @classmethod
    def _scene_class_only(cls, label: ReferenceClass) -> ReferenceClass:
        if label not in SCENE_CLASSES:
            raise ValueError("known_color is reserved for user matches")
        return label

    description: str
    usable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    concerns: list[str]
    boxes: list[list[float]] = Field(default_factory=list)

    def clean_boxes(self) -> tuple[BoxXyxy, ...]:
        """Well-formed boxes clipped to the image; malformed or tiny ones are dropped."""
        cleaned = []
        for box in self.boxes:
            if len(box) != 4:
                continue
            x0, x1 = sorted(min(max(v, 0.0), 1.0) for v in (box[0], box[2]))
            y0, y1 = sorted(min(max(v, 0.0), 1.0) for v in (box[1], box[3]))
            if x1 - x0 >= MIN_BOX_SIDE and y1 - y0 >= MIN_BOX_SIDE:
                cleaned.append((x0, y0, x1, y1))
        return tuple(cleaned[:MAX_BOXES_PER_CANDIDATE])


class IntentResponse(_Strict):
    adaptation: AdaptationMode
    strength: float = Field(ge=0.0, le=1.0)


class SceneAnalysisResponse(_Strict):
    """Exactly the JSON the VLM must return."""

    environment: Literal["indoor", "outdoor", "mixed", "unknown"]
    time_of_day: str
    weather: str
    visible_light_sources: list[str]
    illuminant_hypotheses: list[HypothesisResponse]
    reference_candidates: list[CandidateResponse]
    user_intent: IntentResponse
    rationale: str

    def to_domain(self) -> SceneContext:
        """Convert to the immutable domain object, sorting CCT ranges."""
        return SceneContext(
            environment=self.environment,
            time_of_day=self.time_of_day,
            weather=self.weather,
            visible_light_sources=tuple(self.visible_light_sources),
            illuminant_hypotheses=tuple(
                IlluminantHypothesis(h.family, _sorted_range(h.cct_range_k), h.confidence)
                for h in self.illuminant_hypotheses
            ),
            reference_candidates=tuple(
                ReferenceCandidate(
                    c.label,
                    c.description,
                    c.usable,
                    c.confidence,
                    tuple(c.concerns),
                    c.clean_boxes(),
                )
                for c in self.reference_candidates
            ),
            user_intent=UserIntent(self.user_intent.adaptation, self.user_intent.strength),
            source="vlm",
            rationale=self.rationale,
        )


def _sorted_range(values: tuple[float, float]) -> tuple[float, float]:
    low, high = sorted(values)
    return low, high


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def response_json_schema() -> dict[str, Any]:
    """Hand-written JSON schema in the subset accepted by structured-output decoders.

    Numeric bounds are enforced afterwards by Pydantic, since decoders do not all support them.
    """
    number, string = {"type": "number"}, {"type": "string"}
    strings = {"type": "array", "items": string}
    return _object(
        {
            "environment": {"type": "string", "enum": ["indoor", "outdoor", "mixed", "unknown"]},
            "time_of_day": string,
            "weather": string,
            "visible_light_sources": strings,
            "illuminant_hypotheses": {
                "type": "array",
                "items": _object(
                    {
                        "family": {"type": "string", "enum": ["daylight", "blackbody"]},
                        "cct_range_k": {"type": "array", "items": number},
                        "confidence": number,
                    }
                ),
            },
            "reference_candidates": {
                "type": "array",
                "items": _object(
                    {
                        "label": {"type": "string", "enum": [c.value for c in SCENE_CLASSES]},
                        "description": string,
                        "usable": {"type": "boolean"},
                        "confidence": number,
                        "concerns": strings,
                        "boxes": {"type": "array", "items": {"type": "array", "items": number}},
                    }
                ),
            },
            "user_intent": _object(
                {
                    "adaptation": {"type": "string", "enum": [m.value for m in AdaptationMode]},
                    "strength": number,
                }
            ),
            "rationale": string,
        }
    )
