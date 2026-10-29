"""Structured VLM response for scene analysis, validated with Pydantic."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from natural_color_charts.domain import (
    AdaptationMode,
    IlluminantHypothesis,
    ReferenceCandidate,
    ReferenceClass,
    SceneContext,
    UserIntent,
)

FamilyName = Literal["daylight", "blackbody"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HypothesisResponse(_Strict):
    family: FamilyName
    cct_range_k: tuple[float, float]
    confidence: float = Field(ge=0.0, le=1.0)


class CandidateResponse(_Strict):
    label: ReferenceClass
    description: str
    usable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    concerns: list[str]


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
                    c.label, c.description, c.usable, c.confidence, tuple(c.concerns)
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
                        "label": {"type": "string", "enum": [c.value for c in ReferenceClass]},
                        "description": string,
                        "usable": {"type": "boolean"},
                        "confidence": number,
                        "concerns": strings,
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
