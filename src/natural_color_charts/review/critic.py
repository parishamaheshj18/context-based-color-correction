"""The critic: an independent VLM that judges a corrected image from the review sheet.

It never sees the scene analysis's rationale, so it cannot simply agree with it.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from natural_color_charts.context.analyzer import request_validated
from natural_color_charts.context.clients import VisionLanguageClient
from natural_color_charts.domain import Critique, ReferenceVerdict

CRITIC_PROMPT = """\
You are reviewing an automatic color correction. The image shows the photo as shot (left) and \
after correction (right). Numbered boxes on the right mark the references the correction used: \
A<n> were chosen by an AI, U<n> are color matches made by the user; red boxes were not used. \
Below, each swatch pair compares a reference after correction (left square) with the true \
color of its material under neutral daylight (right square), at equal lightness.

The user asked for: {intent}. If they asked to keep a mood, a slight tint in the image is \
intended; the measurements below always use a fully neutral correction.

{evidence}

Judge the correction, not the scene. Write observations first: do whites, grays, skin, and \
foliage look natural; is a cast left, and where; does any reference disagree with the others \
or with its material. Then give every reference a verdict: "consistent", "doubtful", or \
"wrong" (not that material, lit by a different light, or a mistaken user match), with a short \
reason. Give remaining_cast and cast_severity from 0 (perfect) to 1 (badly off). In \
artifacts, list every visible problem the correction introduced that the as-shot photo does \
not have, and where: blown or blotchy highlights, a sky turned white or cyan, banding, hue \
shifts in bright areas; [] if none. In suggestions, name concrete fixes: missed materials \
that could serve as references (say where), references to drop, or a different light source. \
Set acceptable to true only if a careful photo editor would leave this correction alone; any \
artifact rules that out. Be critical: do not assume the correction is right."""

CastName = Literal["none", "warm", "cool", "green", "magenta", "mixed"]
VerdictName = Literal["consistent", "doubtful", "wrong"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerdictResponse(_Strict):
    id: str
    verdict: VerdictName
    reason: str


class CritiqueResponse(_Strict):
    """Exactly the JSON the critic must return; observations come first so it reasons first."""

    observations: str
    references: list[VerdictResponse]
    remaining_cast: CastName
    cast_severity: float = Field(ge=0.0, le=1.0)
    artifacts: list[str]
    suggestions: list[str]
    acceptable: bool

    def to_domain(self) -> Critique:
        return Critique(
            observations=self.observations,
            verdicts=tuple(ReferenceVerdict(r.id, r.verdict, r.reason) for r in self.references),
            remaining_cast=self.remaining_cast,
            cast_severity=self.cast_severity,
            suggestions=tuple(self.suggestions),
            acceptable=self.acceptable,
            artifacts=tuple(self.artifacts),
        )


def critique_json_schema() -> dict[str, Any]:
    """Structured-output schema for CritiqueResponse; bounds are checked by Pydantic."""

    def strict(properties: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    string = {"type": "string"}
    verdict = strict(
        {
            "id": string,
            "verdict": {"type": "string", "enum": list(get_args(VerdictName))},
            "reason": string,
        }
    )
    return strict(
        {
            "observations": string,
            "references": {"type": "array", "items": verdict},
            "remaining_cast": {"type": "string", "enum": list(get_args(CastName))},
            "cast_severity": {"type": "number"},
            "artifacts": {"type": "array", "items": string},
            "suggestions": {"type": "array", "items": string},
            "acceptable": {"type": "boolean"},
        }
    )


def describe_critique(critique: Critique) -> str:
    """The critique as text for the scene analysis's revision."""
    lines = [
        f"Independent reviewer: remaining cast {critique.remaining_cast}, severity "
        f"{critique.cast_severity:.2f} (0 perfect, 1 badly off), "
        f"{'acceptable' if critique.acceptable else 'not acceptable'}.",
        f"Observations: {critique.observations}",
        *(f"{v.reference_id}: {v.verdict} ({v.reason})" for v in critique.verdicts),
        *(f"Artifact: {a}" for a in critique.artifacts),
        *(f"Suggestion: {s}" for s in critique.suggestions),
    ]
    return "\n".join(lines)


class VlmCritic:
    """Asks a VLM to judge a correction from its review sheet and measurements."""

    def __init__(self, client: VisionLanguageClient, max_retries: int) -> None:
        self._client = client
        self._max_retries = max_retries

    def critique(
        self, sheet_png: bytes, evidence_text: str, intent_text: str | None
    ) -> Critique | None:
        """The critic's judgment, or None if the VLM fails."""
        prompt = CRITIC_PROMPT.format(
            intent=intent_text or "(no request: accurate, neutral colors)",
            evidence=evidence_text,
        )
        response = request_validated(
            self._client,
            sheet_png,
            prompt,
            critique_json_schema(),
            CritiqueResponse,
            self._max_retries,
        )
        return response.to_domain() if response else None
