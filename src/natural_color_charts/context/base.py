"""Stage 3 interfaces and the model-free default context."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from natural_color_charts.config import EstimationConfig
from natural_color_charts.domain import (
    AdaptationMode,
    IlluminantHypothesis,
    PreviewImage,
    ReferenceCandidate,
    ReferenceClass,
    SceneContext,
    UserIntent,
)


class SceneContextProvider(Protocol):
    """Produces lighting priors, reference candidates, and parsed intent for an image."""

    def analyze(self, preview: PreviewImage, intent_text: str | None) -> SceneContext: ...


class IntentParser(Protocol):
    """Turns free-text intent into an adaptation mode and strength."""

    def parse(self, intent_text: str | None) -> UserIntent: ...


_AS_SHOT_PATTERN = re.compile(r"\b(as[- ]shot|original|unchanged|don'?t change|no correction)\b")
_AMBIANCE_PATTERN = re.compile(
    r"\b(keep|preserve|retain|mood|ambian\w*|atmospher\w*|warm|cozy|golden|sunset|evening|candle)"
)


class KeywordIntentParser:
    """Rule-based intent parsing used when no VLM is available."""

    def __init__(self, ambiance_strength: float) -> None:
        self._ambiance_strength = ambiance_strength

    def parse(self, intent_text: str | None) -> UserIntent:
        text = (intent_text or "").lower()
        if _AS_SHOT_PATTERN.search(text):
            return UserIntent(AdaptationMode.AS_SHOT, 0.0)
        if _AMBIANCE_PATTERN.search(text):
            return UserIntent(AdaptationMode.PRESERVE_AMBIANCE, self._ambiance_strength)
        return UserIntent(AdaptationMode.NEUTRAL, 1.0)


def default_hypotheses(
    families: Sequence[str], config: EstimationConfig
) -> tuple[IlluminantHypothesis, ...]:
    """Uninformative hypotheses spanning each family's full configured range."""
    return tuple(
        IlluminantHypothesis(
            family=name,
            cct_range_k=(config.families[name].cct_min_k, config.families[name].cct_max_k),
            confidence=0.0,
        )
        for name in families
    )


class DefaultSceneContextProvider:
    """Assumes every reference class may be present, with uninformative lighting priors."""

    def __init__(
        self,
        families: Sequence[str],
        estimation_config: EstimationConfig,
        candidate_confidence: float,
        intent_parser: IntentParser,
    ) -> None:
        self._hypotheses = default_hypotheses(families, estimation_config)
        self._candidate_confidence = candidate_confidence
        self._intent_parser = intent_parser

    def analyze(self, preview: PreviewImage, intent_text: str | None) -> SceneContext:
        candidates = tuple(
            ReferenceCandidate(label=label, confidence=self._candidate_confidence)
            for label in ReferenceClass
        )
        return SceneContext(
            illuminant_hypotheses=self._hypotheses,
            reference_candidates=candidates,
            user_intent=self._intent_parser.parse(intent_text),
            source="default",
        )
