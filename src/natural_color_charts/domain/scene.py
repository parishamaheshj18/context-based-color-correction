"""Scene context produced by Stage 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ReferenceClass(StrEnum):
    """Materials with predictable reflectance that can act as chart patches."""

    FOLIAGE = "foliage"
    SKIN = "skin"
    CONCRETE = "concrete"
    PAPER = "paper"
    SNOW = "snow"
    CLOUD = "cloud"


NEUTRAL_CLASSES = frozenset(
    {ReferenceClass.CONCRETE, ReferenceClass.PAPER, ReferenceClass.SNOW, ReferenceClass.CLOUD}
)


class AdaptationMode(StrEnum):
    """How far the correction moves the image toward the target white."""

    NEUTRAL = "neutral"
    PRESERVE_AMBIANCE = "preserve_ambiance"
    AS_SHOT = "as_shot"


@dataclass(frozen=True)
class IlluminantHypothesis:
    """A prior over one illuminant family's correlated color temperature."""

    family: str
    cct_range_k: tuple[float, float]
    confidence: float


@dataclass(frozen=True)
class ReferenceCandidate:
    """A material the scene analysis believes is present, with its trustworthiness."""

    label: ReferenceClass
    description: str = ""
    usable: bool = True
    confidence: float = 0.5
    concerns: tuple[str, ...] = ()


@dataclass(frozen=True)
class UserIntent:
    """Requested adaptation mode; strength is the fraction of correction applied."""

    adaptation: AdaptationMode = AdaptationMode.NEUTRAL
    strength: float = 1.0


@dataclass(frozen=True)
class SceneContext:
    """Lighting priors, reference candidates, and intent for one image."""

    environment: str = "unknown"
    time_of_day: str = "unknown"
    weather: str = "unknown"
    visible_light_sources: tuple[str, ...] = ()
    illuminant_hypotheses: tuple[IlluminantHypothesis, ...] = ()
    reference_candidates: tuple[ReferenceCandidate, ...] = ()
    user_intent: UserIntent = field(default_factory=UserIntent)
    source: str = "default"
    rationale: str = ""

    def candidate_for(self, label: ReferenceClass) -> ReferenceCandidate | None:
        return next((c for c in self.reference_candidates if c.label == label), None)
