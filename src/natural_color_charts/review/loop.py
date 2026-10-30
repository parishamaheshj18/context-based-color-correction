"""The agent loop: correct, let an independent critic judge, let the scene analysis revise.

After the user's matches the agents take over. The scene analysis may re-box or drop its own
references and lower the trust of (or ignore) the user's matches; the physics still computes
every pixel. The loop ends when the critic accepts with nothing in doubt, when a revision
does not improve the critic's rating (the best round is delivered), or when rounds run out.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from natural_color_charts.context.analyzer import encode_png
from natural_color_charts.domain import (
    Critique,
    LinearImage,
    PreviewImage,
    ReferenceEvidence,
    SceneContext,
    UserColorAnchor,
)
from natural_color_charts.pipeline import ColorCorrectionPipeline, PipelineResult
from natural_color_charts.review.critic import describe_critique
from natural_color_charts.review.evidence import (
    ReviewEvidenceBuilder,
    describe_evidence,
    render_review_sheet,
)

logger = logging.getLogger(__name__)


class SceneReviser(Protocol):
    """Revises a scene context, and its trust in the user's matches, from review feedback."""

    def revise(
        self,
        preview: PreviewImage,
        intent_text: str | None,
        previous: SceneContext,
        feedback: str,
        match_count: int,
    ) -> tuple[SceneContext, dict[int, float]] | None: ...


class CorrectionCritic(Protocol):
    """Judges a corrected image from its review sheet and measurements."""

    def critique(
        self, sheet_png: bytes, evidence_text: str, intent_text: str | None
    ) -> Critique | None: ...


@dataclass(frozen=True, eq=False)
class ReviewRound:
    """One correction, its inputs, and the critic's judgment of it."""

    result: PipelineResult
    context: SceneContext
    anchors: tuple[UserColorAnchor, ...]
    evidence: tuple[ReferenceEvidence, ...]
    sheet: NDArray[np.uint8]
    critique: Critique | None

    @property
    def severity(self) -> float:
        """The critic's remaining-cast severity; unjudged rounds rank last."""
        return self.critique.cast_severity if self.critique else float("inf")


@dataclass(frozen=True, eq=False)
class ReviewOutcome:
    """All rounds, which one is delivered, and why the loop stopped."""

    rounds: tuple[ReviewRound, ...]
    best_index: int
    stop_reason: str = ""

    @property
    def best(self) -> ReviewRound:
        return self.rounds[self.best_index]

    @property
    def result(self) -> PipelineResult:
        return self.best.result


class AgenticCorrectionLoop:
    """Alternates correction, critique, and revision for at most max_rounds critiques."""

    def __init__(
        self,
        pipeline: ColorCorrectionPipeline,
        reviser: SceneReviser,
        critic: CorrectionCritic,
        evidence_builder: ReviewEvidenceBuilder,
        max_rounds: int,
        acceptable_severity: float,
        sheet_width_px: int,
    ) -> None:
        self._pipeline = pipeline
        self._reviser = reviser
        self._critic = critic
        self._evidence = evidence_builder
        self._max_rounds = max(1, max_rounds)
        self._acceptable = acceptable_severity
        self._sheet_px = sheet_width_px

    def run(
        self,
        image: LinearImage,
        preview: PreviewImage,
        context: SceneContext,
        anchors: Sequence[UserColorAnchor] = (),
        intent: str | None = None,
        camera: str | None = None,
    ) -> ReviewOutcome:
        """Correct, then review and revise until the critic accepts or rounds run out."""
        rounds: list[ReviewRound] = []
        matches = tuple(anchors)
        while True:
            result = self._pipeline.run_with_context(image, preview, context, camera, matches)
            rounds.append(self._review(result, context, matches, intent))
            best = min(range(len(rounds)), key=lambda i: (rounds[i].severity, i))
            logger.info("Review round %d: severity %.2f", len(rounds), rounds[-1].severity)
            reason = self._stop_reason(rounds, best)
            if reason:
                return ReviewOutcome(tuple(rounds), best, reason)
            revision = self._reviser.revise(
                preview, intent, context, revision_feedback(rounds), len(matches)
            )
            if revision is None:
                return ReviewOutcome(tuple(rounds), best, "the revision step failed")
            context, trust = revision
            matches = tuple(
                dataclasses.replace(a, trust=trust.get(i, a.trust)) for i, a in enumerate(matches)
            )

    def _review(
        self,
        result: PipelineResult,
        context: SceneContext,
        anchors: tuple[UserColorAnchor, ...],
        intent: str | None,
    ) -> ReviewRound:
        evidence = self._evidence.build(result)
        sheet = render_review_sheet(result, evidence, self._sheet_px)
        critique = self._critic.critique(
            encode_png(sheet, max(sheet.shape[:2])), describe_evidence(evidence), intent
        )
        return ReviewRound(result, context, anchors, evidence, sheet, critique)

    def _stop_reason(self, rounds: Sequence[ReviewRound], best: int) -> str | None:
        """Why the loop should stop now, or None to revise again."""
        critique = rounds[-1].critique
        if critique is None:
            return "the critic was unavailable"
        # A revision that does not beat the best rating so far is unlikely to be followed by one
        # that does, and the critic's doubts about it would only steer from a worse start.
        if best != len(rounds) - 1:
            return f"the last revision did not improve on round {best + 1}"
        in_doubt = any(v.verdict != "consistent" for v in critique.verdicts)
        if critique.acceptable and critique.cast_severity <= self._acceptable and not in_doubt:
            return "the critic accepted the correction"
        if len(rounds) >= self._max_rounds:
            return "the round limit was reached"
        return None


def revision_feedback(rounds: Sequence[ReviewRound]) -> str:
    """Measurements and critique of the latest round, plus how earlier rounds were rated."""
    round_ = rounds[-1]
    assert round_.critique is not None
    parts = [describe_evidence(round_.evidence), describe_critique(round_.critique)]
    if len(rounds) > 1:
        scores = ", ".join(f"round {i + 1}: {r.severity:.2f}" for i, r in enumerate(rounds))
        parts.append(f"Reviewer severity so far ({scores}); lower is better.")
    return "\n\n".join(parts)


def review_summary(outcome: ReviewOutcome) -> list[str]:
    """Plain-language lines: each round's verdict and which one was delivered."""
    lines = []
    for number, round_ in enumerate(outcome.rounds, start=1):
        critique = round_.critique
        mark = " (delivered)" if number - 1 == outcome.best_index else ""
        if critique is None:
            lines.append(f"Round {number}: reviewer unavailable{mark}")
        else:
            lines.append(
                f"Round {number}: {critique.remaining_cast} cast, severity "
                f"{critique.cast_severity:.2f}{mark}"
            )
    for index, anchor in enumerate(outcome.best.anchors, start=1):
        if anchor.trust < 1.0:
            lines.append(f"Your match {index}: trusted {anchor.trust:.0%} by the agent")
    if outcome.stop_reason:
        lines.append(f"Stopped because {outcome.stop_reason}.")
    return lines
