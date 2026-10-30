"""The agent loop with scripted agents: best-round selection and overruling user matches."""

import dataclasses

import pytest

from natural_color_charts.config import load_config
from natural_color_charts.domain import Critique, ReferenceVerdict, UserColorAnchor
from natural_color_charts.pipeline_factory import build_evidence_builder, build_pipeline
from natural_color_charts.review.loop import AgenticCorrectionLoop, review_summary
from tests.pipeline.test_user_anchors import NAMES, _default_context, _scene

GRAYS = ["neutral 8 (.23 D)", "neutral 5 (.70 D)", "neutral 6.5 (.44 D)"]


def _critique(severity, acceptable=False, verdicts=()):
    cast = "warm" if severity else "none"
    return Critique("looks fine", tuple(verdicts), cast, severity, (), acceptable)


class ScriptedCritic:
    def __init__(self, critiques):
        self.critiques = list(critiques)
        self.calls = []

    def critique(self, sheet_png, evidence_text, intent_text):
        assert sheet_png.startswith(b"\x89PNG")
        self.calls.append(evidence_text)
        return self.critiques.pop(0)


class ScriptedReviser:
    """Drops the user's third match; records which context each revision started from."""

    def __init__(self, fail=False):
        self.fail = fail
        self.started_from = []

    def revise(self, preview, intent_text, previous, feedback, match_count):
        assert "Independent reviewer" in feedback and match_count == 3
        self.started_from.append(previous)
        if self.fail:
            return None
        revised = dataclasses.replace(previous, rationale=f"revision {len(self.started_from)}")
        return revised, {2: 0.0}


@pytest.fixture(scope="module")
def scene():
    config = load_config(overrides={"correction": {"raster_method": "spectral"}})
    pipe = build_pipeline(config)
    image, boxes = _scene(GRAYS)
    preview = pipe.render_preview(image)
    claimed = [GRAYS[0], GRAYS[1], "light skin"]
    anchors = [UserColorAnchor(b, NAMES.index(n)) for b, n in zip(boxes, claimed, strict=True)]
    return config, pipe, image, preview, _default_context(config, preview), anchors


def _loop(scene, critic, reviser, max_rounds=3):
    config, pipe = scene[0], scene[1]
    return AgenticCorrectionLoop(
        pipe, reviser, critic, build_evidence_builder(config), max_rounds, 0.15, 640
    )


def _run(scene, loop):
    _, _, image, preview, context, anchors = scene
    return loop.run(image, preview, context, anchors)


def test_an_accepted_first_correction_ends_the_loop(scene):
    reviser = ScriptedReviser()
    outcome = _run(scene, _loop(scene, ScriptedCritic([_critique(0.05, True)]), reviser))
    assert len(outcome.rounds) == 1 and not reviser.started_from
    assert outcome.stop_reason == "the critic accepted the correction"


def test_the_agent_can_ignore_a_wrong_user_match(scene):
    critic = ScriptedCritic([_critique(0.6), _critique(0.1, True)])
    outcome = _run(scene, _loop(scene, critic, ScriptedReviser()))
    assert len(outcome.rounds) == 2 and outcome.best_index == 1
    dropped = [p for p in outcome.result.patches if p.source == "user"][2]
    assert dropped.weight == 0.0
    assert [a.trust for a in outcome.best.anchors] == [1.0, 1.0, 0.0]
    assert "Your match 3: trusted 0% by the agent" in review_summary(outcome)
    assert "U3" in critic.calls[0]


def test_a_revision_that_does_not_improve_ends_the_loop_with_the_best(scene):
    reviser = ScriptedReviser()
    outcome = _run(scene, _loop(scene, ScriptedCritic([_critique(0.3), _critique(0.6)]), reviser))
    assert len(outcome.rounds) == 2 and outcome.best_index == 0
    assert len(reviser.started_from) == 1
    assert outcome.stop_reason == "the last revision did not improve on round 1"


def test_an_equally_rated_revision_counts_as_no_improvement(scene):
    critic = ScriptedCritic([_critique(0.3), _critique(0.08), _critique(0.08)])
    outcome = _run(scene, _loop(scene, critic, ScriptedReviser(), max_rounds=5))
    assert len(outcome.rounds) == 3 and outcome.best_index == 1


def test_a_doubtful_reference_keeps_the_agents_working(scene):
    doubt = ReferenceVerdict("U3", "doubtful", "gray patch matched to skin")
    critic = ScriptedCritic([_critique(0.05, True, [doubt]), _critique(0.04, True)])
    reviser = ScriptedReviser()
    outcome = _run(scene, _loop(scene, critic, reviser))
    assert len(outcome.rounds) == 2 and len(reviser.started_from) == 1


def test_a_failed_revision_delivers_the_best_round(scene):
    outcome = _run(scene, _loop(scene, ScriptedCritic([_critique(0.4)]), ScriptedReviser(True)))
    assert len(outcome.rounds) == 1 and outcome.best_index == 0


def test_without_a_critique_the_first_correction_is_delivered(scene):
    outcome = _run(scene, _loop(scene, ScriptedCritic([None]), ScriptedReviser()))
    assert len(outcome.rounds) == 1
    assert review_summary(outcome)[0] == "Round 1: reviewer unavailable (delivered)"
    assert review_summary(outcome)[-1] == "Stopped because the critic was unavailable."
