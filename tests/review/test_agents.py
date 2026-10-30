"""The two VLM agents' requests and responses: the critic and the scene analysis revision."""

import json

import pytest

from natural_color_charts.context.analyzer import VlmSceneContextAnalyzer
from natural_color_charts.context.base import DefaultSceneContextProvider, KeywordIntentParser
from natural_color_charts.context.schema import (
    SceneAnalysisResponse,
    revision_json_schema,
)
from natural_color_charts.review.critic import VlmCritic, critique_json_schema
from tests.context.test_vlm_analyzer import PREVIEW, VALID, _ScriptedClient

CRITIQUE = {
    "observations": "The wall still reads warm.",
    "references": [{"id": "U1", "verdict": "wrong", "reason": "cream, not white"}],
    "remaining_cast": "warm",
    "cast_severity": 0.4,
    "suggestions": ["use the gray pavement at the bottom"],
    "acceptable": False,
}


@pytest.fixture
def analyzer(config):
    fallback = DefaultSceneContextProvider(
        ["daylight"], config.estimation, 0.5, KeywordIntentParser(0.6)
    )

    def make(replies):
        client = _ScriptedClient(replies)
        return VlmSceneContextAnalyzer(client, fallback, 1, 512), client

    return make


def test_previous_answer_round_trips_through_the_domain():
    context = SceneAnalysisResponse.model_validate(VALID).to_domain()
    again = SceneAnalysisResponse.from_domain(context).to_domain()
    assert again == context


def test_revision_shows_the_previous_answer_and_feedback(analyzer):
    reply = dict(
        VALID,
        user_match_decisions=[
            {"match": 1, "trust": 0.0, "reason": "cream wall"},
            {"match": 9, "trust": 1.0, "reason": "no such match"},
        ],
    )
    agent, client = analyzer([json.dumps(reply)])
    previous = SceneAnalysisResponse.model_validate(VALID).to_domain()
    context, trust = agent.revise(PREVIEW, None, previous, "FEEDBACK-TEXT", 2)
    assert trust == {0: 0.0}  # decisions for matches that do not exist are dropped
    assert context.environment == "outdoor"
    prompt = client.prompts[0]
    assert "FEEDBACK-TEXT" in prompt and '"sunlit lawn"' in prompt and "U1 to U2" in prompt
    assert "user_match_decisions" in revision_json_schema()["required"]


def test_revision_failure_returns_none(analyzer):
    agent, _ = analyzer(["not json", "still not json"])
    previous = SceneAnalysisResponse.model_validate(VALID).to_domain()
    assert agent.revise(PREVIEW, None, previous, "", 0) is None


def test_critic_returns_a_structured_critique():
    client = _ScriptedClient([json.dumps(CRITIQUE)])
    critique = VlmCritic(client, 1).critique(b"\x89PNG...", "EVIDENCE", "keep the mood")
    assert critique.cast_severity == 0.4 and not critique.acceptable
    assert critique.verdicts[0].reference_id == "U1" and critique.verdicts[0].verdict == "wrong"
    assert "EVIDENCE" in client.prompts[0] and "keep the mood" in client.prompts[0]
    assert next(iter(critique_json_schema()["properties"])) == "observations"


def test_critic_retries_an_out_of_range_severity():
    bad = dict(CRITIQUE, cast_severity=3)
    client = _ScriptedClient([json.dumps(bad), json.dumps(CRITIQUE)])
    assert VlmCritic(client, 1).critique(b"\x89PNG", "", None).cast_severity == 0.4
    assert "cast_severity" in client.prompts[1]
