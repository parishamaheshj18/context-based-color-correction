import json

import numpy as np
import pytest

from natural_color_charts.context.analyzer import VlmSceneContextAnalyzer
from natural_color_charts.context.base import DefaultSceneContextProvider, KeywordIntentParser
from natural_color_charts.context.clients import VlmError
from natural_color_charts.context.schema import SceneAnalysisResponse, response_json_schema
from natural_color_charts.domain import AdaptationMode, ReferenceClass

VALID = {
    "environment": "outdoor",
    "time_of_day": "late_afternoon",
    "weather": "clear",
    "visible_light_sources": ["sun"],
    "illuminant_hypotheses": [
        {"family": "daylight", "cct_range_k": [5500, 4000], "confidence": 0.7}
    ],
    "reference_candidates": [
        {
            "label": "foliage",
            "description": "sunlit lawn",
            "usable": True,
            "confidence": 0.8,
            "concerns": [],
        },
        {
            "label": "skin",
            "description": "face near a red neon sign",
            "usable": False,
            "confidence": 0.2,
            "concerns": ["colored local light"],
        },
    ],
    "user_intent": {"adaptation": "preserve_ambiance", "strength": 0.6},
    "rationale": "Low sun, long shadows.",
}


class _ScriptedClient:
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete_json(self, image_png, prompt, schema):
        assert image_png.startswith(b"\x89PNG")
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def fallback(config):
    return DefaultSceneContextProvider(
        ["daylight"], config.estimation, 0.5, KeywordIntentParser(0.6)
    )


PREVIEW = np.zeros((40, 60, 3), np.uint8)


def test_valid_response_becomes_scene_context(fallback):
    client = _ScriptedClient([json.dumps(VALID)])
    context = VlmSceneContextAnalyzer(client, fallback, 2, 512).analyze(PREVIEW, "keep the mood")
    assert context.source == "vlm" and context.environment == "outdoor"
    assert context.illuminant_hypotheses[0].cct_range_k == (4000, 5500)
    assert context.candidate_for(ReferenceClass.SKIN).usable is False
    assert context.user_intent.adaptation is AdaptationMode.PRESERVE_AMBIANCE
    assert "keep the mood" in client.prompts[0]


def test_invalid_response_is_retried_with_feedback(fallback):
    bad = dict(VALID, user_intent={"adaptation": "neutral", "strength": 3})
    client = _ScriptedClient([json.dumps(bad), json.dumps(VALID)])
    context = VlmSceneContextAnalyzer(client, fallback, 2, 512).analyze(PREVIEW, None)
    assert context.source == "vlm"
    assert "invalid" in client.prompts[1] and "strength" in client.prompts[1]


def test_falls_back_to_default_context_after_retries(fallback):
    client = _ScriptedClient([VlmError("offline"), "not json", VlmError("offline")])
    context = VlmSceneContextAnalyzer(client, fallback, 2, 512).analyze(PREVIEW, "as shot please")
    assert context.source == "default" and "scripted unavailable" in context.rationale
    assert context.user_intent.adaptation is AdaptationMode.AS_SHOT


def test_schema_requires_every_property():
    def check(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                check(child)
        if node.get("type") == "array":
            check(node["items"])

    check(response_json_schema())
    assert set(response_json_schema()["properties"]) == set(SceneAnalysisResponse.model_fields)


def test_cached_client_answers_once_then_from_disk(tmp_path):
    from natural_color_charts.context.clients import CachedVisionLanguageClient

    inner = _ScriptedClient([json.dumps(VALID)])
    cached = CachedVisionLanguageClient(inner, "model-x", tmp_path)
    image = b"\x89PNG fake"
    first = cached.complete_json(image, "prompt", {"type": "object"})
    second = cached.complete_json(image, "prompt", {"type": "object"})
    assert first == second and inner.replies == []
    offline = CachedVisionLanguageClient(None, "model-x", tmp_path)
    assert offline.complete_json(image, "prompt", {"type": "object"}) == first
    with pytest.raises(VlmError):
        offline.complete_json(image, "other prompt", {"type": "object"})
