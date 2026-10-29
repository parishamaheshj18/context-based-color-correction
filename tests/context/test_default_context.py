import pytest

from natural_color_charts.context.base import DefaultSceneContextProvider, KeywordIntentParser
from natural_color_charts.domain import AdaptationMode, ReferenceClass


@pytest.mark.parametrize(
    ("text", "mode"),
    [
        (None, AdaptationMode.NEUTRAL),
        ("neutral product colors", AdaptationMode.NEUTRAL),
        ("keep the warm evening mood", AdaptationMode.PRESERVE_AMBIANCE),
        ("leave it as shot", AdaptationMode.AS_SHOT),
    ],
)
def test_keyword_intent_parser(text, mode):
    assert KeywordIntentParser(0.6).parse(text).adaptation is mode


def test_default_provider_lists_all_classes(config):
    provider = DefaultSceneContextProvider(
        ["daylight", "blackbody"], config.estimation, 0.5, KeywordIntentParser(0.6)
    )
    context = provider.analyze(None, "keep the mood")
    assert {c.label for c in context.reference_candidates} == set(ReferenceClass)
    assert [h.family for h in context.illuminant_hypotheses] == ["daylight", "blackbody"]
    assert context.user_intent.strength == 0.6
