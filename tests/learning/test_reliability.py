import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from natural_color_charts.camera.repository import srgb_equivalent_sensitivity
from natural_color_charts.cli import app
from natural_color_charts.config import load_config
from natural_color_charts.context.schema import response_json_schema
from natural_color_charts.domain import ReferenceClass, WavelengthGrid
from natural_color_charts.evaluation.dataset import load_samples
from natural_color_charts.evaluation.synthetic import (
    SyntheticBenchmarkSpec,
    sample_prior_reflectances,
    write_synthetic_benchmark,
)
from natural_color_charts.learning.reliability import (
    INVALID_RESPONSE_REWARD,
    Helpfulness,
    ReferenceLabeler,
    classify_effect,
    context_reward,
    supervised_target,
)
from natural_color_charts.pipeline_factory import build_loader, build_pipeline
from natural_color_charts.spectral.bootstrap_priors import build_bootstrap_priors
from natural_color_charts.spectral.datasets import chart_reflectances
from natural_color_charts.spectral.illuminants import BlackbodyFamily

GRID = WavelengthGrid(400, 700, 10)
C = ReferenceClass
# Leave-one-out rejection off, so the mislabeled region actually hurts the estimate.
OVERRIDES = {"detection": {"detector": "manual"}, "estimation": {"loo_min_references": 10_000}}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    config = load_config()
    priors = build_bootstrap_priors(GRID, config.priors)
    rng = np.random.default_rng(3)
    names, chart = chart_reflectances("BabelColor Average", GRID)

    def sample(label):
        p = priors[label]
        return label, sample_prior_reflectances(p.mean_n, p.basis_nxk, p.variances_k, 1, rng)[0]

    refs = [sample(C.CONCRETE), sample(C.SNOW), (C.PAPER, chart[names.index("blue sky")])]
    spec = SyntheticBenchmarkSpec(
        sensitivity_3xn=srgb_equivalent_sensitivity(GRID),
        illuminants=[BlackbodyFamily(GRID, 1800, 6500).spd(3300)],
        reference_reflectances=[refs],
        background_reflectances=[chart[names.index("neutral 5 (.70 D)")]],
        chart_reflectances_kxn=chart,
    )
    return write_synthetic_benchmark(tmp_path_factory.mktemp("rel"), spec, rng)


def test_classify_effect():
    assert classify_effect(1.0, 3.0, 0.1) is Helpfulness.HELPFUL
    assert classify_effect(3.0, 1.0, 0.1) is Helpfulness.HARMFUL
    assert classify_effect(1.0, 1.05, 0.1) is Helpfulness.NEUTRAL


def test_mislabeled_reference_is_labeled_harmful(dataset):
    config = load_config(overrides=OVERRIDES)
    labeler = ReferenceLabeler(build_pipeline(config), build_loader(config), 0.1)
    context, labels = labeler.label(load_samples(dataset)[0])
    effects = {item.label: item.helpfulness for item in labels}
    assert effects[C.PAPER] is Helpfulness.HARMFUL
    assert effects[C.CONCRETE] is not Helpfulness.HARMFUL
    target = supervised_target(context, labels)
    usable = {c["label"]: c["usable"] for c in target["reference_candidates"]}
    assert usable == {"concrete": True, "snow": True, "paper": False}
    assert set(target) == set(response_json_schema()["properties"])


def test_reward_prefers_contexts_that_drop_harmful_references(dataset):
    config = load_config(overrides=OVERRIDES)
    pipeline = build_pipeline(config)
    sample = load_samples(dataset)[0]
    labeler = ReferenceLabeler(pipeline, build_loader(config), 0.1)
    image, preview, context = labeler.prepare(sample)
    _, labels = labeler.label(sample)
    good = json.dumps(supervised_target(context, labels))
    trusting = json.loads(good)
    for candidate in trusting["reference_candidates"]:
        candidate.update(usable=True, confidence=1.0)
    bad = json.dumps(trusting)
    good_reward = context_reward(good, pipeline, image, preview, sample)
    assert good_reward > context_reward(bad, pipeline, image, preview, sample)
    assert good_reward > -3.0
    assert context_reward("{oops", pipeline, image, preview, sample) == INVALID_RESPONSE_REWARD


def test_label_references_command(dataset, tmp_path):
    result = CliRunner().invoke(
        app,
        ["label-references", str(dataset), "--detector", "manual", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    record = json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])
    assert record["image"].endswith(".tif") and "Reference materials" in record["prompt"]
    assert json.loads(record["target"])["user_intent"]["adaptation"] == "neutral"
