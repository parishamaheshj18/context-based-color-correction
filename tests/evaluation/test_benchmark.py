import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from natural_color_charts.camera.repository import srgb_equivalent_sensitivity
from natural_color_charts.cli import app
from natural_color_charts.config import load_config
from natural_color_charts.domain import ReferenceClass, WavelengthGrid
from natural_color_charts.evaluation.metrics import (
    reproduction_angular_error,
    summarize_errors,
)
from natural_color_charts.evaluation.runner import BenchmarkRunner, summarize_results, write_results
from natural_color_charts.evaluation.synthetic import (
    SyntheticBenchmarkSpec,
    sample_prior_reflectances,
    write_synthetic_benchmark,
)
from natural_color_charts.spectral.bootstrap_priors import build_bootstrap_priors
from natural_color_charts.spectral.datasets import chart_reflectances
from natural_color_charts.spectral.illuminants import BlackbodyFamily, DaylightFamily

GRID = WavelengthGrid(400, 700, 10)
C = ReferenceClass


def test_summary_statistics():
    stats = summarize_errors([1, 2, 3, 4, 5, 6, 7, 8])
    assert stats["mean"] == 4.5 and stats["median"] == 4.5
    assert stats["best25"] == 1.5 and stats["worst25"] == 7.5
    assert reproduction_angular_error(np.array([1, 2, 3]), np.array([2, 4, 6])) < 1e-6


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    config = load_config()
    priors = build_bootstrap_priors(GRID, config.priors)
    rng = np.random.default_rng(7)
    names, chart = chart_reflectances("BabelColor Average", GRID)
    backgrounds = [chart[names.index(n)] for n in ("moderate red", "yellow green", "purplish blue")]
    illuminants = [
        BlackbodyFamily(GRID, 1800, 6500).spd(2900),
        DaylightFamily(GRID, 4000, 25000).spd(6500),
        DaylightFamily(GRID, 4000, 25000).spd(4500),
    ]

    def sample(label):
        p = priors[label]
        return label, sample_prior_reflectances(p.mean_n, p.basis_nxk, p.variances_k, 1, rng)[0]

    spec = SyntheticBenchmarkSpec(
        sensitivity_3xn=srgb_equivalent_sensitivity(GRID),
        illuminants=illuminants,
        reference_reflectances=[[sample(C.CONCRETE), sample(C.PAPER), sample(C.FOLIAGE)]] * 3,
        background_reflectances=backgrounds,
        chart_reflectances_kxn=chart,
    )
    return write_synthetic_benchmark(tmp_path_factory.mktemp("bench"), spec, rng)


def test_benchmark_runner_compares_pipeline_baselines_and_upper_bound(
    dataset: Path, tmp_path: Path
):
    config = load_config(overrides={"detection": {"detector": "manual"}})
    results = BenchmarkRunner(config).run(dataset, ["full", "no_reliability"], ["gray_world"])
    summary = summarize_results(results)
    assert set(summary) == {
        "pipeline:full",
        "pipeline:no_reliability",
        "baseline:gray_world",
        "chart_fit_upper_bound",
    }
    assert summary["pipeline:full"]["recovery_deg"]["mean"] < 2.0
    assert summary["baseline:gray_world"]["recovery_deg"]["mean"] > 5.0
    assert (
        summary["chart_fit_upper_bound"]["delta_e00_mean"]
        < summary["baseline:gray_world"]["delta_e00_mean"]
    )
    assert summary["pipeline:full"]["fallback_rate"] == 0.0
    rows, summary_path = write_results(tmp_path, results)
    assert rows.read_text().count("\n") == len(results) + 1
    assert json.loads(summary_path.read_text())["pipeline:full"]["recovery_deg"]["count"] == 3


def test_chart_is_hidden_from_the_pipeline(dataset: Path):
    config = load_config(overrides={"detection": {"detector": "none"}})
    results = BenchmarkRunner(config).run(dataset, ["full"], [])
    estimators = {r.estimator for r in results if r.method == "pipeline:full"}
    assert estimators == {"shades_of_gray"}  # no references and no chart to lean on


def test_correct_command_writes_report_overlay_and_comparison(dataset: Path, tmp_path: Path):
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "correct",
            str(dataset / "scene_000.tif"),
            "--detector",
            "manual",
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((out / "report.json").read_text())
    assert report["illuminant"]["method"] == "spectral"
    assert report["illuminant"]["family"] == "blackbody"
    assert abs(report["illuminant"]["cct_k"] - 2900) < 300
    assert {r["label"] for r in report["references"] if r["used_by_estimator"]} == {
        "concrete",
        "paper",
        "foliage",
    }
    for name in ("overlay.png", "comparison.png", "corrected.tif"):
        assert (out / name).stat().st_size > 0
