"""Command-line interface (`ncc`)."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from natural_color_charts.config import AppConfig, load_config, merge_nested
from natural_color_charts.evaluation.dataset import load_samples
from natural_color_charts.evaluation.runner import (
    BenchmarkRunner,
    format_summary_table,
    write_results,
)
from natural_color_charts.io.writers import write_image8, write_tiff16
from natural_color_charts.learning.reliability import (
    ReferenceLabeler,
    supervised_target,
    write_supervised_records,
)
from natural_color_charts.pipeline import PipelineResult
from natural_color_charts.pipeline_factory import (
    build_grid,
    build_loader,
    build_pipeline,
    build_repository,
)
from natural_color_charts.reporting.report import build_report, to_jsonable, write_report
from natural_color_charts.reporting.showcase import ShowcaseBuilder
from natural_color_charts.reporting.visualization import render_overlay, save_comparison
from natural_color_charts.spectral.library_reader import build_priors_from_library
from natural_color_charts.spectral.priors import save_prior

app = typer.Typer(help="Context-aware color correction without a color checker.")


@app.callback()
def main() -> None:
    """Context-aware color correction without a color checker."""
    load_dotenv(Path(".env"))


def load_dotenv(path: Path) -> None:
    """Set KEY=VALUE lines from a git-ignored .env file, never overriding the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


DEFAULT_BASELINES = ("gray_world", "white_patch", "shades_of_gray", "gray_edge", "as_shot")

SetOption = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override any config value, e.g. --set detection.device=cuda"),
]
ConfigOption = Annotated[Path | None, typer.Option("--config", help="YAML config file.")]


def _stage_overrides(
    vlm: str | None, detector: str | None, settings: list[str] | None
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for setting in settings or []:
        overrides = merge_nested(overrides, parse_setting(setting))
    if vlm:
        overrides = merge_nested(overrides, {"context": {"client": vlm}})
    if detector:
        overrides = merge_nested(overrides, {"detection": {"detector": detector}})
    return overrides


def parse_setting(setting: str) -> dict[str, Any]:
    """Turn "detection.device=cuda" into {"detection": {"device": "cuda"}}; values are YAML."""
    key, separator, raw_value = setting.partition("=")
    if not separator or not key:
        raise typer.BadParameter(f"Expected section.key=value, got '{setting}'")
    value: Any = yaml.safe_load(raw_value)
    for part in reversed(key.split(".")):
        value = {part: value}
    return dict(value)


def _load(
    config_path: Path | None, verbose: bool, overrides: dict[str, Any] | None = None
) -> AppConfig:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")
    return load_config(config_path, overrides)


@app.command()
def correct(
    photo: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path | None, typer.Option(help="Defaults to outputs/<stem>.")] = None,
    camera: Annotated[str | None, typer.Option(help="Force a camera's sensitivity curves.")] = None,
    intent: Annotated[str | None, typer.Option(help='E.g. "keep the warm evening mood".')] = None,
    vlm: Annotated[str | None, typer.Option(help="none | ollama | anthropic")] = None,
    detector: Annotated[str | None, typer.Option(help="none | manual | vlm | transformers")] = None,
    settings: SetOption = None,
    config_path: ConfigOption = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Correct the colors of PHOTO and write images and a report."""
    config = _load(config_path, verbose, _stage_overrides(vlm, detector, settings))
    result = build_pipeline(config).run(photo, camera=camera, intent=intent)
    target_dir = output_dir or config.output.directory / photo.stem
    written = write_outputs(result, target_dir, config)
    for path in written:
        typer.echo(str(path))


@app.command("list-cameras")
def list_cameras(config_path: ConfigOption = None) -> None:
    """List cameras with spectral sensitivity data."""
    for name in build_repository(_load(config_path, verbose=False)).list_cameras():
        typer.echo(name)


@app.command("build-priors")
def build_priors_command(
    library_dir: Annotated[Path | None, typer.Option(help="Class subfolders of spectra.")] = None,
    config_path: ConfigOption = None,
) -> None:
    """Build PCA reflectance priors from measured spectra (Stage 6)."""
    config = _load(config_path, verbose=True)
    grid = build_grid(config)
    source = library_dir or config.priors.library_dir
    for prior in build_priors_from_library(source, grid, config.priors):
        path = save_prior(config.priors.directory, prior)
        typer.echo(
            f"{prior.label.value}: {prior.sample_count} spectra, {prior.components} components, "
            f"{prior.explained_variance_ratio:.1%} variance explained -> {path}"
        )


@app.command()
def evaluate(
    dataset_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    ablation: Annotated[list[str] | None, typer.Option(help="Repeatable; see ABLATIONS.")] = None,
    baseline: Annotated[list[str] | None, typer.Option(help="Repeatable estimator name.")] = None,
    output_dir: Annotated[Path | None, typer.Option()] = None,
    vlm: Annotated[str | None, typer.Option(help="none | ollama | anthropic")] = None,
    detector: Annotated[str | None, typer.Option(help="none | manual | vlm | transformers")] = None,
    settings: SetOption = None,
    config_path: ConfigOption = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Benchmark the pipeline, ablations, and baselines on DATASET_DIR (Stage 11)."""
    config = _load(config_path, verbose, _stage_overrides(vlm, detector, settings))
    results = BenchmarkRunner(config).run(
        dataset_dir, ablation or ["full"], baseline or list(DEFAULT_BASELINES)
    )
    target = output_dir or config.output.directory / f"evaluation_{dataset_dir.name}"
    _, summary_path = write_results(target, results)
    typer.echo(format_summary_table(json.loads(summary_path.read_text())))
    typer.echo(f"\nDetails: {target}")


@app.command("label-references")
def label_references(
    dataset_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path | None, typer.Option()] = None,
    vlm: Annotated[str | None, typer.Option(help="none | ollama | anthropic")] = None,
    detector: Annotated[str | None, typer.Option(help="none | manual | vlm | transformers")] = None,
    settings: SetOption = None,
    config_path: ConfigOption = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Label references helpful/harmful and export fine-tuning records (Stage 12)."""
    config = _load(config_path, verbose, _stage_overrides(vlm, detector, settings))
    labeler = ReferenceLabeler(
        build_pipeline(config), build_loader(config), config.learning.helpfulness_margin_deg
    )
    target = output_dir or config.output.directory / f"labels_{dataset_dir.name}"
    target.mkdir(parents=True, exist_ok=True)
    records, all_labels = [], []
    for sample in load_samples(dataset_dir):
        context, labels = labeler.label(sample)
        all_labels.extend(labels)
        records.append((sample, supervised_target(context, labels)))
    (target / "labels.json").write_text(json.dumps(to_jsonable(all_labels), indent=2))
    write_supervised_records(target / "sft.jsonl", records)
    counts = Counter(label.helpfulness.value for label in all_labels)
    typer.echo(f"{len(all_labels)} references: {dict(counts)} -> {target}")


@app.command()
def showcase(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)] = Path(
        "showcase/cases.yaml"
    ),
    site_dir: Annotated[Path, typer.Option(help="Static site root.")] = Path("site"),
    case: Annotated[list[str] | None, typer.Option(help="Rebuild only these case ids.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Build the showcase site's images and data from a manifest of test photos."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")
    path = ShowcaseBuilder.from_manifest(manifest, site_dir).build(set(case) if case else None)
    typer.echo(f"Wrote {path}")


def write_outputs(result: PipelineResult, directory: Path, config: AppConfig) -> list[Path]:
    """Write corrected images and report.json into directory."""
    encoded = result.rendered.encoded_srgb
    return [
        write_tiff16(directory / "corrected.tif", encoded),
        write_image8(directory / "corrected.png", encoded),
        write_image8(directory / "corrected.jpg", encoded, config.output.jpeg_quality),
        write_image8(
            directory / "overlay.png", render_overlay(result, config.output.overlay_alpha_max) / 255
        ),
        save_comparison(result, directory / "comparison.png"),
        write_report(directory / "report.json", build_report(result)),
    ]


if __name__ == "__main__":
    app()
