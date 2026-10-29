"""Command-line interface (`ncc`)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from natural_color_charts.config import AppConfig, load_config
from natural_color_charts.io.writers import write_image8, write_tiff16
from natural_color_charts.pipeline import PipelineResult
from natural_color_charts.pipeline_factory import build_grid, build_pipeline, build_repository
from natural_color_charts.reporting.report import build_report, write_report
from natural_color_charts.spectral.library_reader import build_priors_from_library
from natural_color_charts.spectral.priors import save_prior

app = typer.Typer(help="Context-aware color correction without a color checker.")


@app.callback()
def main() -> None:
    """Context-aware color correction without a color checker."""


ConfigOption = Annotated[Path | None, typer.Option("--config", help="YAML config file.")]


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
    detector: Annotated[str | None, typer.Option(help="none | manual | transformers")] = None,
    config_path: ConfigOption = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Correct the colors of PHOTO and write images and a report."""
    overrides: dict[str, Any] = {}
    if vlm:
        overrides["context"] = {"client": vlm}
    if detector:
        overrides["detection"] = {"detector": detector}
    config = _load(config_path, verbose, overrides)
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


def write_outputs(result: PipelineResult, directory: Path, config: AppConfig) -> list[Path]:
    """Write corrected images and report.json into directory."""
    encoded = result.rendered.encoded_srgb
    return [
        write_tiff16(directory / "corrected.tif", encoded),
        write_image8(directory / "corrected.png", encoded),
        write_image8(directory / "corrected.jpg", encoded, config.output.jpeg_quality),
        write_report(directory / "report.json", build_report(result)),
    ]


if __name__ == "__main__":
    app()
