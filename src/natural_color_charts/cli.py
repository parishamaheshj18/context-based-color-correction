"""Command-line interface (`ncc`)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from natural_color_charts.config import AppConfig, load_config
from natural_color_charts.io.writers import write_image8, write_tiff16
from natural_color_charts.pipeline import PipelineResult
from natural_color_charts.pipeline_factory import build_pipeline, build_repository
from natural_color_charts.reporting.report import build_report, write_report

app = typer.Typer(help="Context-aware color correction without a color checker.")


@app.callback()
def main() -> None:
    """Context-aware color correction without a color checker."""


ConfigOption = Annotated[Path | None, typer.Option("--config", help="YAML config file.")]


def _load(config_path: Path | None, verbose: bool) -> AppConfig:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")
    return load_config(config_path)


@app.command()
def correct(
    photo: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path | None, typer.Option(help="Defaults to outputs/<stem>.")] = None,
    camera: Annotated[str | None, typer.Option(help="Force a camera's sensitivity curves.")] = None,
    config_path: ConfigOption = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Correct the colors of PHOTO and write images and a report."""
    config = _load(config_path, verbose)
    result = build_pipeline(config).run(photo, camera=camera)
    target_dir = output_dir or config.output.directory / photo.stem
    written = write_outputs(result, target_dir, config)
    for path in written:
        typer.echo(str(path))


@app.command("list-cameras")
def list_cameras(config_path: ConfigOption = None) -> None:
    """List cameras with spectral sensitivity data."""
    for name in build_repository(_load(config_path, verbose=False)).list_cameras():
        typer.echo(name)


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
