import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from natural_color_charts.cli import app
from natural_color_charts.colorimetry import encode_srgb
from natural_color_charts.io.writers import write_image8
from tests.conftest import make_cast_image


def test_correct_command_writes_outputs(tmp_path: Path):
    cast = np.array([1.4, 1.0, 0.6])
    encoded = encode_srgb(make_cast_image(cast).rgb)
    photo = write_image8(tmp_path / "photo.png", encoded)
    result = CliRunner().invoke(app, ["correct", str(photo), "--output-dir", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["input"]["mode"] == "raster"
    white = np.array(report["illuminant"]["white_rgb"])
    assert np.degrees(np.arccos(white @ cast / np.linalg.norm(white) / np.linalg.norm(cast))) < 3
    assert (tmp_path / "out" / "corrected.tif").exists()


def test_parse_setting_builds_nested_override():
    from natural_color_charts.cli import parse_setting

    assert parse_setting("detection.device=cuda") == {"detection": {"device": "cuda"}}
    assert parse_setting("patches.min_valid_pixels=50") == {"patches": {"min_valid_pixels": 50}}
