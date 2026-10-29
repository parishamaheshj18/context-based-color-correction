from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from natural_color_charts.colorimetry import decode_srgb
from natural_color_charts.domain import InputMode
from natural_color_charts.io import loaders
from natural_color_charts.io.loaders import (
    RasterImageLoader,
    RawImageLoader,
    SuffixDispatchingLoader,
)
from natural_color_charts.io.writers import write_image8, write_tiff16


def test_raster_loader_linearizes_png(tmp_path: Path):
    encoded = np.full((8, 8, 3), 0.5, dtype=np.float32)
    encoded[0, 0] = 1.0
    path = write_image8(tmp_path / "x.png", encoded)
    image = RasterImageLoader(saturation_threshold=0.98).load(path)
    assert image.mode is InputMode.RASTER
    np.testing.assert_allclose(image.rgb[1, 1], decode_srgb(np.float32(128 / 255)), atol=1e-6)
    assert image.saturation_mask[0, 0] and image.saturation_mask.sum() == 1


def test_raster_loader_keeps_16_bit_tiff(tmp_path: Path):
    encoded = np.full((4, 4, 3), 1000 / 65535, dtype=np.float32)
    path = write_tiff16(tmp_path / "x.tif", encoded)
    image = RasterImageLoader(0.98).load(path)
    np.testing.assert_allclose(image.rgb[0, 0], decode_srgb(encoded[0, 0]), rtol=1e-4)


class _FakeRaw:
    camera_whitebalance = (2.0, 1.0, 1.5, 1.0)
    rgb_xyz_matrix = np.vstack([np.eye(3) * 0.8, np.zeros((1, 3))]).astype(np.float32)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def postprocess(self, **kwargs):
        assert kwargs["gamma"] == (1, 1) and kwargs["no_auto_bright"]
        rgb = np.full((4, 6, 3), 32768, dtype=np.uint16)
        rgb[0, 0, 1] = 65535
        return rgb


def test_raw_loader_reads_linear_camera_rgb(monkeypatch, tmp_path: Path):
    path = tmp_path / "photo.dng"
    path.write_bytes(b"")
    fake_rawpy = SimpleNamespace(imread=lambda _: _FakeRaw(), ColorSpace=SimpleNamespace(raw=0))
    monkeypatch.setattr(loaders, "import_rawpy", lambda: fake_rawpy)
    image = RawImageLoader(saturation_threshold=0.98).load(path)
    assert image.mode is InputMode.RAW
    np.testing.assert_allclose(image.rgb[1, 1], 32768 / 65535, rtol=1e-6)
    assert image.saturation_mask[0, 0] and not image.saturation_mask[1, 1]
    np.testing.assert_allclose(image.metadata.as_shot_white_rgb, [0.5, 1.0, 1 / 1.5])
    np.testing.assert_allclose(image.metadata.xyz_to_camera_matrix, np.eye(3) * 0.8, rtol=1e-6)


def test_dispatching_loader_rejects_unknown_suffix(tmp_path: Path):
    loader = SuffixDispatchingLoader([RawImageLoader(0.98), RasterImageLoader(0.98)])
    with pytest.raises(ValueError):
        loader.load(tmp_path / "x.gif")
    assert loader.supports(Path("a.NEF")) and loader.supports(Path("a.jpg"))


def test_as_shot_white_ignores_invalid_multipliers():
    assert loaders._as_shot_white([0.0, 1.0, 1.0, 0.0]) is None
