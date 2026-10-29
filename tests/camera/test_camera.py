from pathlib import Path

import numpy as np
import pytest

from natural_color_charts.camera.repository import (
    SRGB_EQUIVALENT_NAME,
    CameraSensitivityRepository,
    normalize_camera_name,
)
from natural_color_charts.camera.selector import CameraSensitivitySelector
from natural_color_charts.domain import ImageMetadata, InputMode, WavelengthGrid
from tests.conftest import make_image

GRID = WavelengthGrid(400, 700, 10)


def _write_csv(path: Path, peak_nm: float) -> None:
    wavelengths = np.arange(380, 781, 5)
    rows = ["wavelength,r,g,b"]
    for w in wavelengths:
        r, g, b = (np.exp(-(((w - c) / 30) ** 2)) for c in (peak_nm + 60, peak_nm, peak_nm - 70))
        rows.append(f"{w},{r},{g},{b}")
    path.write_text("\n".join(rows))


@pytest.fixture
def repository(tmp_path: Path) -> CameraSensitivityRepository:
    _write_csv(tmp_path / "Canon_EOS_5D_Mark_II.csv", 530)
    (tmp_path / "camspec.txt").write_text(
        "Nikon D5100\n" + "\n".join(" ".join(["0.5"] * 33) for _ in range(3)) + "\n"
    )
    return CameraSensitivityRepository(GRID, tmp_path, include_builtin=True)


def test_repository_loads_csv_jiang_and_builtin(repository):
    names = repository.list_cameras()
    assert {"Canon EOS 5D Mark II", "Nikon D5100", "Nikon 5100", "Sigma SDMerill"} <= set(names)
    curves = repository.get_sensitivity("Canon EOS 5D Mark II").sensitivity_3xn
    assert curves.shape == (3, GRID.size) and curves.max() == pytest.approx(1.0)


def test_normalize_camera_name_drops_repeated_make():
    assert normalize_camera_name("Canon Canon EOS 5D Mark II") == "canon eos 5d mark ii"
    assert normalize_camera_name("NIKON CORPORATION NIKON D5100") == "nikon d5100"


def test_selector_prefers_request_then_exif(repository):
    selector = CameraSensitivitySelector(repository, 0.6, use_same_make_fallback=False)
    raw = make_image(
        np.ones((2, 2, 3)), InputMode.RAW, ImageMetadata(make="Canon", model="EOS 5D Mark II")
    )
    assert selector.select(raw).name == "Canon EOS 5D Mark II"
    requested = selector.select(raw, requested="nikon d5100")
    assert requested.name == "Nikon D5100" and "requested" in requested.selection_reason
    with pytest.raises(LookupError):
        selector.select(raw, requested="Hasselblad X2D")


def test_selector_fallbacks(repository):
    selector = CameraSensitivitySelector(repository, 0.9, use_same_make_fallback=True)
    raster = make_image(np.ones((2, 2, 3)), InputMode.RASTER)
    assert selector.select(raster).name == SRGB_EQUIVALENT_NAME
    unknown_canon = make_image(
        np.ones((2, 2, 3)), InputMode.RAW, ImageMetadata(make="Canon", model="R5")
    )
    chosen = selector.select(unknown_canon)
    assert chosen.name == "Canon EOS 5D Mark II" and chosen.is_approximate
    unknown = make_image(
        np.ones((2, 2, 3)), InputMode.RAW, ImageMetadata(make="Leica", model="M11")
    )
    assert selector.select(unknown) is None
