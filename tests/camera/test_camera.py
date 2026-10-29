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
    assert normalize_camera_name("Canon Canon EOS 5D Mark II") == "canon eos 5 d mark ii"
    assert normalize_camera_name("NIKON CORPORATION NIKON D5100") == "nikon d 5100"


def test_selector_never_matches_a_different_model_number(tmp_path: Path):
    for name in ("iPhone12ProMAX", "iPhone8"):
        _write_csv(tmp_path / f"{name}.csv", 530)
    repository = CameraSensitivityRepository(GRID, tmp_path, include_builtin=False)
    selector = CameraSensitivitySelector(repository, 0.6, use_same_make_fallback=False)

    def pick(model):
        image = make_image(
            np.ones((2, 2, 3)), InputMode.RAW, ImageMetadata(make="Apple", model=model)
        )
        chosen = selector.select(image)
        return chosen.name if chosen else None

    assert pick("iPhone 12 Pro Max") == "iPhone12ProMAX"
    assert pick("iPhone 16 Pro") is None


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


def test_repository_reads_headerless_txt_and_xlsx(tmp_path: Path):
    import openpyxl

    rows = [[0.1, 0.2, 1.0 if i == 6 else 0.3] for i in range(31)]
    (tmp_path / "PhoneA.txt").write_text("\n".join("\t".join(f"{v:.2E}" for v in r) for r in rows))
    workbook = openpyxl.Workbook()
    for row in rows:
        workbook.active.append(row)
    workbook.save(tmp_path / "PhoneB.xlsx")
    repository = CameraSensitivityRepository(GRID, tmp_path, include_builtin=False)
    assert repository.list_cameras() == ["PhoneA", "PhoneB"]
    blue = repository.get_sensitivity("PhoneB").sensitivity_3xn[2]
    assert GRID.wavelengths_nm[int(np.argmax(blue))] == 460
