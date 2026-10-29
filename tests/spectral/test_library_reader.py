from pathlib import Path

import numpy as np

from natural_color_charts.domain import ReferenceClass, WavelengthGrid
from natural_color_charts.spectral.library_reader import (
    build_priors_from_library,
    read_spectra_file,
)

GRID = WavelengthGrid(400, 700, 10)


def test_reads_ecostress_units(tmp_path: Path):
    path = tmp_path / "leaf.txt"
    lines = [
        "Name: Green leaf",
        "X Units: Wavelength (micrometers)",
        "Y Units: Reflectance (percent)",
    ]
    lines += [f"{w:.3f}\t{10 + 30 * (w > 0.55):.2f}" for w in np.arange(0.35, 0.80, 0.005)]
    path.write_text("\n".join(lines))
    wavelengths, values = read_spectra_file(path)
    assert wavelengths.min() == 350 and values.shape == (1, wavelengths.size)
    assert values.min() > 0.09 and values.max() < 0.41


def test_builds_priors_for_class_folders(tmp_path: Path, config):
    rng = np.random.default_rng(0)
    (tmp_path / "snow").mkdir()
    wavelengths = np.arange(380, 781, 5)
    for i in range(3):
        samples = 0.85 + 0.02 * rng.normal(size=(wavelengths.size, 4))
        table = np.column_stack([wavelengths, samples])
        np.savetxt(tmp_path / "snow" / f"s{i}.csv", table, delimiter=",", header="nm,a,b,c,d")
    priors = build_priors_from_library(tmp_path, GRID, config.priors)
    assert [p.label for p in priors] == [ReferenceClass.SNOW]
    assert priors[0].sample_count == 12 and abs(priors[0].mean_n.mean() - 0.85) < 0.01
