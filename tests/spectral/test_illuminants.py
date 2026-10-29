import colour
import numpy as np
import pytest

from natural_color_charts.domain import WavelengthGrid
from natural_color_charts.spectral.datasets import cie_1931_cmfs, standard_illuminant_spd
from natural_color_charts.spectral.illuminants import (
    BlackbodyFamily,
    DaylightFamily,
    blackbody_spd,
    normalize_spd,
)
from natural_color_charts.spectral.locus import match_locus
from natural_color_charts.spectral.rendering import white_response

GRID = WavelengthGrid(400, 700, 10)


def test_daylight_6504k_matches_cie_d65():
    d65 = normalize_spd(standard_illuminant_spd("D65", GRID), GRID)
    np.testing.assert_allclose(DaylightFamily(GRID, 4000, 25000).spd(6504), d65, rtol=0.02)


def test_blackbody_2856k_matches_cie_a():
    a = normalize_spd(standard_illuminant_spd("A", GRID), GRID)
    np.testing.assert_allclose(blackbody_spd(2856, GRID), a, rtol=0.01)


@pytest.mark.parametrize(("family_name", "cct"), [("daylight", 5200.0), ("blackbody", 3100.0)])
def test_locus_match_recovers_cct(family_name, cct):
    families = [DaylightFamily(GRID, 4000, 25000), BlackbodyFamily(GRID, 1800, 6500)]
    sensitivity = cie_1931_cmfs(GRID)
    truth = next(f for f in families if f.name == family_name).spd(cct)
    match = match_locus(white_response(sensitivity, truth), sensitivity, families)
    assert match.family == family_name
    assert abs(match.cct_k - cct) / cct < 0.01


def test_daylight_chromaticity_follows_cie_formula():
    xy = colour.XYZ_to_xy(
        white_response(cie_1931_cmfs(GRID), DaylightFamily(GRID, 4000, 25000).spd(5000))
    )
    np.testing.assert_allclose(
        xy, colour.temperature.CCT_to_xy_CIE_D(5000 * 1.4388 / 1.4380), atol=2e-3
    )
