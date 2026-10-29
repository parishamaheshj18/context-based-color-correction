import numpy as np
import pytest

from natural_color_charts.detection.patch_statistics import PatchStatisticsCalculator, erode_mask
from natural_color_charts.domain import ReferenceClass, ReferenceRegion, SceneContext
from tests.conftest import make_image


def _region(mask, label=ReferenceClass.CONCRETE, confidence=1.0):
    return ReferenceRegion(
        label=label, mask=mask, detector_score=1.0, candidate_confidence=confidence
    )


@pytest.fixture
def calculator(config):
    return PatchStatisticsCalculator(config.patches.model_copy(update={"min_valid_pixels": 50}))


def test_erode_mask_removes_border():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    eroded = erode_mask(mask, 2)
    assert eroded.sum() == 36 and not eroded[5, 5]


def test_robust_color_ignores_saturated_and_specular_pixels(calculator):
    rgb = np.full((40, 40, 3), [0.30, 0.20, 0.10], np.float32)
    rgb[:1] = [0.95, 0.94, 0.93]  # small low-chroma highlight
    image = make_image(rgb)
    image.saturation_mask[35:] = True
    image.rgb[35:] = [1.0, 0.2, 0.2]
    patch = calculator.measure(image, [_region(np.ones((40, 40), bool))], SceneContext())[0]
    np.testing.assert_allclose(patch.rgb, [0.30, 0.20, 0.10], atol=1e-6)
    assert patch.kept


def test_small_regions_are_rejected_with_reason(calculator):
    mask = np.zeros((40, 40), bool)
    mask[:3, :3] = True
    patch = calculator.measure(
        make_image(np.full((40, 40, 3), 0.3)), [_region(mask)], SceneContext()
    )[0]
    assert not patch.kept and "valid pixels" in patch.rejection_reason


def test_weight_combines_confidence_size_exposure_and_class(calculator, config):
    image = make_image(np.full((100, 100, 3), 0.3))
    mask = np.ones((100, 100), bool)
    concrete, skin = calculator.measure(
        image,
        [_region(mask, ReferenceClass.CONCRETE, 0.8), _region(mask, ReferenceClass.SKIN, 0.8)],
        SceneContext(),
    )
    strengths = config.patches.class_prior_strength
    size_factor = min(1.0, concrete.pixel_count / config.patches.full_weight_pixels)
    assert concrete.weight == pytest.approx(0.8 * size_factor * strengths["concrete"])
    assert skin.weight / concrete.weight == pytest.approx(strengths["skin"] / strengths["concrete"])


def test_outdoor_paper_is_down_weighted(calculator, config):
    image = make_image(np.full((100, 100, 3), 0.3))
    region = [_region(np.ones((100, 100), bool), ReferenceClass.PAPER)]
    indoor = calculator.measure(image, region, SceneContext(environment="indoor"))[0]
    outdoor = calculator.measure(image, region, SceneContext(environment="outdoor"))[0]
    assert outdoor.weight == pytest.approx(indoor.weight * config.patches.outdoor_paper_factor)
