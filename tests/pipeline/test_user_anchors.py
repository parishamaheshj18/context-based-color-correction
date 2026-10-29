"""User color matches: a box whose true color is a known chart patch."""

import numpy as np
import pytest

from natural_color_charts.camera.repository import srgb_equivalent_sensitivity
from natural_color_charts.colorimetry import angular_error_deg
from natural_color_charts.config import load_config
from natural_color_charts.context.base import DefaultSceneContextProvider, KeywordIntentParser
from natural_color_charts.domain import (
    InputMode,
    ReferenceClass,
    UserColorAnchor,
    WavelengthGrid,
)
from natural_color_charts.evaluation.synthetic import make_patch_image
from natural_color_charts.pipeline_factory import build_pipeline
from natural_color_charts.spectral.datasets import chart_reflectances
from natural_color_charts.spectral.illuminants import BlackbodyFamily
from natural_color_charts.spectral.rendering import sensor_responses, white_response

GRID = WavelengthGrid(400, 700, 10)
NAMES, CHART = chart_reflectances("BabelColor Average", GRID)
SENSITIVITY = srgb_equivalent_sensitivity(GRID)
LIGHT = BlackbodyFamily(GRID, 1800, 6500).spd(3000)
TRUTH = white_response(SENSITIVITY, LIGHT)


def _scene(patch_names):
    reflectances = np.stack([CHART[NAMES.index(n)] for n in patch_names])
    rgb = sensor_responses(SENSITIVITY, LIGHT, reflectances)
    rgb = rgb / rgb.max() * 0.7
    background = sensor_responses(SENSITIVITY, LIGHT, CHART[NAMES.index("purplish blue")][None])[0]
    scene = make_patch_image(
        rgb,
        [ReferenceClass.KNOWN_COLOR] * len(patch_names),
        48,
        background / rgb.max() * 0.7,
        InputMode.RASTER,
    )
    height, width = scene.image.height, scene.image.width
    boxes = []
    for region in scene.regions:
        ys, xs = np.nonzero(region.mask)
        boxes.append(
            (xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height)
        )
    return scene.image, boxes


@pytest.fixture(scope="module")
def pipeline():
    config = load_config(overrides={"correction": {"raster_method": "spectral"}})
    return build_pipeline(config), config


def _default_context(config, preview):
    provider = DefaultSceneContextProvider(
        ["daylight", "blackbody"], config.estimation, 0.5, KeywordIntentParser(0.6)
    )
    return provider.analyze(preview, None)


def test_a_single_known_colored_object_recovers_the_light(pipeline):
    pipe, config = pipeline
    image, boxes = _scene(["red"])
    preview = pipe.render_preview(image)
    context = _default_context(config, preview)
    without = pipe.run_with_context(image, preview, context)
    anchors = [UserColorAnchor(boxes[0], NAMES.index("red"))]
    with_anchor = pipe.run_with_context(image, preview, context, anchors=anchors)
    assert with_anchor.estimate.method == "spectral"
    assert angular_error_deg(with_anchor.estimate.white_rgb, TRUTH) < 1.0
    assert angular_error_deg(without.estimate.white_rgb, TRUTH) > 5.0  # blue wall fools statistics
    user_patch = next(p for p in with_anchor.patches if p.source == "user")
    assert user_patch.kept and user_patch.weight > 0


def test_user_matches_are_never_rejected_as_outliers(pipeline):
    pipe, config = pipeline
    # The third match is wrong on purpose (a gray patch labeled as light skin).
    image, boxes = _scene(["neutral 8 (.23 D)", "neutral 5 (.70 D)", "neutral 6.5 (.44 D)"])
    preview = pipe.render_preview(image)
    context = _default_context(config, preview)
    claimed = ["neutral 8 (.23 D)", "neutral 5 (.70 D)", "light skin"]
    anchors = [UserColorAnchor(box, NAMES.index(n)) for box, n in zip(boxes, claimed, strict=True)]
    result = pipe.run_with_context(image, preview, context, anchors=anchors)
    user_patches = [p for p in result.patches if p.source == "user"]
    assert len(user_patches) == 3 and all(p.kept for p in user_patches)


def test_out_of_range_patch_index_is_rejected(pipeline):
    pipe, config = pipeline
    image, boxes = _scene(["red"])
    preview = pipe.render_preview(image)
    with pytest.raises(ValueError):
        pipe.run_with_context(
            image,
            preview,
            _default_context(config, preview),
            anchors=[UserColorAnchor(boxes[0], 24)],
        )
