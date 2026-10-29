"""The matching window: click a pixel, click its chart patch, add Claude's picks, correct."""

import numpy as np
import pytest

from natural_color_charts.colorimetry import angular_error_deg, encode_srgb
from natural_color_charts.config import load_config
from natural_color_charts.domain import ReferenceCandidate, ReferenceClass, SceneContext
from natural_color_charts.gui.picker import MatchPicker
from natural_color_charts.io.writers import write_tiff16
from natural_color_charts.pipeline_factory import build_match_session
from tests.pipeline.test_user_anchors import NAMES, TRUTH, _scene

pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")


@pytest.fixture
def session(tmp_path):
    image, boxes = _scene(["red"])
    photo = write_tiff16(tmp_path / "scene.tif", encode_srgb(image.rgb))
    config = load_config(overrides={"correction": {"raster_method": "spectral"}})
    return build_match_session(config, photo, None, 2048), boxes[0]


def test_chart_swatches_are_the_24_patches_in_order(session):
    swatches = session[0].swatches
    assert len(swatches) == 24 and swatches[0].name == NAMES[0]
    white = np.array(swatches[18].srgb)
    assert white.min() > 0.85 and np.ptp(white) < 0.03
    red = np.array(swatches[NAMES.index("red")].srgb)
    assert red[0] > 2 * red[1]


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def test_clicked_pixel_paired_with_a_patch_corrects_the_photo(session, tmp_path):
    match_session, box = session
    picker = MatchPicker(match_session, "scene", tmp_path / "out")
    picker.choose_patch(0)  # no pixel yet: ignored
    assert picker.matches == []
    picker.pick_pixel(*_center(box))
    picker.choose_patch(NAMES.index("red"))
    assert len(picker.matches) == 1 and picker.pending is None
    pixel = picker.matches[0].pixel_rgb
    assert pixel[0] > pixel[1] and pixel[0] > pixel[2]  # the patch looks red-orange
    saved = picker.correct()
    assert saved.exists() and (tmp_path / "out" / "comparison.png").exists()
    rows = (tmp_path / "out" / "matches.csv").read_text().splitlines()
    assert len(rows) == 2 and ",red," in rows[1]
    result = match_session.correct(picker.anchors)
    assert angular_error_deg(result.estimate.white_rgb, TRUTH) < 1.5


def test_a_click_measures_a_small_square_around_the_pixel(session, tmp_path):
    match_session, _ = session
    picker = MatchPicker(match_session, "scene", tmp_path)
    picker.pick_pixel(0.5, 0.5)
    picker.choose_patch(0)
    x0, y0, x1, y1 = picker.anchors[0].box_xyxy
    width, height = match_session.preview.shape[1], match_session.preview.shape[0]
    assert x0 < 0.5 < x1 and y0 < 0.5 < y1
    # Small images get the 12 px floor on each side, enough pixels to measure.
    assert round((x1 - x0) * width) == 24 and round((y1 - y0) * height) == 24


def test_undo_drops_pending_pixel_then_last_pair(session, tmp_path):
    match_session, box = session
    picker = MatchPicker(match_session, "scene", tmp_path)
    picker.pick_pixel(*_center(box))
    picker.choose_patch(3)
    picker.pick_pixel(*_center(box))
    picker.undo()
    assert picker.pending is None and len(picker.matches) == 1
    picker.undo()
    assert picker.matches == []


def test_clicking_a_claude_pick_ignores_it(session, tmp_path):
    match_session, _ = session
    match_session.claude = SceneContext(
        reference_candidates=(
            ReferenceCandidate(
                ReferenceClass.PAPER, "sheet", True, 0.9, (), ((0.1, 0.1, 0.4, 0.4),)
            ),
        ),
        source="vlm",
    )
    picker = MatchPicker(match_session, "scene", tmp_path)
    picker.toggle_claude_at(0.2, 0.2)
    assert picker.ignored == {0}
    assert not match_session._context(picker.ignored).reference_candidates[0].usable
    picker.toggle_claude_at(0.2, 0.2)
    assert picker.ignored == set()
