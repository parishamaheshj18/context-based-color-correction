"""Review evidence: corrected reference colors against their materials, and the review sheet."""

import numpy as np
import pytest

from natural_color_charts.colorimetry import D65_WHITE_XYZ, LINEAR_SRGB_TO_XYZ, xyz_to_lab
from natural_color_charts.config import load_config
from natural_color_charts.domain import UserColorAnchor
from natural_color_charts.pipeline_factory import build_evidence_builder, build_pipeline
from natural_color_charts.review.evidence import (
    cast_words,
    describe_evidence,
    render_review_sheet,
)
from tests.pipeline.test_user_anchors import NAMES, _default_context, _scene

GRAYS = ["neutral 8 (.23 D)", "neutral 5 (.70 D)", "neutral 6.5 (.44 D)"]


@pytest.fixture(scope="module")
def corrected():
    config = load_config(overrides={"correction": {"raster_method": "spectral"}})
    pipe = build_pipeline(config)
    image, boxes = _scene(GRAYS)
    preview = pipe.render_preview(image)
    claimed = [GRAYS[0], GRAYS[1], "light skin"]  # the third match is wrong on purpose
    anchors = [UserColorAnchor(b, NAMES.index(n)) for b, n in zip(boxes, claimed, strict=True)]
    result = pipe.run_with_context(
        image, preview, _default_context(config, preview), anchors=anchors
    )
    return result, build_evidence_builder(config).build(result)


def test_lab_of_the_white_is_100_0_0():
    assert np.allclose(xyz_to_lab(D65_WHITE_XYZ, D65_WHITE_XYZ), [100, 0, 0], atol=1e-6)
    gray = xyz_to_lab(LINEAR_SRGB_TO_XYZ @ np.full(3, 0.18), D65_WHITE_XYZ)
    assert abs(gray[1]) < 0.01 and abs(gray[2]) < 0.01


def test_user_matches_are_numbered_and_measured(corrected):
    _, evidence = corrected
    assert [e.reference_id for e in evidence] == ["U1", "U2", "U3"]
    assert all(e.delta_e is not None and e.box_xyxy is not None for e in evidence)


def test_a_wrong_match_stands_out_from_the_right_ones(corrected):
    _, evidence = corrected
    right, wrong = evidence[:2], evidence[2]
    assert all(e.delta_e < wrong.delta_e for e in right)
    assert wrong.delta_e > wrong.typical_spread  # gray is far from "light skin"


def test_description_names_the_direction_of_a_cast(corrected):
    _, evidence = corrected
    text = describe_evidence(evidence)
    assert "U3 [known_color] you: light skin" in text and "ΔE" in text
    assert cast_words(0.0, 5.0) == ["too yellow/warm (+5.0 b*)"]
    assert cast_words(-3.0, -0.5) == ["too green (-3.0 a*)"]


def test_sheet_shows_both_images_above_the_swatches(corrected):
    result, evidence = corrected
    sheet = render_review_sheet(result, evidence, 800)
    assert sheet.dtype == np.uint8 and sheet.shape[1] == 800
    assert sheet.shape[0] > result.preview.shape[0] * 400 / result.preview.shape[1]
