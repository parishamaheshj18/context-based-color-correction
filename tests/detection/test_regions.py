import json
from pathlib import Path

import numpy as np

from natural_color_charts.detection.base import DetectionRequest, box_to_pixels
from natural_color_charts.detection.region_finder import ReferenceRegionFinder
from natural_color_charts.detection.simple import (
    BoxSegmenter,
    ManualRegionDetector,
    NullDetector,
    SkinPixelRefiner,
)
from natural_color_charts.detection.transformers_models import box_iou, suppress_overlaps
from natural_color_charts.domain import (
    Detection,
    ReferenceCandidate,
    ReferenceClass,
    SceneContext,
)
from tests.conftest import make_image

PAPER, SKIN = ReferenceClass.PAPER, ReferenceClass.SKIN


def test_box_to_pixels_clamps_and_orders():
    assert box_to_pixels((0.5, 0.9, -0.1, 0.1), 100, 50) == (0, 5, 50, 45)


def test_manual_detector_reads_sidecar(tmp_path: Path):
    photo = tmp_path / "shot.dng"
    (tmp_path / "shot.regions.json").write_text(
        json.dumps(
            [
                {"label": "paper", "box": [0.1, 0.1, 0.5, 0.5]},
                {"label": "snow", "box": [0, 0, 1, 1]},
            ]
        )
    )
    request = DetectionRequest(np.zeros((10, 10, 3), np.uint8), [PAPER], photo)
    detections = ManualRegionDetector(".regions.json").detect(request)
    assert [d.label for d in detections] == [PAPER]


def test_box_segmenter_insets_box():
    masks = BoxSegmenter(0.1).segment(
        np.zeros((100, 100, 3), np.uint8), [Detection(PAPER, (0, 0, 0.5, 0.5), 1.0)]
    )
    assert masks[0].sum() == 40 * 40 and masks[0][5, 5] and not masks[0][2, 2]


def test_skin_refiner_keeps_skin_chroma_only():
    preview = np.zeros((2, 2, 3), np.uint8)
    preview[0, 0] = (198, 134, 66)  # medium-deep skin tone
    preview[0, 1] = (255, 219, 172)  # light skin tone
    preview[1, 0] = (20, 60, 200)  # blue clothing
    preview[1, 1] = (250, 250, 250)  # eye white
    mask = np.ones((2, 2), bool)
    refined = SkinPixelRefiner((133, 173), (77, 127)).refine(preview, SKIN, mask)
    assert refined.tolist() == [[True, True], [False, False]]
    assert SkinPixelRefiner((133, 173), (77, 127)).refine(preview, PAPER, mask).all()


def test_suppress_overlaps_keeps_best_per_label():
    detections = [
        Detection(PAPER, (0, 0, 1, 1), 0.9),
        Detection(PAPER, (0, 0, 0.95, 0.95), 0.8),
        Detection(PAPER, (2, 2, 3, 3), 0.5),
        Detection(SKIN, (0, 0, 1, 1), 0.7),
    ]
    kept = suppress_overlaps(detections, 0.5, max_per_label=3)
    assert [d.score for d in kept] == [0.9, 0.7, 0.5]
    assert box_iou((0, 0, 1, 1), (0.5, 0, 1.5, 1)) == 1 / 3


class _FixedDetector:
    def detect(self, request):
        return [Detection(label, (0.0, 0.0, 0.5, 0.5), 0.9) for label in request.labels]


def test_region_finder_only_detects_usable_candidates_and_upscales():
    image = make_image(np.full((40, 80, 3), 0.2))
    context = SceneContext(
        reference_candidates=(
            ReferenceCandidate(PAPER, "letter", usable=True, confidence=0.7),
            ReferenceCandidate(SKIN, "face under neon", usable=False),
        )
    )
    finder = ReferenceRegionFinder(_FixedDetector(), BoxSegmenter(0.0), [])
    regions = finder.find(image, np.zeros((10, 20, 3), np.uint8), context)
    assert [r.label for r in regions] == [PAPER]
    assert regions[0].mask.shape == (40, 80) and regions[0].mask.sum() == 20 * 40
    assert regions[0].candidate_confidence == 0.7


def test_null_detector_finds_nothing():
    assert NullDetector().detect(DetectionRequest(np.zeros((2, 2, 3), np.uint8), [PAPER])) == []


def test_context_box_detector_uses_vlm_boxes_of_usable_candidates():
    from natural_color_charts.detection.simple import ContextBoxDetector

    context = SceneContext(
        reference_candidates=(
            ReferenceCandidate(
                PAPER, "letter", True, 0.8, (), ((0.1, 0.1, 0.4, 0.4), (0.5, 0.5, 0.6, 0.6))
            ),
            ReferenceCandidate(SKIN, "face under neon", False, 0.2, (), ((0.0, 0.0, 1.0, 1.0),)),
        )
    )
    request = DetectionRequest(np.zeros((4, 4, 3), np.uint8), [PAPER, SKIN], None, context)
    detections = ContextBoxDetector().detect(request)
    assert [d.label for d in detections] == [PAPER, PAPER]
    assert detections[0].candidate_confidence == 0.8 and detections[0].description == "letter"


def test_chroma_coherence_refiner_drops_off_color_pixels():
    from natural_color_charts.detection.simple import ChromaCoherenceRefiner

    preview = np.full((10, 10, 3), 200, np.uint8)
    preview[:, 7:] = (200, 40, 40)  # red object inside the box
    refined = ChromaCoherenceRefiner(0.04).refine(preview, PAPER, np.ones((10, 10), bool))
    assert refined[:, :7].all() and not refined[:, 7:].any()


def test_vlm_boxes_are_cleaned():
    from natural_color_charts.context.schema import CandidateResponse

    candidate = CandidateResponse(
        label="paper", description="", usable=True, confidence=0.5, concerns=[],
        boxes=[[0.5, 0.2, 0.1, 0.9], [0.2, 0.2, 0.2005, 0.3], [0.1, 0.1], [-0.2, 0.0, 0.3, 1.4]],
    )  # fmt: skip
    assert candidate.clean_boxes() == ((0.1, 0.2, 0.5, 0.9), (0.0, 0.0, 0.3, 1.0))
