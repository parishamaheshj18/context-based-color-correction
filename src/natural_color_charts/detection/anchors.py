"""User color matches: a box in the photo plus the chart patch its true color matches."""

from __future__ import annotations

from collections.abc import Sequence

from natural_color_charts.detection.base import MaskRefiner, PreviewImage, Segmenter
from natural_color_charts.detection.region_finder import upscale_mask
from natural_color_charts.domain import (
    Detection,
    LinearImage,
    ReferenceClass,
    ReferenceRegion,
    SpectralArray,
    UserColorAnchor,
)


class AnchorRegionBuilder:
    """Segments each user box and attaches the matched patch's measured reflectance."""

    def __init__(
        self,
        chart_names: Sequence[str],
        chart_reflectances_kxn: SpectralArray,
        segmenter: Segmenter,
        refiners: Sequence[MaskRefiner],
    ) -> None:
        self._names = list(chart_names)
        self._reflectances = chart_reflectances_kxn
        self._segmenter = segmenter
        self._refiners = list(refiners)

    def build(
        self, image: LinearImage, preview: PreviewImage, anchors: Sequence[UserColorAnchor]
    ) -> list[ReferenceRegion]:
        """One region per anchor, in the order given."""
        for anchor in anchors:
            if not 0 <= anchor.chart_patch_index < len(self._names):
                raise ValueError(f"Chart patch index {anchor.chart_patch_index} out of range")
        detections = [Detection(ReferenceClass.KNOWN_COLOR, a.box_xyxy, 1.0) for a in anchors]
        masks = self._segmenter.segment(preview, detections)
        regions = []
        for anchor, mask in zip(anchors, masks, strict=True):
            for refiner in self._refiners:
                mask = refiner.refine(preview, ReferenceClass.KNOWN_COLOR, mask)
            regions.append(
                ReferenceRegion(
                    label=ReferenceClass.KNOWN_COLOR,
                    mask=upscale_mask(mask, image.width, image.height),
                    detector_score=1.0,
                    box_xyxy=anchor.box_xyxy,
                    candidate_confidence=anchor.trust,
                    description=f"you: {self._names[anchor.chart_patch_index]}",
                    source="user",
                    known_reflectance=self._reflectances[anchor.chart_patch_index],
                )
            )
        return regions
