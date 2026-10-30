"""One photo's matching session: the user's chart matches, Claude's context, the correction."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from natural_color_charts.colorimetry import XYZ_TO_LINEAR_SRGB, encode_srgb
from natural_color_charts.context.base import SceneContextProvider
from natural_color_charts.domain import (
    LinearImage,
    SceneContext,
    SpectralArray,
    UserColorAnchor,
)
from natural_color_charts.io.writers import to_uint8
from natural_color_charts.pipeline import ColorCorrectionPipeline, PipelineResult
from natural_color_charts.review.loop import AgenticCorrectionLoop, ReviewOutcome


class Swatch(NamedTuple):
    """One chart patch and its sRGB color under D65."""

    name: str
    srgb: tuple[float, float, float]


def chart_swatches(
    names: Sequence[str],
    reflectances_kxn: SpectralArray,
    cmfs_3xn: SpectralArray,
    d65_spd: SpectralArray,
) -> list[Swatch]:
    """Each chart patch as it looks under D65 in sRGB, in chart order (row by row)."""
    xyz = (reflectances_kxn * d65_spd) @ cmfs_3xn.T / (d65_spd @ cmfs_3xn[1])
    srgb = encode_srgb(xyz @ XYZ_TO_LINEAR_SRGB.T).astype(float)
    return [Swatch(name, (srgb[i, 0], srgb[i, 1], srgb[i, 2])) for i, name in enumerate(names)]


class MatchSession:
    """Holds one loaded photo; Claude's answer is kept so the user can ignore some of its picks.

    With a review loop, the user's matches and ignores are the first word, not the last: the
    agents then correct, critique, and revise, and may overrule them.
    """

    def __init__(
        self,
        pipeline: ColorCorrectionPipeline,
        default_context: SceneContextProvider,
        image: LinearImage,
        swatches: list[Swatch],
        intent: str | None,
        review_loop: AgenticCorrectionLoop | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._default_context = default_context
        self._image = image
        self.preview = pipeline.render_preview(image)
        self.swatches = swatches
        self._intent = intent
        self._review_loop = review_loop
        self.claude: SceneContext | None = None
        self.review: ReviewOutcome | None = None

    def ask_claude(self) -> SceneContext:
        """Scene analysis by the configured VLM; falls back to the default context on failure."""
        self.claude = self._pipeline.analyze_scene(self.preview, self._intent)
        return self.claude

    def correct(
        self, anchors: Sequence[UserColorAnchor], ignored: Sequence[int] = ()
    ) -> PipelineResult:
        """Correct with the user's matches plus Claude's picks, minus the ones the user ignored."""
        context = self._context(set(ignored))
        if self._review_loop is None or context.source != "vlm":
            return self._pipeline.run_with_context(
                self._image, self.preview, context, anchors=anchors
            )
        self.review = self._review_loop.run(
            self._image, self.preview, context, anchors, self._intent
        )
        return self.review.result

    def _context(self, ignored: set[int]) -> SceneContext:
        if self.claude is None:
            return self._default_context.analyze(self.preview, self._intent)
        candidates = tuple(
            dataclasses.replace(c, usable=False, concerns=(*c.concerns, "ignored by the user"))
            if i in ignored
            else c
            for i, c in enumerate(self.claude.reference_candidates)
        )
        return dataclasses.replace(self.claude, reference_candidates=candidates)


def corrected_uint8(result: PipelineResult) -> NDArray[np.uint8]:
    """The corrected image as 8-bit sRGB."""
    return to_uint8(result.rendered.encoded_srgb)


def describe(result: PipelineResult) -> list[str]:
    """Plain-language lines: the light found and which regions it relied on."""
    estimate = result.estimate
    light = (
        f"{estimate.family} light, about {round(estimate.cct_k, -1):.0f} K"
        if estimate.cct_k
        else "light color"
    )
    lines = [f"{light}, from {estimate.method.replace('_', ' ')}"]
    for patch in result.patches:
        region = result.regions[patch.region_index] if patch.region_index >= 0 else None
        who = "you" if patch.source == "user" else "Claude"
        name = region.description if region and region.description else patch.label.value
        used = patch.kept and patch.label.value in estimate.used_labels
        status = f"weight {patch.weight:.2f}" if used else patch.rejection_reason or "not used"
        lines.append(f"{who}: {name.removeprefix('you: ')} ({status})")
    return lines
