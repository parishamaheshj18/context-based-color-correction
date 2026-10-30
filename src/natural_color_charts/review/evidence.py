"""Review evidence: each reference's corrected color against its material, and the review sheet.

Vision-language models judge "do these two colors match?" far better than "is this gray?", so
every reference is shown and measured next to how its material should look under the target
light, at equal lightness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from natural_color_charts.colorimetry import (
    XYZ_TO_LINEAR_SRGB,
    apply_matrix,
    encode_srgb,
    xyz_to_lab,
)
from natural_color_charts.domain import (
    PatchMeasurement,
    ReferenceClass,
    ReferenceEvidence,
    RgbVector,
    SpectralArray,
)
from natural_color_charts.io.writers import to_uint8
from natural_color_charts.pipeline import PipelineResult
from natural_color_charts.spectral.priors import ReflectancePrior

SWATCH_LUMINANCE = 0.35  # swatches share one lightness so only their color differs
AI_COLOR, USER_COLOR, UNUSED_COLOR = (98, 80, 214), (235, 104, 52), (230, 30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX
CELL_W, CELL_H, SWATCH_PX, GAP_PX = 150, 92, 56, 12


class ReviewEvidenceBuilder:
    """Measures every reference of a pipeline result against its material's expected color."""

    def __init__(
        self,
        priors: Mapping[ReferenceClass, ReflectancePrior],
        cmfs_3xn: SpectralArray,
        target_spd: SpectralArray,
        min_spread_delta_e: float,
    ) -> None:
        self._priors = dict(priors)
        self._cmfs = cmfs_3xn
        self._spd = target_spd
        self._min_spread = min_spread_delta_e
        self._white = self._xyz(np.ones(cmfs_3xn.shape[1]))

    def build(self, result: PipelineResult) -> tuple[ReferenceEvidence, ...]:
        """Evidence for every patch, numbered A1.. for AI regions and U1.. for user matches."""
        transform = result.neutral_transform or result.transform
        counts = {"A": 0, "U": 0}
        items = []
        for patch in result.patches:
            prefix = "U" if patch.source == "user" else "A"
            counts[prefix] += 1
            region = (
                result.regions[patch.region_index]
                if 0 <= patch.region_index < len(result.regions)
                else None
            )
            used = patch.kept and patch.label.value in result.estimate.used_labels
            colors = self._compare(patch, transform.camera_to_xyz)
            items.append(
                ReferenceEvidence(
                    reference_id=f"{prefix}{counts[prefix]}",
                    label=patch.label,
                    name=(region.description if region else "") or patch.label.value,
                    used=used,
                    weight=patch.weight,
                    status=f"used, weight {patch.weight:.2f}"
                    if used
                    else patch.rejection_reason or "not used",
                    box_xyxy=region.box_xyxy if region else None,
                    **colors,
                )
            )
        return tuple(items)

    def _compare(
        self, patch: PatchMeasurement, camera_to_xyz: NDArray[np.float64]
    ) -> dict[str, Any]:
        reflectance = patch.known_reflectance
        prior = self._priors.get(patch.label)
        if reflectance is None and prior is not None:
            reflectance = prior.mean_n
        measured = camera_to_xyz @ patch.rgb
        if reflectance is None or patch.pixel_count == 0 or measured[1] <= 0:
            return {}
        expected = self._xyz(reflectance)
        measured = measured * expected[1] / measured[1]
        expected_lab = xyz_to_lab(expected, self._white)
        spread = self._spread(prior, expected) if patch.known_reflectance is None else 0.0
        return {
            "measured_lab": _triple(xyz_to_lab(measured, self._white)),
            "expected_lab": _triple(expected_lab),
            "typical_spread": max(spread, self._min_spread),
            "swatch_srgb": (_swatch(measured), _swatch(expected)),
        }

    def _spread(self, prior: ReflectancePrior | None, expected_xyz: RgbVector) -> float:
        """Largest ΔE*ab of a one-standard-deviation change of the material, at equal lightness."""
        if prior is None or prior.components == 0:
            return 0.0
        steps = (prior.basis_nxk * np.sqrt(prior.variances_k)).T
        variants = np.maximum(np.concatenate([prior.mean_n + steps, prior.mean_n - steps]), 1e-4)
        xyz = self._xyz(variants)
        xyz = xyz * (expected_xyz[1] / xyz[:, 1])[:, None]
        ab = xyz_to_lab(xyz, self._white)[:, 1:] - xyz_to_lab(expected_xyz, self._white)[1:]
        return float(np.max(np.hypot(ab[:, 0], ab[:, 1])))

    def _xyz(self, reflectance: SpectralArray) -> NDArray[np.float64]:
        """XYZ under the target light, relative to a perfect white of luminance 1."""
        spd = self._spd
        return np.asarray((reflectance * spd) @ self._cmfs.T / (spd @ self._cmfs[1]))


def describe_evidence(items: Sequence[ReferenceEvidence]) -> str:
    """The measurement table given to both agents."""
    lines = [
        "References after a fully neutral correction. ΔE is the color difference from what the "
        "material should look like under neutral daylight, at the same lightness; within the "
        "material's typical spread is normal. A = chosen by the AI, U = the user's color match."
    ]
    for item in items:
        line = f"{item.reference_id} [{item.label.value}] {item.name}: {item.status}"
        if item.delta_e is not None and item.typical_spread is not None:
            line += f"; ΔE {item.delta_e:.1f} (typical spread {item.typical_spread:.1f})"
            line += "".join(f", {word}" for word in cast_words(*item.delta_ab))
        lines.append(line)
    return "\n".join(lines)


def describe_highlights(result: PipelineResult) -> str:
    """How much of the photo the correction pushed past the display maximum, and where.

    Those pixels are rolled off, not clipped, but large bright areas are where a correction
    most often looks wrong, so the critic is told where to look.
    """
    image = result.image
    linear = apply_matrix(image.rgb, XYZ_TO_LINEAR_SRGB @ result.transform.camera_to_xyz)
    pushed = (linear * result.rendered.exposure_gain).max(axis=2) > 1.0
    if pushed.mean() < 0.001:
        return "Highlights: the correction pushed no part of the photo past the display maximum."
    height = pushed.shape[0]
    thirds = [pushed[i * height // 3 : (i + 1) * height // 3].mean() for i in range(3)]
    where = ("top", "middle", "bottom")[int(np.argmax(thirds))]
    return (
        f"Highlights: the correction pushed {pushed.mean():.1%} of the photo past the display "
        f"maximum (rolled off smoothly), mostly in the {where} third; "
        f"{image.saturation_mask.mean():.1%} was already at the maximum in the file."
    )


def cast_words(delta_a: float, delta_b: float, threshold: float = 1.0) -> list[str]:
    """Plain words for a color difference in a* (green-magenta) and b* (blue-yellow)."""
    words = []
    if abs(delta_b) >= threshold:
        words.append(f"{'too yellow/warm' if delta_b > 0 else 'too blue/cool'} ({delta_b:+.1f} b*)")
    if abs(delta_a) >= threshold:
        words.append(f"{'too magenta' if delta_a > 0 else 'too green'} ({delta_a:+.1f} a*)")
    return words


def render_review_sheet(
    result: PipelineResult, items: Sequence[ReferenceEvidence], width_px: int
) -> NDArray[np.uint8]:
    """Before | after with numbered reference boxes, above a strip of swatch pairs."""
    half = (width_px - GAP_PX) // 2
    # The as-shot render shares the correction's exposure and tone steps, so only color differs.
    as_shot = to_uint8(result.as_shot.encoded_srgb) if result.as_shot else result.preview
    before = _fit_width(as_shot, half)
    rendered = to_uint8(result.rendered.encoded_srgb)
    size = (before.shape[1], before.shape[0])
    after = np.asarray(cv2.resize(rendered, size, interpolation=cv2.INTER_AREA), np.uint8)
    for item in items:
        if item.box_xyxy is not None:
            _draw_box(after, item)
    before, after = _title(before, "As shot"), _title(after, "Corrected")
    gap = np.full((before.shape[0], GAP_PX, 3), 255, np.uint8)
    top = np.hstack([before, gap, after])
    return np.vstack([top, _swatch_strip([i for i in items if i.swatch_srgb], top.shape[1])])


def _fit_width(image: NDArray[np.uint8], width: int) -> NDArray[np.uint8]:
    height = round(image.shape[0] * width / image.shape[1])
    return np.asarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))


def _title(image: NDArray[np.uint8], text: str) -> NDArray[np.uint8]:
    band = np.full((30, image.shape[1], 3), 255, np.uint8)
    cv2.putText(band, text, (6, 22), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([band, image])


def _draw_box(image: NDArray[np.uint8], item: ReferenceEvidence) -> None:
    assert item.box_xyxy is not None
    height, width = image.shape[:2]
    x0, y0, x1, y1 = item.box_xyxy
    p0, p1 = (round(x0 * width), round(y0 * height)), (round(x1 * width), round(y1 * height))
    color = (
        UNUSED_COLOR if not item.used else USER_COLOR if item.reference_id[0] == "U" else AI_COLOR
    )
    cv2.rectangle(image, p0, p1, color, 2)
    origin = (p0[0], max(p0[1] - 4, 14))
    cv2.putText(image, item.reference_id, origin, FONT, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, item.reference_id, origin, FONT, 0.55, color, 1, cv2.LINE_AA)


def _swatch_strip(items: Sequence[ReferenceEvidence], width: int) -> NDArray[np.uint8]:
    """One cell per reference: corrected color (left) beside the material's color (right)."""
    columns = max(1, width // CELL_W)
    rows = -(-len(items) // columns)
    strip = np.full((40 + rows * CELL_H, width, 3), 255, np.uint8)
    caption = "Each pair: reference after correction (left) | its material's true color (right)"
    cv2.putText(strip, caption, (6, 26), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    for index, item in enumerate(items):
        assert item.swatch_srgb is not None
        row, column = divmod(index, columns)
        x, y = column * CELL_W + 6, 40 + row * CELL_H
        # A white gap between the two squares makes small differences easier to see.
        for offset, rgb in zip((0, SWATCH_PX + 4), item.swatch_srgb, strict=True):
            fill = tuple(round(v * 255) for v in rgb)
            cv2.rectangle(strip, (x + offset, y + 20), (x + offset + SWATCH_PX, y + 76), fill, -1)
        cv2.putText(strip, item.reference_id, (x, y + 14), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return strip


def _swatch(xyz: RgbVector) -> tuple[float, float, float]:
    linear = XYZ_TO_LINEAR_SRGB @ (np.asarray(xyz) * SWATCH_LUMINANCE / xyz[1])
    r, g, b = (float(v) for v in encode_srgb(linear))
    return r, g, b


def _triple(values: NDArray[np.float64]) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])
