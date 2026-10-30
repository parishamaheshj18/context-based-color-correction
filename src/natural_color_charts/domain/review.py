"""Agentic review: measured evidence per reference and the critic's judgment of a correction."""

from __future__ import annotations

from dataclasses import dataclass

from natural_color_charts.domain.scene import BoxXyxy, ReferenceClass


@dataclass(frozen=True)
class ReferenceEvidence:
    """How one reference looks after neutral correction, against what its material should be.

    Both colors are compared at the expected lightness, so exposure does not count as error.
    reference_id is "A<n>" for the scene analysis's regions and "U<n>" for the user's matches.
    """

    reference_id: str
    label: ReferenceClass
    name: str
    used: bool
    weight: float
    status: str
    box_xyxy: BoxXyxy | None = None
    measured_lab: tuple[float, float, float] | None = None
    expected_lab: tuple[float, float, float] | None = None
    typical_spread: float | None = None  # ΔE*ab of one standard deviation of the material
    # (corrected, expected) display colors at a shared lightness, for the review sheet
    swatch_srgb: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    @property
    def delta_e(self) -> float | None:
        """Chromatic difference ΔE*ab between measured and expected (same lightness)."""
        if self.measured_lab is None or self.expected_lab is None:
            return None
        da, db = self.delta_ab
        return float((da**2 + db**2) ** 0.5)

    @property
    def delta_ab(self) -> tuple[float, float]:
        """Measured minus expected a* and b*."""
        if self.measured_lab is None or self.expected_lab is None:
            return 0.0, 0.0
        return (
            self.measured_lab[1] - self.expected_lab[1],
            self.measured_lab[2] - self.expected_lab[2],
        )


@dataclass(frozen=True)
class ReferenceVerdict:
    """The critic's judgment of one reference."""

    reference_id: str
    verdict: str  # consistent | doubtful | wrong
    reason: str


@dataclass(frozen=True)
class Critique:
    """The critic's judgment of one corrected image."""

    observations: str
    verdicts: tuple[ReferenceVerdict, ...]
    remaining_cast: str
    cast_severity: float
    suggestions: tuple[str, ...]
    acceptable: bool
