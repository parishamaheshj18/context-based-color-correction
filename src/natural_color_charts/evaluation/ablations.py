"""Named configuration overrides for the ablation study."""

from __future__ import annotations

from typing import Any

from natural_color_charts.camera.repository import SRGB_EQUIVALENT_NAME
from natural_color_charts.domain import ReferenceClass

ABLATIONS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_context": {"context": {"client": "none"}},
    "no_reliability": {
        "patches": {"use_reliability_weights": False},
        "estimation": {"loo_min_references": 10_000},
    },
    "generic_sensitivity": {"camera": {"force_camera": SRGB_EQUIVALENT_NAME}},
    **{
        f"without_{label.value}": {"detection": {"excluded_classes": [label.value]}}
        for label in ReferenceClass
    },
}


def ablation_overrides(name: str) -> dict[str, Any]:
    """Overrides for a named ablation."""
    if name not in ABLATIONS:
        raise KeyError(f"Unknown ablation '{name}'. Choose from: {', '.join(ABLATIONS)}")
    return ABLATIONS[name]
