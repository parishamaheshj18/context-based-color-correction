"""Stage 10: machine-readable report of a pipeline run."""

from __future__ import annotations

import dataclasses
import enum
import json
from pathlib import Path
from typing import Any

import numpy as np

from natural_color_charts.domain import PatchMeasurement
from natural_color_charts.io.writers import write_image8
from natural_color_charts.pipeline import PipelineResult
from natural_color_charts.review.loop import ReviewOutcome


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, enums, paths, and numpy values into JSON-compatible objects."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return np.round(value, 6).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    return value


def build_report(result: PipelineResult) -> dict[str, Any]:
    """Summarize inputs, decisions, and the final transform."""
    image = result.image
    return {
        "input": {
            "path": to_jsonable(image.source_path),
            "mode": image.mode.value,
            "size": [image.width, image.height],
            "metadata": to_jsonable(image.metadata),
            "saturated_fraction": float(image.saturation_mask.mean()),
        },
        "camera": _camera_section(result),
        "scene_context": to_jsonable(result.context),
        "references": [_reference_entry(result, patch) for patch in result.patches],
        "illuminant": _illuminant_section(result),
        "correction": to_jsonable(result.transform),
        "exposure_gain": result.rendered.exposure_gain,
    }


def _reference_entry(result: PipelineResult, patch: PatchMeasurement) -> dict[str, Any]:
    region = (
        result.regions[patch.region_index]
        if 0 <= patch.region_index < len(result.regions)
        else None
    )
    return {
        "label": patch.label.value,
        "description": region.description if region else "",
        "box_xyxy": to_jsonable(region.box_xyxy) if region else None,
        "detector_score": region.detector_score if region else None,
        "pixel_count": patch.pixel_count,
        "robust_rgb": to_jsonable(patch.rgb),
        "weight": round(patch.weight, 4),
        "kept": patch.kept,
        "rejection_reason": patch.rejection_reason,
        "used_by_estimator": patch.kept and patch.label.value in result.estimate.used_labels,
    }


def _illuminant_section(result: PipelineResult) -> dict[str, Any]:
    section: dict[str, Any] = to_jsonable(result.estimate)
    if result.estimate.spectrum is not None:
        section["spectrum_wavelengths_nm"] = to_jsonable(result.grid.wavelengths_nm)
    return section


def _camera_section(result: PipelineResult) -> dict[str, Any] | None:
    sensitivity = result.sensitivity
    if sensitivity is None:
        return None
    return {
        "name": sensitivity.name,
        "selection_reason": sensitivity.selection_reason,
        "is_approximate": sensitivity.is_approximate,
    }


def write_review(outcome: ReviewOutcome, directory: Path) -> list[Path]:
    """review.json with every round's evidence and critique, plus the sheet the critic saw."""
    rounds = []
    paths = []
    for number, round_ in enumerate(outcome.rounds, start=1):
        paths.append(write_image8(directory / f"review_round{number}.png", round_.sheet / 255))
        rounds.append(
            {
                "round": number,
                "delivered": number - 1 == outcome.best_index,
                "illuminant": to_jsonable(round_.result.estimate),
                "user_match_trust": [a.trust for a in round_.anchors],
                "scene_context": to_jsonable(round_.context),
                "evidence": to_jsonable(round_.evidence),
                "critique": to_jsonable(round_.critique),
            }
        )
    paths.insert(0, write_report(directory / "review.json", {"rounds": rounds}))
    return paths


def write_report(path: Path, report: dict[str, Any]) -> Path:
    """Write the report as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path
