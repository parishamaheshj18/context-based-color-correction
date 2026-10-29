"""Stage 10 visualizations: reference overlay and before/after comparison."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless rendering, e.g. on cluster nodes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from natural_color_charts.domain import ReferenceClass
from natural_color_charts.io.preview import resize_to_long_edge
from natural_color_charts.io.writers import to_uint8
from natural_color_charts.pipeline import PipelineResult
from natural_color_charts.spectral.illuminants import DaylightBasis, normalize_spd

LABEL_COLORS: dict[ReferenceClass, tuple[int, int, int]] = {
    ReferenceClass.FOLIAGE: (40, 200, 60),
    ReferenceClass.SKIN: (240, 140, 60),
    ReferenceClass.CONCRETE: (150, 150, 150),
    ReferenceClass.PAPER: (60, 120, 255),
    ReferenceClass.SNOW: (0, 220, 220),
    ReferenceClass.CLOUD: (200, 80, 220),
}
REJECTED_COLOR = (230, 30, 30)


def render_overlay(result: PipelineResult, alpha_max: float) -> NDArray[np.uint8]:
    """Preview with each reference mask tinted by label; opacity is proportional to weight."""
    overlay = result.preview.astype(np.float32)
    height, width = overlay.shape[:2]
    max_weight = max((p.weight for p in result.patches if p.kept), default=1.0) or 1.0
    for patch in result.patches:
        if not 0 <= patch.region_index < len(result.regions):
            continue
        mask = cv2.resize(
            result.regions[patch.region_index].mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        color = np.array(LABEL_COLORS[patch.label] if patch.kept else REJECTED_COLOR, np.float32)
        alpha = alpha_max * (patch.weight / max_weight if patch.kept else 0.5)
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
        _outline(overlay, mask, color)
    return np.asarray(np.clip(overlay, 0, 255), dtype=np.uint8)


def _outline(
    image: NDArray[np.float32], mask: NDArray[np.bool_], color: NDArray[np.float32]
) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, color.tolist(), 1)


def save_comparison(result: PipelineResult, path: Path) -> Path:
    """Before/after side by side, plus the estimated illuminant spectrum against D65."""
    after = resize_to_long_edge(result.rendered.encoded_srgb, max(result.preview.shape[:2]))
    has_spectrum = result.estimate.spectrum is not None
    figure, axes = plt.subplots(
        1, 3 if has_spectrum else 2, figsize=(15 if has_spectrum else 10, 4.5)
    )
    axes[0].imshow(result.preview)
    axes[0].set_title("Before (camera as-shot preview)")
    axes[1].imshow(to_uint8(after))
    axes[1].set_title(
        f"After ({result.estimate.method}, D={result.transform.adaptation_degree:.2f})"
    )
    for axis in axes[:2]:
        axis.axis("off")
    if has_spectrum:
        _plot_spectrum(axes[2], result)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _plot_spectrum(axis: Axes, result: PipelineResult) -> None:
    grid = result.grid
    wavelengths = grid.wavelengths_nm
    assert result.estimate.spectrum is not None
    estimate = normalize_spd(np.asarray(result.estimate.spectrum), grid)
    axis.plot(wavelengths, estimate, color="#d9480f", label=_spectrum_label(result))
    axis.plot(
        wavelengths,
        DaylightBasis(grid).spd(6504),
        color="#1c7ed6",
        linestyle="--",
        label="D65 target",
    )
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Relative power (1 at 560 nm)")
    axis.set_title("Estimated illuminant")
    axis.legend()
    axis.grid(alpha=0.3)


def _spectrum_label(result: PipelineResult) -> str:
    estimate = result.estimate
    cct = f" {estimate.cct_k:.0f} K" if estimate.cct_k else ""
    spread = f" ± {estimate.cct_uncertainty_k:.0f} K" if estimate.cct_uncertainty_k else ""
    return f"{estimate.family or 'estimate'}{cct}{spread}"
