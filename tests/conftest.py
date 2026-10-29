from __future__ import annotations

import numpy as np
import pytest

from natural_color_charts.config import AppConfig, load_config
from natural_color_charts.domain import ImageMetadata, InputMode, LinearImage


def make_image(
    rgb: np.ndarray,
    mode: InputMode = InputMode.RASTER,
    metadata: ImageMetadata | None = None,
) -> LinearImage:
    rgb = np.asarray(rgb, dtype=np.float32)
    return LinearImage(
        rgb=rgb,
        saturation_mask=np.zeros(rgb.shape[:2], dtype=bool),
        metadata=metadata or ImageMetadata(),
        mode=mode,
    )


def make_cast_image(white: np.ndarray, size: int = 64, seed: int = 0) -> LinearImage:
    """Random gray-world-balanced reflectances lit by a colored illuminant."""
    rng = np.random.default_rng(seed)
    reflectance = rng.uniform(0.05, 0.6, size=(size, size, 3))
    reflectance /= reflectance.reshape(-1, 3).mean(axis=0)
    return make_image(0.2 * reflectance * np.asarray(white))


@pytest.fixture(scope="session")
def config() -> AppConfig:
    return load_config()
