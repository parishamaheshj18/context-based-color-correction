"""Writers for encoded output images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from numpy.typing import NDArray
from PIL import Image


def to_uint8(encoded: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Quantize encoded values in [0, 1] to 8 bits."""
    return np.asarray(np.round(np.clip(encoded, 0.0, 1.0) * 255.0), dtype=np.uint8)


def to_uint16(encoded: NDArray[np.floating]) -> NDArray[np.uint16]:
    """Quantize encoded values in [0, 1] to 16 bits."""
    return np.asarray(np.round(np.clip(encoded, 0.0, 1.0) * 65535.0), dtype=np.uint16)


def write_tiff16(path: Path, encoded: NDArray[np.floating]) -> Path:
    """Write a 16-bit RGB TIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, to_uint16(encoded), photometric="rgb")
    return path


def write_image8(path: Path, encoded: NDArray[np.floating], jpeg_quality: int = 95) -> Path:
    """Write an 8-bit PNG or JPEG depending on the file suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(to_uint8(encoded))
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=jpeg_quality)
    else:
        image.save(path)
    return path
