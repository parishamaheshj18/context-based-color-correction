"""Stage 1: decode images into linear camera RGB."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import tifffile
from numpy.typing import NDArray
from PIL import Image, ImageOps

from natural_color_charts.colorimetry import decode_srgb
from natural_color_charts.domain.image import InputMode, LinearImage
from natural_color_charts.io.metadata import read_exif_metadata

logger = logging.getLogger(__name__)

RAW_SUFFIXES = frozenset(
    {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
)
RASTER_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"})


class ImageLoader(Protocol):
    """Turns an image file into a LinearImage."""

    def supports(self, path: Path) -> bool: ...

    def load(self, path: Path) -> LinearImage: ...


def import_rawpy() -> Any:
    """Import rawpy on first RAW decode; it has no WebAssembly build, so it stays optional."""
    import rawpy

    return rawpy


class RawImageLoader:
    """Decodes RAW files to demosaiced linear camera RGB with no camera processing."""

    def __init__(self, saturation_threshold: float, half_size: bool = False) -> None:
        self._saturation_threshold = saturation_threshold
        self._half_size = half_size

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in RAW_SUFFIXES

    def load(self, path: Path) -> LinearImage:
        rawpy = import_rawpy()
        with rawpy.imread(str(path)) as raw:
            rgb16 = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=False,
                use_auto_wb=False,
                user_wb=[1.0, 1.0, 1.0, 1.0],
                output_color=rawpy.ColorSpace.raw,
                output_bps=16,
                half_size=self._half_size,
            )
            as_shot = _as_shot_white(raw.camera_whitebalance)
            xyz_to_camera = _xyz_to_camera_matrix(raw.rgb_xyz_matrix)
        rgb = rgb16.astype(np.float32) / 65535.0
        metadata = dataclasses.replace(
            read_exif_metadata(path),
            as_shot_white_rgb=as_shot,
            xyz_to_camera_matrix=xyz_to_camera,
        )
        return LinearImage(
            rgb=rgb,
            saturation_mask=_saturation_mask(rgb, self._saturation_threshold),
            metadata=metadata,
            mode=InputMode.RAW,
            source_path=path,
        )


class RasterImageLoader:
    """Loads display-referred images and linearizes them, treating linear sRGB as sensor space."""

    def __init__(self, saturation_threshold: float) -> None:
        self._saturation_threshold = saturation_threshold

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in RASTER_SUFFIXES

    def load(self, path: Path) -> LinearImage:
        encoded = read_encoded_rgb(path)
        return LinearImage(
            rgb=decode_srgb(encoded),
            saturation_mask=_saturation_mask(encoded, self._saturation_threshold),
            metadata=read_exif_metadata(path),
            mode=InputMode.RASTER,
            source_path=path,
        )


class SuffixDispatchingLoader:
    """Delegates to the first loader that supports the file type."""

    def __init__(self, loaders: list[ImageLoader]) -> None:
        self._loaders = loaders

    def supports(self, path: Path) -> bool:
        return any(loader.supports(path) for loader in self._loaders)

    def load(self, path: Path) -> LinearImage:
        for loader in self._loaders:
            if loader.supports(path):
                logger.info("Loading %s with %s", path, type(loader).__name__)
                return loader.load(path)
        raise ValueError(f"Unsupported image type: {path.suffix}")


def read_encoded_rgb(path: Path) -> NDArray[np.float32]:
    """Read an 8- or 16-bit display-referred image as floats in [0, 1]."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        array = np.asarray(tifffile.imread(path))
    elif suffix == ".png":
        # OpenCV keeps 16-bit PNG precision, which Pillow drops for RGB images.
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise ValueError(f"Could not read {path}")
        array = bgr[..., 2::-1] if bgr.ndim == 3 else bgr
    else:
        _register_heif_if_available(path)
        with Image.open(path) as image:
            array = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    array = array[..., :3] if array.ndim == 3 else np.repeat(array[..., None], 3, axis=2)
    scale = 65535.0 if array.dtype == np.uint16 else 255.0
    return (array.astype(np.float32) / scale).astype(np.float32)


def _register_heif_if_available(path: Path) -> None:
    if path.suffix.lower() not in {".heic", ".heif"}:
        return
    try:
        from pillow_heif import register_heif_opener
    except ImportError as error:
        raise ImportError("HEIC support needs the 'heic' extra: pip install .[heic]") from error
    register_heif_opener()


def _saturation_mask(rgb: NDArray[np.float32], threshold: float) -> NDArray[np.bool_]:
    return np.asarray(np.any(rgb >= threshold, axis=2))


def _as_shot_white(multipliers: Sequence[float]) -> NDArray[np.float64] | None:
    # rawpy reports per-channel gains; the as-shot white is their reciprocal.
    gains = np.asarray(multipliers[:3], dtype=np.float64)
    if np.any(gains <= 0):
        return None
    white = 1.0 / gains
    return white / white[1]


def _xyz_to_camera_matrix(matrix: NDArray[np.float32]) -> NDArray[np.float64] | None:
    xyz_to_camera = np.asarray(matrix, dtype=np.float64)[:3]
    if not np.any(xyz_to_camera) or abs(np.linalg.det(xyz_to_camera)) < 1e-9:
        return None
    return xyz_to_camera
