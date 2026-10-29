"""EXIF metadata extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import exifread

from natural_color_charts.domain.image import ImageMetadata

logger = logging.getLogger(__name__)


def read_exif_metadata(path: Path) -> ImageMetadata:
    """Read camera make, model, and exposure settings; missing tags become None."""
    try:
        with path.open("rb") as handle:
            tags = exifread.process_file(handle, details=False)
    except Exception:  # exifread raises many unrelated types on unsupported files
        logger.warning("Could not read EXIF from %s", path)
        return ImageMetadata()
    return ImageMetadata(
        make=_text(tags, "Image Make"),
        model=_text(tags, "Image Model"),
        exposure_time_s=_ratio(tags, "EXIF ExposureTime"),
        iso=_ratio(tags, "EXIF ISOSpeedRatings"),
        f_number=_ratio(tags, "EXIF FNumber"),
        capture_time=_text(tags, "EXIF DateTimeOriginal"),
    )


def _text(tags: dict[str, Any], key: str) -> str | None:
    value = tags.get(key)
    text = str(value).strip() if value is not None else ""
    return text or None


def _ratio(tags: dict[str, Any], key: str) -> float | None:
    value = tags.get(key)
    if value is None or not value.values:
        return None
    first = value.values[0]
    try:
        return float(first.num) / float(first.den) if hasattr(first, "num") else float(first)
    except (ZeroDivisionError, ValueError, TypeError):
        return None
