"""EXIF metadata extraction."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import exifread

from natural_color_charts.domain.image import ImageMetadata

logger = logging.getLogger(__name__)


def read_exif_metadata(path: Path) -> ImageMetadata:
    """Read camera make, model, and exposure settings; missing tags become None.

    exifread cannot parse CR3; exiftool, when installed, fills that gap.
    """
    metadata = _read_with_exifread(path)
    if metadata.make is None and shutil.which("exiftool"):
        return _read_with_exiftool(path)
    return metadata


def _read_with_exiftool(path: Path) -> ImageMetadata:
    fields = ["Make", "Model", "ExposureTime", "ISO", "FNumber", "DateTimeOriginal"]
    command = ["exiftool", "-j", "-n", *(f"-{field}" for field in fields), str(path)]
    try:
        output = subprocess.run(command, capture_output=True, check=True, timeout=30).stdout
        tags = json.loads(output)[0]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        logger.warning("exiftool could not read %s", path)
        return ImageMetadata()
    return ImageMetadata(
        make=_optional_text(tags.get("Make")),
        model=_optional_text(tags.get("Model")),
        exposure_time_s=_optional_float(tags.get("ExposureTime")),
        iso=_optional_float(tags.get("ISO")),
        f_number=_optional_float(tags.get("FNumber")),
        capture_time=_optional_text(tags.get("DateTimeOriginal")),
    )


def _optional_text(value: Any) -> str | None:
    return str(value).strip() or None if value is not None else None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _read_with_exifread(path: Path) -> ImageMetadata:
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
