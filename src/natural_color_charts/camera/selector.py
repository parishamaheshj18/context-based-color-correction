"""Stage 2: choose the camera sensitivity for an image and record why."""

from __future__ import annotations

import dataclasses
import difflib
import logging

from natural_color_charts.camera.repository import (
    SRGB_EQUIVALENT_NAME,
    CameraSensitivityRepository,
    compact_camera_name,
    model_numbers,
)
from natural_color_charts.domain import CameraSensitivity, InputMode, LinearImage

logger = logging.getLogger(__name__)


class CameraSensitivitySelector:
    """Explicit request, then EXIF match, then a mode-dependent fallback."""

    def __init__(
        self,
        repository: CameraSensitivityRepository,
        fuzzy_cutoff: float,
        use_same_make_fallback: bool,
        default_request: str | None = None,
    ) -> None:
        self._repository = repository
        self._default_request = default_request
        self._fuzzy_cutoff = fuzzy_cutoff
        self._use_same_make_fallback = use_same_make_fallback
        self._by_compact = {compact_camera_name(n): n for n in repository.list_cameras()}

    def select(self, image: LinearImage, requested: str | None = None) -> CameraSensitivity | None:
        """Return a sensitivity, or None when only the camera matrix path is possible."""
        requested = requested or self._default_request
        if requested == SRGB_EQUIVALENT_NAME:
            return dataclasses.replace(
                self._repository.srgb_equivalent(), selection_reason="requested generic curves"
            )
        if requested:
            match = self._find(requested)
            if match is None:
                raise LookupError(f"No sensitivity data for requested camera '{requested}'")
            return self._chosen(match, f"requested by user ('{requested}')")
        exif_name = image.metadata.camera_name
        if exif_name and (match := self._find(exif_name)):
            return self._chosen(match, f"EXIF match for '{exif_name}'")
        return self._fallback(image)

    def _find(self, name: str) -> str | None:
        compact = compact_camera_name(name)
        if compact in self._by_compact:
            return self._by_compact[compact]
        # Fuzzy matching tolerates naming differences, never a different model number.
        numbers = model_numbers(compact)
        candidates = [c for c in self._by_compact if model_numbers(c) == numbers]
        close = difflib.get_close_matches(compact, candidates, 1, self._fuzzy_cutoff)
        return self._by_compact[close[0]] if close else None

    def _chosen(self, name: str, reason: str, approximate: bool = False) -> CameraSensitivity:
        logger.info("Camera sensitivity: %s (%s)", name, reason)
        sensitivity = self._repository.get_sensitivity(name)
        return dataclasses.replace(sensitivity, selection_reason=reason, is_approximate=approximate)

    def _fallback(self, image: LinearImage) -> CameraSensitivity | None:
        if image.mode is InputMode.RASTER:
            sensitivity = self._repository.srgb_equivalent()
            return dataclasses.replace(
                sensitivity, selection_reason="display-referred input: linear sRGB as sensor"
            )
        make = image.metadata.make
        if self._use_same_make_fallback and make:
            same_make = self._same_make(make)
            if same_make:
                return self._chosen(same_make, f"closest camera from make '{make}'", True)
        logger.info("No sensitivity data for %s", image.metadata.camera_name)
        return None

    def _same_make(self, make: str) -> str | None:
        prefix = compact_camera_name(make)
        candidates = sorted(c for c in self._by_compact if c.startswith(prefix))
        return self._by_compact[candidates[0]] if candidates else None
