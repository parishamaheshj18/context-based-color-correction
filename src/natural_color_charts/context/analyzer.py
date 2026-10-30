"""Stage 3: scene context from a vision-language model, with schema validation and retries."""

from __future__ import annotations

import dataclasses
import io
import logging
from typing import Any, TypeVar

from PIL import Image
from pydantic import BaseModel, ValidationError

from natural_color_charts.context.base import SceneContextProvider
from natural_color_charts.context.clients import VisionLanguageClient, VlmError
from natural_color_charts.context.schema import (
    RevisionResponse,
    SceneAnalysisResponse,
    response_json_schema,
    revision_json_schema,
)
from natural_color_charts.domain import PreviewImage, SceneContext

ResponseT = TypeVar("ResponseT", bound=BaseModel)

logger = logging.getLogger(__name__)

SCENE_PROMPT = """\
You are helping a color-correction system that uses everyday materials in a photo as if \
they were patches of a color chart. Analyze the attached photo and return JSON only.

1. Lighting. Describe environment, time of day, weather, and visible light sources. Give \
illuminant hypotheses as priors, not final answers: "daylight" (CIE daylight, roughly \
4000-25000 K) or "blackbody" (tungsten/halogen/candle, roughly 1800-6500 K), each with a \
CCT range in kelvin and a confidence from 0 to 1. Judge the light from scene content \
(lamps, windows, sky, shadows), not from the overall color cast, which may be wrong.

2. Reference materials. For each of these classes that is visible, add one candidate: \
foliage (healthy green leaves or grass), skin, concrete (gray, dry), paper (white), snow, \
cloud (white or gray), neutral_surface (white or gray man-made surfaces that are not \
paper: plastic, painted walls, laminate, ceramic, white printed areas). Set usable=false \
and list concerns when the material is atypical or its light differs from the main light: \
autumn or dry foliage, wet or stained concrete, tinted or colored paper, skin lit by \
colored or neon light, snow in blue shade, cream, beige or tinted surfaces, glossy \
reflections, regions mostly in shadow or reflecting a colored surface. Omit classes not \
present. List each distinct surface as its own candidate. For each usable candidate give \
up to 3 boxes [x0, y0, x1, y1] in coordinates normalized to the image (0,0 = top left, \
1,1 = bottom right), each tightly around a flat, evenly lit area of that material that \
avoids edges, highlights, deep shadows, and other objects. Use [] when it is not usable.

3. Intent. Map the user's request to adaptation "neutral" (accurate, white-balanced \
colors), "preserve_ambiance" (keep some of the light's mood; strength 0-1 is the \
fraction of correction to apply, e.g. 0.6), or "as_shot" (keep the original white \
balance; strength 0). Without a request, use neutral with strength 1.

User request: {intent}

Put a one-paragraph explanation of your judgment in rationale."""

REVISION_PROMPT = """\
{scene_prompt}

You analyzed this photo before and the system corrected it with your answer. Your previous \
answer was:
{previous}

{feedback}

Revise your answer to fix what went wrong: drop, add, or re-box references, change their \
confidence, or change the illuminant hypotheses. A material whose measured color is far \
outside its typical spread is either not that material or not lit by the main light. The \
reviewer can be wrong too: keep what the measurements support, and explain in rationale \
where you disagree.

{match_note}"""

MATCH_NOTE = """\
The user made {count} color match(es), U1 to U{count}: each is a pixel they say truly has the \
color of a chart patch. Users make mistakes (a cream wall matched to white, a patch in shade \
or colored light, a mis-click). For each match give trust 1 when it agrees with the scene \
and the other references, lower when doubtful, or 0 to ignore it, with a reason."""

NO_MATCH_NOTE = "The user made no color matches; return user_match_decisions as []."


def encode_png(preview: PreviewImage, long_edge_px: int) -> bytes:
    """PNG bytes of the preview, downscaled to long_edge_px."""
    image = Image.fromarray(preview)
    image.thumbnail((long_edge_px, long_edge_px))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class VlmSceneContextAnalyzer:
    """Asks the VLM for structured scene context; falls back if it fails repeatedly."""

    def __init__(
        self,
        client: VisionLanguageClient,
        fallback: SceneContextProvider,
        max_retries: int,
        preview_long_edge_px: int,
    ) -> None:
        self._client = client
        self._fallback = fallback
        self._max_retries = max_retries
        self._preview_px = preview_long_edge_px

    def analyze(self, preview: PreviewImage, intent_text: str | None) -> SceneContext:
        prompt = SCENE_PROMPT.format(intent=intent_text or "(none)")
        response = request_validated(
            self._client,
            encode_png(preview, self._preview_px),
            prompt,
            response_json_schema(),
            SceneAnalysisResponse,
            self._max_retries,
        )
        if response is not None:
            return response.to_domain()
        logger.warning("Falling back to default scene context")
        fallback = self._fallback.analyze(preview, intent_text)
        return _note_fallback(fallback, self._client.name)

    def revise(
        self,
        preview: PreviewImage,
        intent_text: str | None,
        previous: SceneContext,
        feedback: str,
        match_count: int,
    ) -> tuple[SceneContext, dict[int, float]] | None:
        """A revised context and per-match trust after review feedback; None if the VLM fails."""
        note = MATCH_NOTE.format(count=match_count) if match_count else NO_MATCH_NOTE
        prompt = REVISION_PROMPT.format(
            scene_prompt=SCENE_PROMPT.format(intent=intent_text or "(none)"),
            previous=SceneAnalysisResponse.from_domain(previous).model_dump_json(),
            feedback=feedback,
            match_note=note,
        )
        response = request_validated(
            self._client,
            encode_png(preview, self._preview_px),
            prompt,
            revision_json_schema(),
            RevisionResponse,
            self._max_retries,
        )
        if response is None:
            return None
        trust = {i: t for i, t in response.match_trust().items() if 0 <= i < match_count}
        return response.to_domain(), trust


def request_validated(
    client: VisionLanguageClient,
    image_png: bytes,
    prompt: str,
    schema: dict[str, Any],
    response_type: type[ResponseT],
    max_retries: int,
) -> ResponseT | None:
    """Ask until the answer validates, telling the model what was wrong; None if it never does."""
    for attempt in range(max_retries + 1):
        try:
            text = client.complete_json(image_png, prompt, schema)
            return response_type.model_validate_json(text)
        except ValidationError as error:
            logger.warning("VLM response failed validation (attempt %d)", attempt + 1)
            prompt = _with_correction(prompt, error)
        except VlmError as error:
            logger.warning("VLM call failed (attempt %d): %s", attempt + 1, error)
    return None


def _with_correction(prompt: str, error: ValidationError) -> str:
    issues = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return f"{prompt}\n\nYour previous answer was invalid ({issues}). Return corrected JSON."


def _note_fallback(context: SceneContext, client_name: str) -> SceneContext:
    return dataclasses.replace(
        context, rationale=f"{client_name} unavailable; default context used"
    )
