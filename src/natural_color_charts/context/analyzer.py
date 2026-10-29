"""Stage 3: scene context from a vision-language model, with schema validation and retries."""

from __future__ import annotations

import dataclasses
import io
import logging

from PIL import Image
from pydantic import ValidationError

from natural_color_charts.context.base import SceneContextProvider
from natural_color_charts.context.clients import VisionLanguageClient, VlmError
from natural_color_charts.context.schema import SceneAnalysisResponse, response_json_schema
from natural_color_charts.domain import PreviewImage, SceneContext

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
cloud (white or gray). Set usable=false and list concerns when the material is atypical \
or its light differs from the main light: autumn or dry foliage, wet or stained \
concrete, tinted or colored paper, skin lit by colored or neon light, snow in blue shade, \
regions mostly in shadow or reflecting a colored surface. Omit classes not present.

3. Intent. Map the user's request to adaptation "neutral" (accurate, white-balanced \
colors), "preserve_ambiance" (keep some of the light's mood; strength 0-1 is the \
fraction of correction to apply, e.g. 0.6), or "as_shot" (keep the original white \
balance; strength 0). Without a request, use neutral with strength 1.

User request: {intent}

Put a one-paragraph explanation of your judgment in rationale."""


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
        image_png = encode_png(preview, self._preview_px)
        prompt = SCENE_PROMPT.format(intent=intent_text or "(none)")
        schema = response_json_schema()
        for attempt in range(self._max_retries + 1):
            try:
                text = self._client.complete_json(image_png, prompt, schema)
                return SceneAnalysisResponse.model_validate_json(text).to_domain()
            except ValidationError as error:
                logger.warning("VLM response failed validation (attempt %d)", attempt + 1)
                prompt = _with_correction(prompt, error)
            except VlmError as error:
                logger.warning("VLM call failed (attempt %d): %s", attempt + 1, error)
        logger.warning("Falling back to default scene context")
        fallback = self._fallback.analyze(preview, intent_text)
        return _note_fallback(fallback, self._client.name)


def _with_correction(prompt: str, error: ValidationError) -> str:
    issues = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return f"{prompt}\n\nYour previous answer was invalid ({issues}). Return corrected JSON."


def _note_fallback(context: SceneContext, client_name: str) -> SceneContext:
    return dataclasses.replace(
        context, rationale=f"{client_name} unavailable; default context used"
    )
