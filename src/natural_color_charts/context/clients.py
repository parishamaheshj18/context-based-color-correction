"""VisionLanguageClient implementations: a local Ollama model and the hosted Claude API."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class VlmError(RuntimeError):
    """The model could not produce a response (network, refusal, or truncation)."""


class VisionLanguageClient(Protocol):
    """Sends one image and a prompt and returns JSON text matching the schema."""

    name: str

    def complete_json(self, image_png: bytes, prompt: str, schema: dict[str, Any]) -> str: ...


class CachedVisionLanguageClient:
    """Stores each answer on disk keyed by model, image, prompt, and schema.

    Re-running a showcase or benchmark then costs nothing, and the exact model output can be
    committed and shown.
    """

    def __init__(self, inner: VisionLanguageClient | None, model: str, cache_dir: Path) -> None:
        self._inner = inner
        self._model = model
        self._cache_dir = cache_dir
        self.name = inner.name if inner else "cache"

    def cache_path(self, image_png: bytes, prompt: str, schema: dict[str, Any]) -> Path:
        digest = hashlib.sha256()
        for part in (
            self._model.encode(),
            image_png,
            prompt.encode(),
            json.dumps(schema, sort_keys=True).encode(),
        ):
            digest.update(hashlib.sha256(part).digest())
        return self._cache_dir / f"{digest.hexdigest()[:24]}.json"

    def complete_json(self, image_png: bytes, prompt: str, schema: dict[str, Any]) -> str:
        path = self.cache_path(image_png, prompt, schema)
        if path.exists():
            return str(json.loads(path.read_text())["response"])
        if self._inner is None:
            raise VlmError(f"No cached answer at {path} and no live client configured")
        response = self._inner.complete_json(image_png, prompt, schema)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self._model, "response": response}, indent=2))
        return response


class OllamaClient:
    """Local model served by Ollama (e.g. qwen2.5vl), using its JSON-schema `format` option."""

    name = "ollama"

    def __init__(self, host: str, model: str, timeout_s: float) -> None:
        self._url = host.rstrip("/") + "/api/chat"
        self._model = model
        self._timeout_s = timeout_s

    def complete_json(self, image_png: bytes, prompt: str, schema: dict[str, Any]) -> str:
        body = {
            "model": self._model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_png).decode("ascii")],
                }
            ],
        }
        request = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                payload = json.loads(response.read())
        except OSError as error:
            raise VlmError(f"Ollama request failed: {error}") from error
        return str(payload["message"]["content"])


class AnthropicClient:
    """Claude through the Anthropic SDK with JSON-schema structured outputs.

    Requires the 'anthropic' extra. Credentials come from ANTHROPIC_API_KEY or an
    `ant auth login` profile.
    """

    name = "anthropic"

    def __init__(self, model: str, max_tokens: int) -> None:
        try:
            import anthropic
        except ImportError as error:
            raise ImportError("The Claude client needs: pip install '.[anthropic]'") from error
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def complete_json(self, image_png: bytes, prompt: str, schema: dict[str, Any]) -> str:
        import anthropic

        image = base64.standard_b64encode(image_png).decode("utf-8")
        try:
            # Server-side fallback reruns the request on another model if this one declines.
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as error:
            raise VlmError(f"Claude request failed: {error}") from error
        if response.stop_reason in {"refusal", "max_tokens"}:
            raise VlmError(f"Claude stopped with {response.stop_reason}")
        return next(block.text for block in response.content if block.type == "text")
