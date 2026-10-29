"""VisionLanguageClient implementations: a local Ollama model and the hosted Claude API."""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.request
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class VlmError(RuntimeError):
    """The model could not produce a response (network, refusal, or truncation)."""


class VisionLanguageClient(Protocol):
    """Sends one image and a prompt and returns JSON text matching the schema."""

    name: str

    def complete_json(self, image_png: bytes, prompt: str, schema: dict[str, Any]) -> str: ...


class OllamaClient:
    """Local model served by Ollama (e.g. qwen2.5vl), using its JSON-schema `format` option."""

    name = "ollama"

    def __init__(self, host: str, model: str, timeout_s: float) -> None:
        self._url = os.environ.get("OLLAMA_HOST", host).rstrip("/") + "/api/chat"
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
