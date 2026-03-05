"""Nano Banana image polishing integration."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from google import genai
from google.genai import types


@dataclass(slots=True)
class PolishedImageResult:
    """Normalized output from Nano Banana tooling."""

    image_url: str
    revised_prompt: str | None = None
    image_bytes: bytes | None = None
    image_mime_type: str | None = None
    raw: dict[str, Any] | None = None


class NanoBananaConfigurationError(RuntimeError):
    """Raised when Nano Banana direct API configuration is missing or invalid."""


class NanoBananaAPIError(RuntimeError):
    """Raised when the Nano Banana API call fails."""


@dataclass(slots=True)
class NanoBananaDirectConfig:
    """Runtime configuration for direct Nano Banana API calls."""

    api_key: str
    model: str = "gemini-3.1-flash-image-preview"
    timeout_seconds: float = 45.0


class NanoBananaDirectClient:
    """Direct SDK client for product-shot image optimization."""

    def __init__(self, config: NanoBananaDirectConfig) -> None:
        self._config = config

    @classmethod
    def from_env(cls) -> "NanoBananaDirectClient":
        api_key = (
            os.getenv("NANO_BANANA_API_KEY", "").strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
        )
        if not api_key:
            raise NanoBananaConfigurationError(
                "Missing API key. Set NANO_BANANA_API_KEY (or GEMINI_API_KEY)."
            )

        model = os.getenv("NANO_BANANA_MODEL", "gemini-3.1-flash-image-preview").strip()
        timeout_raw = os.getenv("NANO_BANANA_TIMEOUT_SECONDS", "45").strip()
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise NanoBananaConfigurationError(
                "NANO_BANANA_TIMEOUT_SECONDS must be a number."
            ) from exc

        return cls(
            NanoBananaDirectConfig(
                api_key=api_key,
                model=model,
                timeout_seconds=timeout_seconds,
            )
        )

    def generate_product_shot(self, *, image_path: str, prompt: str) -> dict[str, Any]:
        source = Path(image_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source image does not exist: {source}")

        with source.open("rb") as fh:
            source_bytes = fh.read()
        source_mime = _guess_mime_type(source)
        client = genai.Client(
            api_key=self._config.api_key,
            http_options=types.HttpOptions(timeout=int(self._config.timeout_seconds * 1000)),
        )

        try:
            response = client.models.generate_content(
                model=self._config.model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=source_bytes, mime_type=source_mime),
                ],
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
            )
        except Exception as exc:  # noqa: BLE001
            raise NanoBananaAPIError(f"Nano Banana SDK request failed: {exc}") from exc

        image_bytes, image_mime, revised_prompt = _extract_sdk_generation(response)
        if not image_bytes:
            raise NanoBananaAPIError("Nano Banana SDK response did not contain image data.")

        return {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "image_mime_type": image_mime or "image/png",
            "revised_prompt": revised_prompt or prompt,
            "raw_response": response.model_dump(mode="json", exclude_none=True),
        }


class NanoBananaAdapter:
    """Adapter around MCP tool-calling for polished product image generation."""

    def __init__(
        self,
        tool_caller: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
        tool_name: str = "nano_banana_polish",
    ) -> None:
        self._tool_caller = tool_caller
        self._tool_name = tool_name

    async def polish_image(
        self,
        *,
        image_path: str,
        prompt: str,
    ) -> PolishedImageResult:
        payload = {"image_path": image_path, "prompt": prompt}
        response = await self._tool_caller(self._tool_name, payload)

        image_url = (
            response.get("image_url")
            or response.get("url")
            or response.get("output_url")
            or ""
        )
        image_bytes, image_mime = _extract_inline_image(response)
        if not image_url and not image_bytes:
            raise ValueError("Nano Banana response did not include an image URL or inline image.")

        return PolishedImageResult(
            image_url=image_url,
            revised_prompt=response.get("revised_prompt"),
            image_bytes=image_bytes,
            image_mime_type=image_mime,
            raw=response,
        )


def _guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/png"


def _extract_sdk_generation(
    response: types.GenerateContentResponse,
) -> tuple[bytes | None, str | None, str | None]:
    revised_prompt: str | None = None
    for part in response.parts or []:
        inline = getattr(part, "inline_data", None)
        if inline and inline.data:
            return inline.data, inline.mime_type, revised_prompt
        if getattr(part, "text", None) and revised_prompt is None:
            revised_prompt = part.text
    return None, None, revised_prompt


def _extract_inline_image(payload: dict[str, Any]) -> tuple[bytes | None, str | None]:
    b64_value = (
        payload.get("image_base64")
        or payload.get("image_b64")
        or payload.get("b64_json")
        or payload.get("data")
    )
    if b64_value:
        try:
            image_bytes = base64.b64decode(b64_value)
        except (ValueError, TypeError):
            image_bytes = None
        if image_bytes:
            return image_bytes, payload.get("image_mime_type") or payload.get("mime_type")

    raw_bytes = payload.get("image_bytes")
    if isinstance(raw_bytes, bytes):
        return raw_bytes, payload.get("image_mime_type") or payload.get("mime_type")

    return None, None

