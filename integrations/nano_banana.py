"""Nano Banana image polishing integration."""

from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from workflow.events import WorkflowEventEmitter


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


class NanoBananaClient:
    """Direct SDK client for product-shot image optimization."""

    def __init__(self, workflow_emitter: WorkflowEventEmitter) -> None:
        self._workflow_emitter = workflow_emitter
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

        self._config = NanoBananaDirectConfig(
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )

    def generate_product_shot(self, *, image_path: str, prompt: str) -> dict[str, Any]:
        print(f"[NanoBanana] Generating product shot with image path: {image_path} and prompt: {prompt}")
        # Build the path as <project-root>/artifacts/<image_path>
        project_root = Path(__file__).parent.parent.resolve()
        source = (project_root / "artifacts" / image_path).resolve()
        print(f"[NanoBanana] Image source: {source}")
        if not source.exists():
            raise FileNotFoundError(f"Source image does not exist: {source}")

        with source.open("rb") as fh:
            source_bytes = fh.read()
        source_mime = "image/png"
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
            print(f"[NanoBanana] Response: {response}")
        except Exception as exc:  # noqa: BLE001
            raise NanoBananaAPIError(f"Nano Banana SDK request failed: {exc}") from exc

        image_bytes, image_mime, revised_prompt = _extract_sdk_generation(response)
        # print(f"[NanoBanana] Image bytes: {image_bytes}")
        print(f"[NanoBanana] Image mime: {image_mime}")
        print(f"[NanoBanana] Revised prompt: {revised_prompt}")
        if not image_bytes:
            raise NanoBananaAPIError("Nano Banana SDK response did not contain image data.")

        raw_response = response.model_dump(mode="json", exclude_none=True)

        image_url = (
            raw_response.get("image_url")
            or raw_response.get("url")
            or raw_response.get("output_url")
            or ""
        )
        print(f"[NanoBananaAdapter] Image URL: {image_url}")
        image_bytes, image_mime = _extract_inline_image(response)
        if not image_url and not image_bytes:
            raise ValueError("Nano Banana response did not include an image URL or inline image.")

        self._workflow_emitter.emit_polished_image(image_bytes, image_mime or "image/png")
        print("[NanoBanana] Creating Polished image result")
        # result = PolishedImageResult(
        #     image_url=image_url,
        #     revised_prompt=raw_response.get("revised_prompt") or prompt,
        #     image_bytes=image_bytes,
        #     image_mime_type=image_mime or "image/png",
        #     raw=raw_response,
        # )

        # print(f"[NanoBanana] Polished image result: {result}")
        # self._artifacts.polished_image_url = result.image_url
        # print(f"[NanoBanana] Polished image URL: {self._artifacts.polished_image_url}")

        return_object = {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "image_mime_type": image_mime or "image/png",
            "revised_prompt": revised_prompt or prompt,
            "raw_response": response.model_dump(mode="json", exclude_none=True),
        }
        print(f"[NanoBanana] Return object: {return_object}")

        # output_path = project_root / "artifacts" / f"polished_product_live{_guess_mime_type(image_mime)}"
        # output_path.parent.mkdir(parents=True, exist_ok=True)
        # output_path.write_bytes(image_bytes)
        # print(f"[NanoBanana] Saved polished image to: {output_path}")

        # image_url = str(output_path.resolve().as_uri())

        # return_object = {
        #     "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        #     "image_mime_type": image_mime or "image/png",
        #     "revised_prompt": revised_prompt or prompt,
        #     "image_url": image_url,
        #     "image_path": str(output_path),
        #     "raw_response": raw_response,
        # }
        # print(f"[NanoBanana] Return object: {return_object}")
        return return_object

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


def _extract_inline_image(response: Any) -> tuple[bytes | None, str | None]:
    for part in response.parts:
        if part.text is not None:
            print(f"[NanoBanana] Part.text: ${part.text}")
        elif part.inline_data is not None:
            # image_bytes = _decode_base64_image(part.inline_data.data)
            image_bytes = part.inline_data.data
            mime_type = part.inline_data.mime_type
            image = part.as_image()
            image.save("generated_image.png")
            return image_bytes, mime_type
    return None, None


def _decode_base64_image(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None

    encoded = value.strip()
    if not encoded:
        return None

    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1].strip()

    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        try:
            decoded = base64.b64decode(encoded)
        except (binascii.Error, ValueError, TypeError):
            return None

    return decoded or None

