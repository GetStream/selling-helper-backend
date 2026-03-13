"""Nano Banana image polishing integration."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from helpers import get_date_and_time_now
from workflow.events import EventEmitter
from workflow.models import TaskId, TaskProgress


class NanoBananaConfigurationError(RuntimeError):
    """Raised when Nano Banana direct API configuration is missing or invalid."""


class NanoBananaAPIError(RuntimeError):
    """Raised when the Nano Banana API call fails."""


@dataclass(slots=True)
class NanoBananaDirectConfig:
    """Runtime configuration for direct Nano Banana API calls."""

    api_key: str
    model: str = "gemini-3.1-flash-image-preview"
    timeout_seconds: float = 65.0


class NanoBananaClient:
    """Direct SDK client for product-shot image optimization."""

    def __init__(self, event_emitter: EventEmitter) -> None:
        self._event_emitter = event_emitter
        api_key = (
            os.getenv("NANO_BANANA_API_KEY", "").strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
        )
        if not api_key:
            raise NanoBananaConfigurationError(
                "Missing API key. Set NANO_BANANA_API_KEY (or GEMINI_API_KEY)."
            )

        model = os.getenv("NANO_BANANA_MODEL", "gemini-3.1-flash-image-preview").strip()
        timeout_raw = os.getenv("NANO_BANANA_TIMEOUT_SECONDS", "60").strip()
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

    def _get_image_folder(self) -> str:
        """Get the path to the image folder."""
        project_root = Path(__file__).parent.parent.resolve()
        return project_root / "artifacts"

    def _get_image_path(self, image_name: str) -> str:
        """Get the path to the image file. Throws an error if it doesn't exist."""
        image_folder = self._get_image_folder()
        image_path = (image_folder / f"{image_name}.png").resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Source image does not exist: {image_path}")
        return image_path

    def _call_nano_banana(
        self, *, source_bytes: bytes, source_mime: str, prompt: str
    ) -> dict[str, Any]:
        # Initialize the Nano Banana client
        client = genai.Client(
            api_key=self._config.api_key,
            http_options=types.HttpOptions(
                timeout=int(self._config.timeout_seconds * 1000)
            ),
        )

        try:
            # Generate the product shot using nano banana
            response = client.models.generate_content(
                model=self._config.model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=source_bytes, mime_type=source_mime),
                ],
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(
                        aspect_ratio="16:9", image_size="1K"
                    ),
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )
            return response
        except Exception as exc:  # noqa: BLE001
            raise NanoBananaAPIError(f"Nano Banana SDK request failed: {exc}") from exc

    def _save_image_from_response(
        self, response: types.GenerateContentResponse, folder_path: str
    ) -> tuple[str, str]:
        """Save the image from the response to the given path."""
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                image_mime = part.inline_data.mime_type
                product_shot_image_path = (
                    f"{folder_path}/generated_image-{get_date_and_time_now()}.png"
                )
                image.save(product_shot_image_path)
                return product_shot_image_path, image_mime
        raise NanoBananaAPIError("Nano Banana SDK response did not contain image data.")

    async def generate_product_shot(self, *, image_name: str, prompt: str) -> str:
        print(f"[NanoBanana] Product shot from: {image_name} and prompt: {prompt}")

        # Get path for the image
        image_path = self._get_image_path(image_name)

        # Get the bytes and mime type of the image
        with image_path.open("rb") as fh:
            source_bytes = fh.read()
        source_mime = "image/png"

        # Call the Nano Banana API
        response = self._call_nano_banana(
            source_bytes=source_bytes,
            source_mime=source_mime,
            prompt=prompt,
        )

        # Save image from the response
        product_shot_image_path, product_shot_image_mime = (
            self._save_image_from_response(response, self._get_image_folder())
        )

        # Emit the image to the event emitter
        upload_response = await self._event_emitter.emit_polished_image(
            product_shot_image_path, product_shot_image_mime
        )

        await self._event_emitter.emit(
            TaskProgress(
                task_id=TaskId.POLISH_IMAGE,
                progress=100,
                message="Product image was generated successfully",
            )
        )
        print(f"[NanoBanana] Upload response: {upload_response}")

        return "Product image was generated successfully"

    async def wait_until_ready(self, timeout_seconds: float = 30.0) -> bool:
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout_seconds)
            return True
        except TimeoutError:
            return False
