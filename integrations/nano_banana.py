"""Nano Banana image polishing integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(slots=True)
class PolishedImageResult:
    """Normalized output from Nano Banana tooling."""

    image_url: str
    revised_prompt: str | None = None
    raw: dict[str, Any] | None = None


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
        if not image_url:
            raise ValueError("Nano Banana response did not include an image URL.")

        return PolishedImageResult(
            image_url=image_url,
            revised_prompt=response.get("revised_prompt"),
            raw=response,
        )

