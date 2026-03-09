"""Live Nano Banana integration test that captures full API response.

Required environment variables:
- NANO_BANANA_API_KEY (or GEMINI_API_KEY)

Optional environment variables:
- NANO_BANANA_MODEL (default: gemini-3.1-flash-image-preview)
- NANO_BANANA_TIMEOUT_SECONDS (default: 45)

Run command:
    pytest tests/test_nano_banana_live_full_response.py -s
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def _image_extension_for_mime(mime_type: str | None) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/webp":
        return ".webp"
    if mime_type == "image/gif":
        return ".gif"
    return ".png"


def _extract_inline_image(response: Any) -> tuple[bytes | None, str | None]:
    for part in response.parts or []:
        inline = getattr(part, "inline_data", None)
        if inline and inline.data:
            return inline.data, inline.mime_type
    return None, None


def test_nano_banana_live_captures_full_api_response() -> None:
    api_key = os.getenv("NANO_BANANA_API_KEY") or os.getenv("GEMINI_API_KEY")
    assert api_key, (
        "Missing API key for live Nano Banana test. "
        "Set NANO_BANANA_API_KEY (or GEMINI_API_KEY)."
    )

    model = os.getenv("NANO_BANANA_MODEL", "gemini-3.1-flash-image-preview").strip()
    timeout_seconds = float(os.getenv("NANO_BANANA_TIMEOUT_SECONDS", "45").strip())

    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / "artifacts" / "product-sample.jpeg"
    assert source_path.exists(), f"Expected test image at: {source_path}"
    source_bytes = source_path.read_bytes()
    assert source_bytes, f"Source image is empty: {source_path}"

    prompt = (
        "Transform this single handheld photo into a polished ecommerce product shot. "
        "Keep the exact product identity and proportions. "
        "Use a clean neutral studio background, soft shadow, and balanced product lighting. "
        "Remove distracting background objects and hands while preserving realistic texture. "
        "Output a marketplace-ready hero image."
    )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
    )
    response = client.models.generate_content(
        model=model,
        contents=[
            prompt,
            types.Part.from_bytes(data=source_bytes, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )

    raw_response = response.model_dump(mode="json", exclude_none=False)
    output_text_path = repo_root / "artifacts" / "nano_banana_live_full_response.txt"
    output_text_path.parent.mkdir(parents=True, exist_ok=True)
    output_text_path.write_text(
        (
            f"generated_at_utc: {datetime.now(UTC).isoformat()}\n"
            f"model: {model}\n"
            f"source_image: {source_path}\n"
            "\n=== response_repr ===\n"
            f"{response!r}\n"
            "\n=== response_model_dump_json ===\n"
            f"{json.dumps(raw_response, indent=2, ensure_ascii=False, default=str)}\n"
        ),
        encoding="utf-8",
    )

    image_bytes, image_mime_type = _extract_inline_image(response)
    assert image_bytes, "Response did not include an inline image."
    output_image_path = (
        repo_root
        / "artifacts"
        / f"polished_product_live_full_response{_image_extension_for_mime(image_mime_type)}"
    )
    output_image_path.write_bytes(image_bytes)

    assert output_text_path.exists(), f"Expected response dump at: {output_text_path}"
    assert output_text_path.stat().st_size > 0, f"Response dump file is empty: {output_text_path}"
    assert output_image_path.exists(), f"Expected generated image at: {output_image_path}"
    assert output_image_path.stat().st_size > 0, f"Generated image is empty: {output_image_path}"
    print(f"Saved full Nano Banana response dump: {output_text_path}")
    print(f"Saved generated image: {output_image_path}")
