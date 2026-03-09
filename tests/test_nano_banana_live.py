"""Live Nano Banana integration test.

Required environment variables:
- NANO_BANANA_API_KEY (or GEMINI_API_KEY)

Optional environment variables:
- NANO_BANANA_MODEL (default: gemini-3.1-flash-image-preview)
- NANO_BANANA_TIMEOUT_SECONDS (default: 45)

Run command:
    pytest tests/test_nano_banana_live.py -s
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv

from integrations.nano_banana import NanoBananaClient

load_dotenv()


def test_nano_banana_live_generates_polished_product_shot() -> None:
    api_key = os.getenv("NANO_BANANA_API_KEY") or os.getenv("GEMINI_API_KEY")
    assert api_key, (
        "Missing API key for live Nano Banana test. "
        "Set NANO_BANANA_API_KEY (or GEMINI_API_KEY)."
    )

    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / "artifacts" / "product-sample.jpeg"
    assert source_path.exists(), f"Expected test image at: {source_path}"

    prompt = (
        "Transform this single handheld photo into a polished ecommerce product shot. "
        "Keep the exact product identity and proportions. "
        "Use a clean neutral studio background, soft shadow, and balanced product lighting. "
        "Remove distracting background objects and hands while preserving realistic texture. "
        "Output a marketplace-ready hero image."
    )
    client = NanoBananaClient(workflow_emitter=None)
    result = client.generate_product_shot(image_path=str(source_path), prompt=prompt)
    image_b64 = result.get("image_base64")
    assert image_b64, "Nano Banana response did not include image_base64."
    image_bytes = base64.b64decode(image_b64)
    assert image_bytes, "Decoded Nano Banana image bytes are empty."

    mime_type = result.get("image_mime_type", "image/png")
    extension = ".png"
    if mime_type == "image/jpeg":
        extension = ".jpg"
    elif mime_type == "image/webp":
        extension = ".webp"
    elif mime_type == "image/gif":
        extension = ".gif"

    output_path = repo_root / "artifacts" / f"polished_product_live{extension}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)

    assert output_path.exists(), f"Expected output image at: {output_path}"
    assert output_path.stat().st_size > 0, f"Output image is empty: {output_path}"
    print(f"Saved polished product image: {output_path}")
