from __future__ import annotations

import base64
import importlib
import sys
import types


def _load_extract_inline_image():
    if "google" not in sys.modules:
        sys.modules["google"] = types.ModuleType("google")
    google_module = sys.modules["google"]
    if "google.genai" not in sys.modules:
        sys.modules["google.genai"] = types.ModuleType("google.genai")
    genai_module = sys.modules["google.genai"]
    setattr(google_module, "genai", genai_module)
    setattr(genai_module, "types", types.SimpleNamespace(GenerateContentResponse=object))

    nano_banana = importlib.import_module("integrations.nano_banana")
    return nano_banana._extract_inline_image


_extract_inline_image = _load_extract_inline_image()


def test_extract_inline_image_from_candidates_snake_case() -> None:
    expected = b"fake-image-bytes"
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Polished image generated."},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(expected).decode("ascii"),
                            }
                        },
                    ]
                }
            }
        ]
    }

    image_bytes, mime_type = _extract_inline_image(payload)

    assert image_bytes == expected
    assert mime_type == "image/png"


def test_extract_inline_image_from_candidates_camel_case_data_url() -> None:
    expected = b"camel-case-image"
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/webp",
                                "data": f"data:image/webp;base64,{base64.b64encode(expected).decode('ascii')}",
                            }
                        }
                    ]
                }
            }
        ]
    }

    image_bytes, mime_type = _extract_inline_image(payload)

    assert image_bytes == expected
    assert mime_type == "image/webp"


def test_extract_inline_image_legacy_top_level_base64() -> None:
    expected = b"legacy-b64-image"
    payload = {
        "image_base64": base64.b64encode(expected).decode("ascii"),
        "image_mime_type": "image/jpeg",
    }

    image_bytes, mime_type = _extract_inline_image(payload)

    assert image_bytes == expected
    assert mime_type == "image/jpeg"


def test_extract_inline_image_invalid_or_missing_returns_none() -> None:
    invalid_payload = {
        "candidates": [
            {"content": {"parts": [{"inline_data": {"mime_type": "image/png", "data": "%%%not-base64%%%"}}]}}
        ]
    }
    missing_payload = {"candidates": [{"content": {"parts": [{"text": "No image here"}]}}]}

    assert _extract_inline_image(invalid_payload) == (None, None)
    assert _extract_inline_image(missing_payload) == (None, None)
