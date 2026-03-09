"""Live Stream Chat event test for polished image payloads.

Required environment variables:
- STREAM_API_KEY
- STREAM_API_SECRET

Run command:
    pytest tests/test_stream_polished_image_live.py -s
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from uuid import uuid4

import pytest
from dotenv import load_dotenv
from stream_chat import StreamChat

load_dotenv()


def _build_channel_id(call_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in call_id)
    return f"workflow-{safe}"


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        pytest.skip(f"Missing required environment variable for live Stream test: {name}")
    return value


def test_stream_live_sends_polished_image_event_with_base64_payload() -> None:
    api_key = _require_env("STREAM_API_KEY")
    api_secret = _require_env("STREAM_API_SECRET")

    call_id = "call-123"
    channel_type = os.getenv("STREAM_WORKFLOW_CHANNEL_TYPE", "messaging").strip() or "messaging"
    sender_user_id = (
        os.getenv("STREAM_WORKFLOW_SENDER_USER_ID", "workflow-server").strip() or "workflow-server"
    )
    channel_id = _build_channel_id(call_id)

    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / "artifacts" / "polished_product_old-2.png"
    assert image_path.exists(), f"Expected image fixture at: {image_path}"
    image_bytes = image_path.read_bytes()
    assert image_bytes, f"Image fixture is empty: {image_path}"
    image_data_base64 = base64.b64encode(image_bytes).decode("ascii")
    # Stream custom events have a very small custom data size cap (2KB).
    # We send a deterministic preview segment and include the full encoded length.
    preview_size = 1024
    image_data_base64_preview = image_data_base64[:preview_size]
    assert image_data_base64_preview, "Expected non-empty base64 image preview."

    client = StreamChat(api_key=api_key, api_secret=api_secret)
    channel = client.channel(channel_type, channel_id, data={"name": f"Workflow {channel_id}"})
    client.upsert_user({"id": sender_user_id, "name": "Workflow Server"})
    channel.create(sender_user_id)

    payload = {
        "type": "polished-image",
        "workflowId": call_id,
        "imageMimeType": "image/png",
        "imageDataBase64Preview": image_data_base64_preview,
        "imageDataBase64Length": len(image_data_base64),
        "testEventId": f"evt-{uuid4().hex}",
    }
    response = channel.send_event(payload, sender_user_id)

    assert response is not None
    assert isinstance(response, dict)
    sent_event = response.get("event", response)
    assert isinstance(sent_event, dict)
    assert sent_event.get("type") == "polished-image"
    assert sent_event.get("workflowId") == call_id
    assert sent_event.get("imageMimeType") == "image/png"
    assert sent_event.get("imageDataBase64Preview") == image_data_base64_preview
    assert sent_event.get("imageDataBase64Length") == len(image_data_base64)
