"""Live Stream Chat test that sends a message with an image attachment.

Required environment variables:
- STREAM_API_KEY
- STREAM_API_SECRET

Run command:
    pytest tests/test_stream_chat_message_attachment_live.py -s
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import requests
from dotenv import load_dotenv
from stream_chat import StreamChat

load_dotenv()

CHANNEL_TYPE = "messaging"
CHANNEL_ID = "workflow-7430192471"
SENDER_USER_ID = "workflow-server"
SENDER_USER = {"id": SENDER_USER_ID, "name": "Workflow Server"}


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        pytest.skip(f"Missing required environment variable for live Stream test: {name}")
    return value


def _extract_uploaded_asset_url(upload_response: dict[str, Any]) -> str | None:
    direct_url = upload_response.get("file") or upload_response.get("url")
    if isinstance(direct_url, str) and direct_url.strip():
        return direct_url

    nested_file = upload_response.get("file")
    if isinstance(nested_file, dict):
        for key in ("url", "asset_url", "image_url"):
            value = nested_file.get(key)
            if isinstance(value, str) and value.strip():
                return value

    return None


def test_stream_live_sends_chat_message_with_image_attachment() -> None:
    api_key = _require_env("STREAM_API_KEY")
    api_secret = _require_env("STREAM_API_SECRET")

    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / "artifacts" / "polished_product_old-2.png"
    assert image_path.exists(), f"Expected image fixture at: {image_path}"
    assert image_path.stat().st_size > 0, f"Image fixture is empty: {image_path}"

    client = StreamChat(api_key=api_key, api_secret=api_secret)
    channel = client.channel(CHANNEL_TYPE, CHANNEL_ID, data={"name": "Workflow Image Attachment Live"})
    try:
        client.upsert_user(SENDER_USER)
        channel.create(SENDER_USER_ID)
        upload_response = channel.send_image(
            str(image_path),
            image_path.name,
            SENDER_USER,
            content_type="image/png",
        )
    except requests.RequestException as exc:
        pytest.skip(f"Live Stream test could not reach Stream API: {exc}")

    uploaded_asset_url = _extract_uploaded_asset_url(upload_response)
    assert uploaded_asset_url, f"Could not determine uploaded image URL from response: {upload_response}"

    message = {
        "text": "Live test message with polished product image attachment.",
        "attachments": [
            {
                "type": "image",
                "asset_url": uploaded_asset_url,
                "image_url": uploaded_asset_url,
                "title": image_path.name,
            }
        ],
    }
    try:
        send_response = channel.send_message(message, SENDER_USER_ID)
    except requests.RequestException as exc:
        pytest.skip(f"Live Stream test could not send message due to network error: {exc}")

    assert isinstance(send_response, dict)
    sent_message = send_response.get("message", send_response)
    assert isinstance(sent_message, dict)
    assert sent_message.get("id"), f"Expected sent message id in response: {send_response}"
    attachments = sent_message.get("attachments") or []
    assert attachments, f"Expected message attachments in response: {send_response}"
    first_attachment = attachments[0]
    assert isinstance(first_attachment, dict)
    assert (
        first_attachment.get("asset_url")
        or first_attachment.get("image_url")
        or first_attachment.get("thumb_url")
    ), f"Expected URL field in message attachment: {first_attachment}"

    print(
        "Sent Stream message with attachment "
        f"channel={CHANNEL_TYPE}:{CHANNEL_ID} message_id={sent_message['id']}"
    )
