"""Live Stream Chat test that sends a message with an image attachment.

Required environment variables:
- STREAM_API_KEY
- STREAM_API_SECRET

Run command:
    pytest tests/test_stream_chat_message_attachment_live.py -s
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv
from stream_chat import StreamChat

from helpers import (
    ensure_workflow_channel_is_created,
    get_stream_call_api,
    get_stream_workflow_channel_id,
)
from workflow.events import StreamEventEmitter, WorkflowContext

load_dotenv()

CHANNEL_TYPE = "messaging"
WORKFLOW_ID = get_stream_call_api()
CHANNEL_ID = get_stream_workflow_channel_id()
SENDER_USER_ID = "workflow-server"


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        pytest.skip(
            f"Missing required environment variable for live Stream test: {name}"
        )
    return value


def test_stream_live_sends_chat_message_with_image_attachment() -> None:
    api_key = _require_env("STREAM_API_KEY")
    api_secret = _require_env("STREAM_API_SECRET")

    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / "artifacts" / "polished_product_old-2.png"
    assert image_path.exists(), f"Expected image fixture at: {image_path}"
    assert image_path.stat().st_size > 0, f"Image fixture is empty: {image_path}"

    client = StreamChat(api_key=api_key, api_secret=api_secret)
    channel = ensure_workflow_channel_is_created(client, channel_id=CHANNEL_ID)
    event_emitter = StreamEventEmitter(
        channel=channel,
        context=WorkflowContext(
            workflow_id=WORKFLOW_ID,
            channel_type="messaging",
            channel_id=CHANNEL_ID,
            sender_user_id="workflow-server",
        ),
        strict=True,
    )

    try:
        send_response = asyncio.run(
            event_emitter.emit_polished_image(str(image_path), "image/png")
        )
    except requests.RequestException as exc:
        pytest.skip(f"Live Stream test could not reach Stream API: {exc}")

    assert isinstance(send_response, dict)
    sent_message = send_response.get("message", send_response)
    assert isinstance(sent_message, dict)
    assert sent_message.get("id"), (
        f"Expected sent message id in response: {send_response}"
    )
    attachments = sent_message.get("attachments") or []
    assert attachments, f"Expected message attachments in response: {send_response}"
    first_attachment = attachments[0]
    assert isinstance(first_attachment, dict)
    assert first_attachment.get("title") == image_path.name
    assert (
        first_attachment.get("asset_url")
        or first_attachment.get("image_url")
        or first_attachment.get("thumb_url")
    ), f"Expected URL field in message attachment: {first_attachment}"

    print(
        "Sent Stream message with attachment "
        f"channel={CHANNEL_TYPE}:{CHANNEL_ID} message_id={sent_message['id']}"
    )
