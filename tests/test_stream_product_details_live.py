"""Live Stream Chat event test for product-details payloads.

Required environment variables:
- STREAM_API_KEY
- STREAM_API_SECRET

Run command:
    pytest tests/test_stream_product_details_live.py -s
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

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
from workflow.models import ProductDetails

load_dotenv()

CHANNEL_TYPE = "messaging"
CHANNEL_ID = get_stream_workflow_channel_id()
WORKFLOW_ID = get_stream_call_api()
SENDER_USER_ID = "workflow-server"


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        pytest.skip(
            f"Missing required environment variable for live Stream test: {name}"
        )
    return value


def test_stream_live_emits_product_details_event() -> None:
    api_key = _require_env("STREAM_API_KEY")
    api_secret = _require_env("STREAM_API_SECRET")

    client = StreamChat(api_key=api_key, api_secret=api_secret)
    channel = ensure_workflow_channel_is_created(client, channel_id=CHANNEL_ID)
    emitter = StreamEventEmitter(
        channel=channel,
        context=WorkflowContext(
            workflow_id=WORKFLOW_ID,
            channel_type=CHANNEL_TYPE,
            channel_id=CHANNEL_ID,
            sender_user_id=SENDER_USER_ID,
        ),
        strict=True,
    )

    unique_suffix = uuid4().hex[:8]
    product_details = ProductDetails(
        name=f"Frontend verification speaker {unique_suffix}",
        description="Portable Bluetooth speaker with 360-degree sound and 20-hour battery life.",
        specifications="IP67 waterproof, USB-C charging, stereo pairing, integrated microphone.",
        pricing=129.99,
    )

    try:
        payload = asyncio.run(emitter.emit_product_details(product_details))
    except requests.RequestException as exc:
        pytest.skip(f"Live Stream test could not reach Stream API: {exc}")

    assert payload["type"] == "product-details"
    assert payload["workflowId"] == WORKFLOW_ID
    assert payload["channelType"] == CHANNEL_TYPE
    assert payload["channelId"] == CHANNEL_ID
    assert payload["senderUserId"] == SENDER_USER_ID
    assert payload["productDetails"] == product_details.model_dump(
        mode="json", exclude_none=False
    )
    assert payload["timestamp"]

    print(
        "Emitted Stream product-details event "
        f"channel={CHANNEL_TYPE}:{CHANNEL_ID} "
        f"workflow_id={WORKFLOW_ID} "
        f"product_name={product_details.name}"
    )
