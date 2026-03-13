import os
from datetime import datetime
from typing import Any

from stream_chat import StreamChat


def get_date_and_time_now() -> str:
    """Return a dash-separated timestamp suitable for filenames, e.g. 2024-04-25-13-54-03."""
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def build_channel_id(call_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in call_id)
    return f"workflow-{safe}"


def get_stream_call_api() -> str:
    call_id = os.getenv("STREAM_CALL_API", "").strip()
    if not call_id:
        raise RuntimeError(
            "Stream workflow call is not configured. Set STREAM_CALL_API."
        )
    return call_id


def get_stream_workflow_channel_id() -> str:
    return build_channel_id(get_stream_call_api())


def ensure_workflow_channel_is_created(
    client: StreamChat,
    *,
    channel_id: str,
) -> Any:
    sender_user_id = os.getenv("STREAM_WORKFLOW_SENDER_USER_ID", "workflow-server")
    channel = client.channel(
        "messaging", channel_id, data={"name": f"Workflow {channel_id}"}
    )
    client.upsert_user({"id": sender_user_id, "name": "Workflow Server"})
    channel.create(sender_user_id)
    return channel