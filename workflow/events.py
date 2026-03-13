"""Workflow event emitters and payload serialization."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from stream_chat import StreamChat

from helpers import ProductDetails, ensure_workflow_channel_is_created
from workflow.models import TaskProgress

logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(slots=True)
class WorkflowContext:
    """Immutable metadata shared across workflow events."""

    workflow_id: str
    channel_type: str
    channel_id: str
    sender_user_id: str


class EventEmitter(Protocol):
    """Transport-agnostic emitter contract used by the orchestrator."""

    async def emit(self, update: TaskProgress) -> dict[str, Any]: ...
    async def emit_polished_image(
        self, image_data: bytes, image_mime_type: str
    ) -> dict[str, Any]: ...


class StreamEventEmitter:
    """Emit workflow events as Stream Chat custom channel events."""

    def __init__(
        self,
        *,
        channel: Any,
        context: WorkflowContext,
        strict: bool = False,
    ) -> None:
        api_key = os.getenv("STREAM_API_KEY")
        api_secret = os.getenv("STREAM_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError(
                "Stream Chat is not configured. Set STREAM_API_KEY and STREAM_API_SECRET."
            )
        stream_chat_client = StreamChat(api_key=api_key, api_secret=api_secret)

        self._channel = ensure_workflow_channel_is_created(
            client=stream_chat_client,
            channel_id="workflow-7430192471",
        )
        self._context = context
        self._strict = strict

    def update_channel(self, channel: Any) -> None:
        self._channel = channel

    async def emit_workflow_started(self) -> dict[str, Any]:
        payload = {
            "type": "workflow-started",
            "workflowId": self._context.workflow_id,
            "channelType": self._context.channel_type,
            "channelId": self._context.channel_id,
            "startedAt": _utc_timestamp(),
        }
        return await self._send(payload)

    async def emit_workflow_completed(
        self, *, result_summary: str | None = None
    ) -> dict[str, Any]:
        payload = {
            "type": "workflow-completed",
            "workflowId": self._context.workflow_id,
            "completedAt": _utc_timestamp(),
        }
        if result_summary:
            payload["resultSummary"] = result_summary
        return await self._send(payload)

    async def emit_workflow_failed(
        self,
        *,
        error_code: str,
        error_message: str,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "type": "workflow-failed",
            "workflowId": self._context.workflow_id,
            "failedAt": _utc_timestamp(),
            "errorCode": error_code,
            "errorMessage": error_message,
        }
        if task_id:
            payload["taskId"] = task_id
        return await self._send(payload)

    async def emit(self, update: TaskProgress) -> dict[str, Any]:
        """Emit a task progress update and return the payload."""
        payload = {
            "type": "workflow-task-progress",
            "workflowId": self._context.workflow_id,
            "taskId": update.task_id.value,
            "progress": update.progress,
            "message": update.message,
            "state": update.state.value,
            "timestamp": _utc_timestamp(),
        }
        return await self._send(payload)

    async def emit_polished_image(
        self, image_path: str, image_mime_type: str
    ) -> dict[str, Any]:
        """Emit a polished image event to the channel."""
        print(
            f"[StreamTaskProgressEmitter] Emitting polished image: {image_path} with mime type: {image_mime_type} to channel: {self._context.channel_type}:{self._context.channel_id} sender: {self._context.sender_user_id}"
        )

        SENDER_USER_ID = "workflow-server"
        SENDER_USER = {"id": SENDER_USER_ID, "name": "Workflow Server"}

        upload_response = self._channel.send_image(
            image_path, image_path, SENDER_USER, content_type="image/png"
        )

        print(f"[StreamTaskProgressEmitter] Upload response: {upload_response}")
        return upload_response

    async def emit_product_details(
        self, product_details: ProductDetails
    ) -> dict[str, Any]:
        payload = {
            "type": "product-details",
            "productDetails": product_details,
            "workflowId": self._context.workflow_id,
            "channelType": self._context.channel_type,
            "channelId": self._context.channel_id,
            "senderUserId": self._context.sender_user_id,
            "timestamp": _utc_timestamp(),
        }
        return await self._send(payload)

    async def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a payload to Stream without blocking the event loop."""
        logger.info(
            "Emitting workflow event channel=%s:%s sender=%s payload=%s",
            self._context.channel_type,
            self._context.channel_id,
            self._context.sender_user_id,
            payload,
        )

        print(
            "Emitting workflow event channel=%s:%s sender=%s payload=%s",
            self._context.channel_type,
            self._context.channel_id,
            self._context.sender_user_id,
            payload,
        )
        try:
            await asyncio.to_thread(
                self._channel.send_event,
                payload,
                self._context.sender_user_id,
            )
            logger.info(
                "Workflow event sent type=%s workflow_id=%s",
                payload.get("type"),
                self._context.workflow_id,
            )
        except Exception:
            logger.exception(
                "Failed to send workflow event", extra={"event_type": payload["type"]}
            )
            if self._strict:
                raise
        return payload
