"""Workflow event emitters and payload serialization."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

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


class WorkflowEventEmitter(Protocol):
    """Transport-agnostic emitter contract used by the orchestrator."""

    async def emit(self, update: TaskProgress) -> dict[str, Any]: ...


class StreamTaskProgressEmitter:
    """Emit workflow events as Stream Chat custom channel events."""

    def __init__(
        self,
        *,
        channel: Any,
        context: WorkflowContext,
        strict: bool = False,
    ) -> None:
        self._channel = channel
        self._context = context
        self._strict = strict

    async def emit_workflow_started(self) -> dict[str, Any]:
        payload = {
            "type": "workflow-started",
            "workflowId": self._context.workflow_id,
            "channelType": self._context.channel_type,
            "channelId": self._context.channel_id,
            "startedAt": _utc_timestamp(),
        }
        return await self._send(payload)

    async def emit_workflow_completed(self, *, result_summary: str | None = None) -> dict[str, Any]:
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

    async def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a payload to Stream without blocking the event loop."""
        logger.info(
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
            logger.exception("Failed to send workflow event", extra={"event_type": payload["type"]})
            if self._strict:
                raise
        return payload

