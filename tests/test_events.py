"""Tests for frontend progress event contract."""

from __future__ import annotations

import pytest

from workflow.events import StreamTaskProgressEmitter, WorkflowContext
from workflow.models import TaskId, TaskProgress


class FakeChannel:
    def __init__(self) -> None:
        self.events: list[tuple[dict, str]] = []

    def send_event(self, payload: dict, user_id: str) -> None:
        self.events.append((payload, user_id))


@pytest.mark.asyncio
async def test_progress_event_contract_shape() -> None:
    channel = FakeChannel()
    emitter = StreamTaskProgressEmitter(
        channel=channel,
        context=WorkflowContext(
            workflow_id="call-123",
            channel_type="messaging",
            channel_id="workflow-call-123",
            sender_user_id="workflow-server",
        ),
    )

    payload = await emitter.emit(
        TaskProgress(
            task_id=TaskId.DETECT_OBJECT,
            progress=45,
            message="Analyzing frames",
        )
    )

    assert payload["type"] == "workflow-task-progress"
    assert payload["workflowId"] == "call-123"
    assert payload["taskId"] == "detect-object"
    assert payload["progress"] == 45
    assert payload["message"] == "Analyzing frames"
    assert payload["state"] == "running"
    assert "timestamp" in payload
    assert len(channel.events) == 1

    sent_payload, sent_user_id = channel.events[0]
    assert sent_payload["type"] == "workflow-task-progress"
    assert sent_payload["taskId"] == "detect-object"
    assert sent_user_id == "workflow-server"

