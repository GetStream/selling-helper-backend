"""Tests for frontend progress event contract."""

from __future__ import annotations

import pytest

from workflow.events import TaskProgressEmitter
from workflow.models import TaskId, TaskProgress


class FakeEventBus:
    def __init__(self) -> None:
        self.registered: list[type] = []
        self.events: list[object] = []

    def register(self, event_type: type) -> None:
        self.registered.append(event_type)

    async def emit(self, event: object) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_progress_event_contract_shape() -> None:
    bus = FakeEventBus()
    emitter = TaskProgressEmitter(bus)

    payload = await emitter.emit(
        TaskProgress(
            task_id=TaskId.DETECT_OBJECT,
            progress=45,
            message="Analyzing frames",
        )
    )

    assert payload["type"] == "task.progress"
    assert payload["payload"] == {
        "taskId": "detect-object",
        "progress": 45,
        "message": "Analyzing frames",
    }
    assert len(bus.events) == 1
    assert bus.registered

