"""Tests for frontend progress event contract."""

from __future__ import annotations

import pytest

from workflow.events import TaskProgressEmitter, TaskProgressEvent
from workflow.models import TaskId, TaskProgress


class FakeEventBus:
    def __init__(self) -> None:
        self.registered: list[type] = []
        self.events: list[object] = []

    def register(self, *event_classes, ignore_not_compatible: bool = False) -> None:
        self.registered.extend(event_classes)

    def send(self, *events: object) -> None:
        self.events.extend(events)


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
    assert payload["taskId"] == "detect-object"
    assert payload["progress"] == 45
    assert payload["message"] == "Analyzing frames"
    assert payload["state"] == "running"
    assert len(bus.events) == 1
    assert bus.registered

    event = bus.events[0]
    assert isinstance(event, TaskProgressEvent)
    assert event.type == "task.progress"
    assert event.task_id == "detect-object"
    assert event.progress == 45

