"""Event bridge for frontend-visible workflow progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from workflow.models import TaskProgress


@dataclass(slots=True)
class TaskProgressEvent:
    """Event payload aligned with frontend contract."""

    type: str = "task.progress"
    payload: dict[str, Any] = field(default_factory=dict)


class TaskProgressEmitter:
    """Emit `task.progress` events through the agent event bus."""

    def __init__(self, event_bus: Any | None) -> None:
        self._event_bus = event_bus
        self._is_registered = False

    def _ensure_registered(self) -> None:
        if self._is_registered or self._event_bus is None:
            return
        register = getattr(self._event_bus, "register", None)
        if callable(register):
            register(TaskProgressEvent)
            self._is_registered = True

    async def emit(self, update: TaskProgress) -> dict[str, Any]:
        """Emit a progress update and return the serialized payload."""
        payload = {
            "taskId": update.task_id.value,
            "progress": update.progress,
            "message": update.message,
        }
        event = TaskProgressEvent(payload=payload)

        self._ensure_registered()
        emit = getattr(self._event_bus, "emit", None)
        if callable(emit):
            await emit(event)
        return {"type": event.type, "payload": payload}

