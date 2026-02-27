"""Event bridge for frontend-visible workflow progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vision_agents.core.events import BaseEvent

from workflow.models import TaskProgress


@dataclass
class TaskProgressEvent(BaseEvent):
    """Event payload aligned with frontend contract.

    Extends BaseEvent so it participates in the Vision Agents EventManager
    lifecycle (register / subscribe / send).
    """

    type: str = field(default="task.progress", init=False)
    task_id: str = ""
    progress: int = 0
    message: str = ""
    state: str = "running"


class TaskProgressEmitter:
    """Emit ``task.progress`` events through the agent EventManager."""

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
        event = TaskProgressEvent(
            task_id=update.task_id.value,
            progress=update.progress,
            message=update.message,
            state=update.state.value,
        )

        self._ensure_registered()
        send = getattr(self._event_bus, "send", None)
        if callable(send):
            send(event)

        return {
            "type": event.type,
            "taskId": event.task_id,
            "progress": event.progress,
            "message": event.message,
            "state": event.state,
        }

