"""Async workflow orchestrator tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from integrations.nano_banana import NanoBananaAdapter
from integrations.product_search import ProductSearchAdapter
from workflow.events import TaskProgressEmitter
from workflow.orchestrator import OrchestratorConfig, TaskWorkflowOrchestrator


class FakeAgent:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def simple_response(self, text: str) -> None:
        self.messages.append(text)


class FakeEventBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def register(self, *event_classes, ignore_not_compatible: bool = False) -> None:
        return None

    def send(self, *events: object) -> None:
        self.events.extend(events)


class FakeCaptureProcessor:
    guidance_message = "Move closer to the webcam."

    async def wait_until_ready(self, timeout_seconds: float = 20.0) -> bool:  # noqa: ARG002
        return True

    async def capture_best_frame(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "fake_source.png"
        image_path.write_text("img", encoding="utf-8")
        return image_path


@pytest.mark.asyncio
async def test_orchestrator_happy_path(tmp_path: Path) -> None:
    async def tool_caller(tool_name: str, payload: dict) -> dict:
        if tool_name == "nano_banana_polish":
            return {
                "image_url": "https://example.com/polished-item.png",
                "revised_prompt": payload["prompt"],
            }
        if tool_name == "product_search":
            return {
                "summary": "Compact espresso machine with stainless finish.",
                "facts": ["Espresso machine", "Stainless steel", "Countertop appliance"],
                "confidence": 0.84,
            }
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    agent = FakeAgent()
    emitter = TaskProgressEmitter(FakeEventBus())
    capture = FakeCaptureProcessor()
    nano = NanoBananaAdapter(tool_caller=tool_caller)
    search = ProductSearchAdapter(tool_caller=tool_caller)

    orchestrator = TaskWorkflowOrchestrator(
        agent=agent,
        progress_emitter=emitter,
        capture_processor=capture,
        nano_banana=nano,
        product_search=search,
        config=OrchestratorConfig(screenshot_dir=tmp_path),
    )

    output = await orchestrator.run()

    assert output.title == "Espresso machine"
    assert str(output.polished_image_url) == "https://example.com/polished-item.png"
    assert output.confidence > 0.7
    assert any("Please hold your item" in msg for msg in agent.messages)

