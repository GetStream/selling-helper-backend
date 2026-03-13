"""Vision Agents backend with staged product-page workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.genai import types
from stream_chat import StreamChat
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.runner.http.options import ServeOptions
from vision_agents.plugins import gemini, getstream

from workflow.models import ProductDetails
from integrations.google_search import GoogleSearchClient
from integrations.nano_banana import (
    NanoBananaClient,
    NanoBananaConfigurationError,
)
from processors.object_capture import CaptureStatus, ObjectCaptureProcessor
from workflow.events import (
    EventEmitter,
    StreamEventEmitter,
    WorkflowContext,
)
from workflow.models import TaskId, TaskProgress

load_dotenv()


# Global variables
_stream_chat_client: StreamChat | None = None
_event_emitter: EventEmitter | None = None
_object_capture_processor: ObjectCaptureProcessor | None = None
_nano_banana_client: NanoBananaClient | None = None


def _get_nano_banana_client() -> NanoBananaClient:
    global _nano_banana_client
    if _nano_banana_client is not None:
        return _nano_banana_client
    _nano_banana_client = NanoBananaClient(event_emitter=_get_event_emitter())
    return _nano_banana_client


# Constants
CAPTURED_PRODUCT_IMAGE_FILENAME = "captured_product_frame"


def _get_event_emitter() -> EventEmitter:
    global _event_emitter
    if _event_emitter is not None:
        return _event_emitter
    _event_emitter = StreamEventEmitter(
        channel=None,
        context=WorkflowContext(
            workflow_id="7430192471",
            channel_type="messaging",
            channel_id="workflow-7430192471",
            sender_user_id="workflow-server",
        ),
    )
    return _event_emitter


def _get_object_capture_processor() -> ObjectCaptureProcessor:
    global _object_capture_processor
    if _object_capture_processor is not None:
        return _object_capture_processor
    _object_capture_processor = ObjectCaptureProcessor(
        min_ready_score=0.4,
        consecutive_ready_frames=10,
        source_frame_filename=CAPTURED_PRODUCT_IMAGE_FILENAME,
    )


async def create_agent(**kwargs) -> Agent:
    """Create a realtime seller agent with MCP/function-calling support."""
    llm = gemini.Realtime(
        model="gemini-3.1-flash-audio-eap",
        fps=5,
        config=types.LiveConnectConfigDict(
            thinking_config=types.ThinkingConfigDict(
                thinking_level=types.ThinkingLevel.MINIMAL
            )
        ),
    )
    # llm = gemini.Realtime(fps=5)

    @llm.register_function(
        name="create_product_screenshot",
        description="Create a product screenshot from user's video feed.",
    )
    async def create_product_screenshot() -> bytes:
        print("[Main] create_product_screenshot()")
        capture = _get_object_capture_processor()
        event_emitter = _get_event_emitter()
        await capture.update_capture_status(CaptureStatus.DETECTING)
        await event_emitter.emit(
            TaskProgress(
                task_id=TaskId.DETECT_OBJECT,
                progress=10,
                message="Starting to analyze frames",
            )
        )
        for attempt in range(4 + 1):
            print(
                f"[Main] Detecting object in frame, attempt {attempt + 1} / {5} (Processor: {capture.capture_status})"
            )
            ready = await capture.wait_until_ready(10.0)
            if ready:
                await event_emitter.emit(
                    TaskProgress(
                        task_id=TaskId.DETECT_OBJECT,
                        progress=100,
                        message="Object clearly detected in frame",
                    )
                )
                await event_emitter.emit(
                    TaskProgress(
                        task_id=TaskId.CAPTURE_SCREENSHOT,
                        progress=10,
                        message="Capturing screenshot",
                    )
                )
                print("[Orchestrator] Object clearly detected in frame")
                await capture.update_capture_status(CaptureStatus.CAPTURING)
                await capture.capture_best_frame(Path("artifacts"))
                await capture.update_capture_status(CaptureStatus.INACTIVE)
                await event_emitter.emit(
                    TaskProgress(
                        task_id=TaskId.CAPTURE_SCREENSHOT,
                        progress=100,
                        message="Screenshot captured",
                    )
                )
                return "Successfully captured product screenshot."

            else:
                capture.min_ready_score -= 0.05

            await event_emitter.emit(
                TaskProgress(
                    task_id=TaskId.DETECT_OBJECT,
                    progress=min(85, 30 + (attempt * 20)),
                    message="Still analyzing frames",
                )
            )
        await capture.update_capture_status(CaptureStatus.INACTIVE)
        return "Failed to capture product screenshot."

    @llm.register_function(
        name="nano_banana_polish",
        description="Create polished ecommerce product screenshots from the source image.",
    )
    async def nano_banana_polish(prompt: str) -> dict[str, Any]:
        print(f"[Main] nano_banana_polish({prompt})")
        try:
            nano_banana_client = _get_nano_banana_client()

            product_shot_result = await nano_banana_client.generate_product_shot(
                image_name=CAPTURED_PRODUCT_IMAGE_FILENAME, prompt=prompt
            )

            return product_shot_result
        except NanoBananaConfigurationError as exc:
            raise RuntimeError(
                "Nano Banana is not configured. Set NANO_BANANA_API_KEY "
                "(or GEMINI_API_KEY) and optionally NANO_BANANA_MODEL."
            ) from exc

    @llm.register_function(
        name="product_search",
        description="Search product details and likely features for a photographed item.",
    )
    async def product_search(
        query: str, image_url: str | None = None
    ) -> ProductDetails:
        print(f"[Main] Searching product details for: {query}")

        try:
            search_client = GoogleSearchClient(event_emitter=_get_event_emitter())
            search_results = search_client.perform_google_search(query=query)
            print(f"[Main] Search results: {search_results}")
        except Exception as exc:
            print(f"[Main] Error searching product details: {exc}")
            _get_event_emitter().emit_workflow_failed(
                error_code=type(exc).__name__,
                error_message=str(exc),
            )
            return "Failed to retrieve product details."
        
        return "Retrieved product details successfully."

    processor = _get_object_capture_processor()
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Assistant", id="agent"),
        instructions=(
            "You are a product-listing assistant. Guide the user to follow @selling-helper.md"
        ),
        llm=llm,
        processors=[processor],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join call and execute the full staged task workflow."""
    await agent.authenticate()
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        await agent.simple_response("Greet the user and ask them to start the process")
        
        await agent.finish()


# CORS configuration
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")
CORS_ALLOW_METHODS = ("GET", "POST", "DELETE", "OPTIONS")
CORS_ALLOW_HEADERS = ("*",)
CORS_ALLOW_CREDENTIALS = False
serve_options = ServeOptions(
    cors_allow_origins=(CORS_ALLOW_ORIGINS,),
    cors_allow_methods=CORS_ALLOW_METHODS,
    cors_allow_headers=CORS_ALLOW_HEADERS,
    cors_allow_credentials=CORS_ALLOW_CREDENTIALS,
)

if __name__ == "__main__":
    processor = _get_object_capture_processor()
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
