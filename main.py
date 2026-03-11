"""Vision Agents backend with staged product-page workflow."""

from __future__ import annotations

import asyncio
import inspect
import os
import random
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from stream_chat import StreamChat
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.runner.http.options import ServeOptions
from vision_agents.plugins import gemini, getstream

from helpers import build_channel_id, ensure_workflow_channel_is_created
from integrations.nano_banana import (
    NanoBananaClient,
    NanoBananaConfigurationError,
)
from integrations.product_search import ProductSearchAdapter
from processors.object_capture import ObjectCaptureProcessor
from workflow.events import EventEmitter, StreamEventEmitter, WorkflowContext
from workflow.orchestrator import OrchestratorConfig, TaskWorkflowOrchestrator

load_dotenv()


@dataclass(slots=True)
class ProductDetails:
    """Details of the searched product."""

    name: str
    description: str
    specifications: str
    pricing: str


# Global variables
_stream_chat_client: StreamChat | None = None
_event_emitter: EventEmitter | None = None
_object_capture_processor: ObjectCaptureProcessor | None = None

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


def _get_stream_chat_client() -> StreamChat:
    global _stream_chat_client
    if _stream_chat_client is not None:
        return _stream_chat_client

    api_key = os.getenv("STREAM_API_KEY")
    api_secret = os.getenv("STREAM_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "Stream Chat is not configured. Set STREAM_API_KEY and STREAM_API_SECRET."
        )
    _stream_chat_client = StreamChat(api_key=api_key, api_secret=api_secret)
    return _stream_chat_client


async def create_agent(**kwargs) -> Agent:
    """Create a realtime seller agent with MCP/function-calling support."""
    # llm = gemini.Realtime(model="gemini-3.1-flash-audio-eap", fps=5)
    llm = gemini.Realtime(fps=5)

    @llm.register_function(
        name="nano_banana_polish",
        description="Create polished ecommerce product screenshots from the source image.",
    )
    async def nano_banana_polish(prompt: str) -> dict[str, Any]:
        try:
            nano_banana_client = NanoBananaClient(event_emitter=_get_event_emitter())
            return await nano_banana_client.generate_product_shot(
                image_name=CAPTURED_PRODUCT_IMAGE_FILENAME, prompt=prompt
            )
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
        print(f"[SearchTool] Searching product details for: ${query}")
        summary = (
            "Likely a consumer product suitable for online resale. "
            "Use concise feature bullets and a condition-focused listing."
        )
        facts = [
            "Detected from webcam image",
            f"Search query: {query}",
            "Recommended to include size/model specifics if known",
        ]
        if image_url:
            facts.append(f"Reference image: {image_url}")
        return ProductDetails(
            name="Product Name",
            description=summary,
            specifications=f"{', '.join(facts)}",
            pricing=f"${random.uniform(10, 1000):.2f}",
        )

    processor = _get_object_capture_processor()
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Assistant", id="agent"),
        instructions=(
            "You are a product-listing assistant. Guide the user to frame an item, "
            "capture it, polish images via tools, research product info, and produce "
            "a structured product description."
        ),
        llm=llm,
        processors=[processor],
    )
    return agent


async def _call_tool(
    agent: Agent, tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    llm = getattr(agent, "llm", None)
    call_tool = getattr(llm, "call_tool", None)
    if callable(call_tool):
        result = await call_tool(tool_name, arguments)
        return result if isinstance(result, dict) else {"result": result}

    fallback_tools = getattr(agent, "_fallback_tools", {})
    tool = fallback_tools.get(tool_name)
    if tool is None:
        raise RuntimeError(f"Tool not found: {tool_name}")

    result = tool(**arguments)
    if inspect.isawaitable(result):
        result = await result
    return result if isinstance(result, dict) else {"result": result}


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join call and execute the full staged task workflow."""
    await agent.authenticate()
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        capture = _get_object_capture_processor()
        stream_client = _get_stream_chat_client()

        channel = await asyncio.to_thread(
            ensure_workflow_channel_is_created,
            stream_client,
            channel_id=build_channel_id(call_id),
        )
        event_emitter = StreamEventEmitter(
            channel=channel,
            context=WorkflowContext(
                workflow_id=call_id,
                channel_type="messaging",
                channel_id=channel.cid,
                sender_user_id=os.getenv(
                    "STREAM_WORKFLOW_SENDER_USER_ID", "workflow-server"
                ),
            ),
        )
        await event_emitter.emit_workflow_started()
        search = ProductSearchAdapter(
            tool_caller=lambda name, args: _call_tool(agent, name, args)
        )

        orchestrator = TaskWorkflowOrchestrator(
            agent=agent,
            progress_emitter=event_emitter,
            capture_processor=capture,
            product_search=search,
            config=OrchestratorConfig(),
        )
        try:
            description = await orchestrator.run()
        except Exception as exc:
            await event_emitter.emit_workflow_failed(
                error_code=type(exc).__name__,
                error_message=str(exc),
            )
            raise

        await event_emitter.emit_workflow_completed(
            result_summary=f"{description.title} ({description.confidence:.2f})"
        )
        await agent.simple_response(
            "Product page draft is ready. "
            f"Title: {description.title}. "
            f"Confidence: {description.confidence:.2f}."
        )
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
