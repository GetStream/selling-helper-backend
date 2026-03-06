"""Vision Agents backend with staged product-page workflow."""

from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any

from dotenv import load_dotenv
from stream_chat import StreamChat

from integrations.nano_banana import (
    NanoBananaAdapter,
    NanoBananaConfigurationError,
    NanoBananaDirectClient,
)
from integrations.product_search import ProductSearchAdapter
from processors.object_capture import ObjectCaptureProcessor, StubObjectCaptureProcessor
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.runner.http.options import ServeOptions
from vision_agents.plugins import gemini, getstream
from workflow.events import StreamTaskProgressEmitter, WorkflowContext
from workflow.orchestrator import OrchestratorConfig, TaskWorkflowOrchestrator

load_dotenv()

_nano_banana_client: NanoBananaDirectClient | None = None
_stream_chat_client: StreamChat | None = None

captured_product_image_filename = "captured_product_frame.png"


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


def _build_channel_id(call_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in call_id)
    return f"workflow-{safe}"


def _ensure_workflow_channel(
    client: StreamChat,
    *,
    channel_type: str,
    channel_id: str,
    server_user_id: str,
) -> Any:
    channel = client.channel(channel_type, channel_id, data={"name": f"Workflow {channel_id}"})
    client.upsert_user({"id": server_user_id, "name": "Workflow Server"})
    channel.create(server_user_id)
    return channel


def nano_banana_polish_direct(image_path: str, prompt: str) -> dict[str, Any]:
    """Directly call Nano Banana to create a polished product-shot image."""
    global _nano_banana_client
    if _nano_banana_client is None:
        _nano_banana_client = NanoBananaDirectClient.from_env()
    return _nano_banana_client.generate_product_shot(image_path=image_path, prompt=prompt)


async def create_agent(**kwargs) -> Agent:
    """Create a realtime seller agent with MCP/function-calling support."""
    # llm = gemini.Realtime(model="gemini-3.1-flash-audio-eap", fps=5)
    llm = gemini.Realtime(fps=5)
    capture_processor = ObjectCaptureProcessor(
        min_ready_score=0.62,
        consecutive_ready_frames=10,
        source_frame_filename=captured_product_image_filename,
    )

    @llm.register_function(
        name="nano_banana_polish",
        description="Create polished ecommerce product screenshots from the source image.",
    )
    def nano_banana_polish(prompt: str) -> dict[str, Any]:
        try:
            return nano_banana_polish_direct(image_path=captured_product_image_filename, prompt=prompt)
        except NanoBananaConfigurationError as exc:
            raise RuntimeError(
                "Nano Banana is not configured. Set NANO_BANANA_API_KEY "
                "(or GEMINI_API_KEY) and optionally NANO_BANANA_MODEL."
            ) from exc

    @llm.register_function(
        name="product_search",
        description="Search product details and likely features for a photographed item.",
    )
    def product_search(query: str, image_url: str | None = None) -> dict[str, Any]:
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
        return {"summary": summary, "facts": facts, "confidence": 0.56}

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Assistant", id="agent"),
        instructions=(
            "You are a product-listing assistant. Guide the user to frame an item, "
            "capture it, polish images via tools, research product info, and produce "
            "a structured product description."
        ),
        llm=llm,
        processors=[capture_processor],
    )
    setattr(agent, "_capture_processor", capture_processor)
    setattr(
        agent,
        "_fallback_tools",
        {
            "nano_banana_polish": nano_banana_polish,
            "product_search": product_search,
        },
    )
    return agent


async def _call_tool(agent: Agent, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        capture = getattr(agent, "_capture_processor", StubObjectCaptureProcessor())
        stream_client = _get_stream_chat_client()
        channel_type = os.getenv("STREAM_WORKFLOW_CHANNEL_TYPE", "messaging")
        channel_id = _build_channel_id(call_id)
        sender_user_id = os.getenv("STREAM_WORKFLOW_SENDER_USER_ID", "workflow-server")
        channel = await asyncio.to_thread(
            _ensure_workflow_channel,
            stream_client,
            channel_type=channel_type,
            channel_id=channel_id,
            server_user_id=sender_user_id,
        )
        progress_emitter = StreamTaskProgressEmitter(
            channel=channel,
            context=WorkflowContext(
                workflow_id=call_id,
                channel_type=channel_type,
                channel_id=channel_id,
                sender_user_id=sender_user_id,
            ),
        )
        await progress_emitter.emit_workflow_started()

        nano = NanoBananaAdapter(tool_caller=lambda name, args: _call_tool(agent, name, args))
        search = ProductSearchAdapter(tool_caller=lambda name, args: _call_tool(agent, name, args))

        orchestrator = TaskWorkflowOrchestrator(
            agent=agent,
            progress_emitter=progress_emitter,
            capture_processor=capture,
            nano_banana=nano,
            product_search=search,
            config=OrchestratorConfig(),
        )
        try:
            description = await orchestrator.run()
        except Exception as exc:
            await progress_emitter.emit_workflow_failed(
                error_code=type(exc).__name__,
                error_message=str(exc),
            )
            raise

        await progress_emitter.emit_workflow_completed(
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
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
