"""Vision Agents backend with staged product-page workflow."""

from __future__ import annotations

import inspect
import os
from typing import Any

from dotenv import load_dotenv

from integrations.mcp_servers import build_mcp_servers
from integrations.nano_banana import NanoBananaAdapter
from integrations.product_search import ProductSearchAdapter
from processors.object_capture import ObjectCaptureProcessor, StubObjectCaptureProcessor
from workflow.events import TaskProgressEmitter
from workflow.orchestrator import OrchestratorConfig, TaskWorkflowOrchestrator
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.runner.http.options import ServeOptions
from vision_agents.plugins import gemini, getstream

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create a realtime seller agent with MCP/function-calling support."""
    # llm = gemini.Realtime(model="gemini-3.1-flash-audio-eap", fps=5)
    llm = gemini.Realtime(fps=5)
    # capture_processor = ObjectCaptureProcessor()

    # @llm.register_function(
    #     name="nano_banana_polish",
    #     description="Create polished ecommerce product screenshots from the source image.",
    # )
    # def nano_banana_polish(image_path: str, prompt: str) -> dict[str, str]:
    #     # Fallback for local development if MCP server tool is unavailable.
    #     return {"image_url": f"file://{image_path}", "revised_prompt": prompt}

    # @llm.register_function(
    #     name="product_search",
    #     description="Search product details and likely features for a photographed item.",
    # )
    # def product_search(query: str, image_url: str | None = None) -> dict[str, Any]:
    #     summary = (
    #         "Likely a consumer product suitable for online resale. "
    #         "Use concise feature bullets and a condition-focused listing."
    #     )
    #     facts = [
    #         "Detected from webcam image",
    #         f"Search query: {query}",
    #         "Recommended to include size/model specifics if known",
    #     ]
    #     if image_url:
    #         facts.append(f"Reference image: {image_url}")
    #     return {"summary": summary, "facts": facts, "confidence": 0.56}

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Assistant", id="agent"),
        instructions=(
            "You are a product-listing assistant. Guide the user to frame an item, "
            "capture it, polish images via tools, research product info, and produce "
            "a structured product description."
        ),
        llm=llm,
        # processors=[capture_processor],
        # mcp_servers=build_mcp_servers(),
    )
    # setattr(agent, "_capture_processor", capture_processor)
    # setattr(
    #     agent,
    #     "_fallback_tools",
    #     {
    #         "nano_banana_polish": nano_banana_polish,
    #         "product_search": product_search,
    #     },
    # )
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
        # capture = getattr(agent, "_capture_processor", StubObjectCaptureProcessor())
        # progress_emitter = TaskProgressEmitter(getattr(agent, "events", None))

        # nano = NanoBananaAdapter(tool_caller=lambda name, args: _call_tool(agent, name, args))
        # search = ProductSearchAdapter(tool_caller=lambda name, args: _call_tool(agent, name, args))

        # orchestrator = TaskWorkflowOrchestrator(
        #     agent=agent,
        #     progress_emitter=progress_emitter,
        #     capture_processor=capture,
        #     nano_banana=nano,
        #     product_search=search,
        #     config=OrchestratorConfig(),
        # )
        # description = await orchestrator.run()
        # await agent.simple_response(
        #     "Product page draft is ready. "
        #     f"Title: {description.title}. "
        #     f"Confidence: {description.confidence:.2f}."
        # )
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
