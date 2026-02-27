"""MCP server and tool-call helpers for workflow integrations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from vision_agents.core.mcp import MCPServerRemote


@dataclass(slots=True)
class MCPServerConfig:
    """Declarative server configuration from environment variables."""

    url: str
    headers: dict[str, str]
    timeout: float = 20.0
    session_timeout: float = 300.0


def _read_headers(var_name: str) -> dict[str, str]:
    raw = os.getenv(var_name, "")
    if not raw:
        return {}
    return json.loads(raw)


def build_mcp_servers() -> list[Any]:
    """Create all configured MCP servers for Agent(..., mcp_servers=[...])."""
    servers: list[Any] = []

    nano_url = os.getenv("NANO_BANANA_MCP_URL", "").strip()
    if nano_url:
        servers.append(
            MCPServerRemote(
                url=nano_url,
                headers=_read_headers("NANO_BANANA_MCP_HEADERS_JSON"),
                timeout=20.0,
                session_timeout=300.0,
            )
        )

    search_url = os.getenv("PRODUCT_SEARCH_MCP_URL", "").strip()
    if search_url:
        servers.append(
            MCPServerRemote(
                url=search_url,
                headers=_read_headers("PRODUCT_SEARCH_MCP_HEADERS_JSON"),
                timeout=20.0,
                session_timeout=300.0,
            )
        )

    return servers


class MCPFunctionCaller:
    """Small adapter for invoking registered MCP tools through an LLM interface."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        call_tool = getattr(self._llm, "call_tool", None)
        if not callable(call_tool):
            raise RuntimeError("LLM does not expose call_tool() for direct MCP invocation.")
        result = await call_tool(tool_name, arguments)
        if isinstance(result, dict):
            return result
        return {"result": result}

