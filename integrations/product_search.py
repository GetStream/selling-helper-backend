"""Product research integration via MCP search tooling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass(slots=True)
class ProductSearchResult:
    """Normalized product research data from search providers."""

    summary: str
    facts: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    confidence: float = 0.5
    raw: dict[str, Any] | None = None


class ProductSearchAdapter:
    """Adapter for MCP-based product search and normalization."""

    def __init__(
        self,
        tool_caller: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
        tool_name: str = "product_search",
    ) -> None:
        self._tool_caller = tool_caller
        self._tool_name = tool_name

    async def research(self, *, query: str, image_url: str | None = None) -> ProductSearchResult:
        payload: dict[str, Any] = {"query": query}
        if image_url:
            payload["image_url"] = image_url

        response = await self._tool_caller(self._tool_name, payload)
        summary = response.get("summary") or response.get("description") or ""
        if not summary:
            raise ValueError("Product search response did not include a summary.")

        facts = response.get("facts") or response.get("highlights") or []
        references = response.get("references") or response.get("urls") or []
        confidence = float(response.get("confidence", 0.5))

        return ProductSearchResult(
            summary=summary,
            facts=[str(item) for item in facts],
            references=[str(item) for item in references],
            confidence=max(0.0, min(1.0, confidence)),
            raw=response,
        )

