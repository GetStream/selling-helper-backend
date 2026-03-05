"""Step-based orchestration for product-page generation."""

from __future__ import annotations

import asyncio
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from integrations.nano_banana import NanoBananaAdapter
from integrations.product_search import ProductSearchAdapter
from schemas.product_description import ProductDescription, build_product_description
from workflow.events import WorkflowEventEmitter
from workflow.models import TaskId, TaskProgress, TaskState, WorkflowArtifacts


class SupportsSimpleResponse(Protocol):
    """Small protocol to keep orchestrator decoupled from Agent concrete type."""

    async def simple_response(self, text: str) -> None: ...


class SupportsCapture(Protocol):
    """Protocol for object readiness and frame capture dependency."""

    guidance_message: str

    async def wait_until_ready(self, timeout_seconds: float = 20.0) -> bool: ...
    async def capture_best_frame(self, output_dir: Path) -> Path: ...


@dataclass(slots=True)
class OrchestratorConfig:
    """Configurable runtime thresholds for the workflow."""

    capture_timeout_seconds: float = 64.0
    detect_retries: int = 2
    screenshot_dir: Path = Path("artifacts")
    nano_prompt: str = (
        "Create polished ecommerce product photos on a clean background with soft lighting."
    )
    search_query_template: str = "Identify this product and key selling points: {title_hint}"


class TaskWorkflowOrchestrator:
    """Coordinates all steps in the selling workflow."""

    def __init__(
        self,
        *,
        agent: SupportsSimpleResponse,
        progress_emitter: WorkflowEventEmitter,
        capture_processor: SupportsCapture,
        nano_banana: NanoBananaAdapter,
        product_search: ProductSearchAdapter,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._agent = agent
        self._progress = progress_emitter
        self._capture = capture_processor
        self._nano = nano_banana
        self._search = product_search
        self._config = config or OrchestratorConfig()
        self._artifacts = WorkflowArtifacts()

    async def run(self) -> ProductDescription:
        await self._detect_object_step()
        await self._capture_screenshot_step()
        await self._polish_image_step()
        await self._research_step()
        return await self._generate_description_step()

    async def _emit(self, task_id: TaskId, progress: int, message: str) -> None:
        await self._progress.emit(
            TaskProgress(
                task_id=task_id,
                progress=progress,
                message=message,
                state=TaskState.RUNNING if progress < 100 else TaskState.COMPLETED,
            )
        )

    async def _detect_object_step(self) -> None:
        await self._emit(TaskId.DETECT_OBJECT, 0, "Detecting object in frame")
        await self._agent.simple_response(
            "Please hold your item in front of the webcam. I will guide you automatically."
        )

        for attempt in range(self._config.detect_retries + 1):
            ready = await self._capture.wait_until_ready(
                timeout_seconds=self._config.capture_timeout_seconds
            )
            if ready:
                await self._emit(TaskId.DETECT_OBJECT, 100, "Object clearly detected in frame")
                return

            await self._emit(
                TaskId.DETECT_OBJECT,
                min(85, 30 + (attempt * 20)),
                "Still analyzing frames",
            )
            await self._agent.simple_response(self._capture.guidance_message)

        raise TimeoutError("Object was not detected clearly enough within timeout window.")

    async def _capture_screenshot_step(self) -> None:
        await self._emit(TaskId.CAPTURE_SCREENSHOT, 0, "Capturing screenshot")
        screenshot_path = await self._capture.capture_best_frame(self._config.screenshot_dir)
        # Keep a canonical absolute path so every integration resolves the same file.
        self._artifacts.source_image_path = str(screenshot_path.resolve())
        await self._emit(TaskId.CAPTURE_SCREENSHOT, 100, "Screenshot captured")

    async def _polish_image_step(self) -> None:
        if not self._artifacts.source_image_path:
            raise RuntimeError("Source image path is missing before polish step.")

        await self._emit(TaskId.POLISH_IMAGE, 0, "Polishing product image")
        result = await asyncio.wait_for(
            self._nano.polish_image(
                image_path=self._artifacts.source_image_path,
                prompt=self._config.nano_prompt,
            ),
            timeout=45.0,
        )
        self._artifacts.polished_image_url = result.image_url

        saved_path = await self._save_polished_image(
            image_url=result.image_url or None,
            image_bytes=result.image_bytes,
            image_mime_type=result.image_mime_type,
        )
        if not self._artifacts.polished_image_url:
            self._artifacts.polished_image_url = saved_path.resolve().as_uri()
        self._artifacts.polished_image_path = str(saved_path)
        await self._emit(TaskId.POLISH_IMAGE, 100, "Polished product image created")

    async def _save_polished_image(
        self,
        *,
        image_url: str | None = None,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> Path:
        """Download/copy/write the polished image to the artifacts directory."""
        output_dir = self._config.screenshot_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"polished_product{_extension_for_mime(image_mime_type)}"

        if image_bytes:
            await asyncio.to_thread(output_path.write_bytes, image_bytes)
            return output_path

        if not image_url:
            raise ValueError("Polished image is missing both URL and inline bytes.")

        parsed = urlparse(image_url)
        if parsed.scheme == "file":
            # Handle both canonical file URIs (file:///abs/path) and legacy malformed
            # relative variants (file://artifacts/source.png).
            if parsed.netloc:
                source = Path(unquote(parsed.netloc)) / unquote(parsed.path.lstrip("/"))
            else:
                source = Path(unquote(parsed.path))
            source = source.resolve()
            if not source.exists():
                raise FileNotFoundError(f"Polished image source not found: {source}")
            await asyncio.to_thread(shutil.copy2, source, output_path)
        elif parsed.scheme in ("http", "https"):
            await asyncio.to_thread(urllib.request.urlretrieve, image_url, output_path)
        else:
            source = Path(image_url)
            if source.exists():
                await asyncio.to_thread(shutil.copy2, source, output_path)
            else:
                raise ValueError(f"Unsupported image URL scheme: {parsed.scheme!r}")

        return output_path

    async def _research_step(self) -> None:
        await self._emit(TaskId.RESEARCH_PRODUCT, 0, "Researching product information")
        query = self._config.search_query_template.format(title_hint="webcam-detected item")
        result = await asyncio.wait_for(
            self._search.research(query=query, image_url=self._artifacts.polished_image_url),
            timeout=30.0,
        )
        self._artifacts.product_search_summary = result.summary
        self._artifacts.product_search_facts = result.facts
        await self._emit(TaskId.RESEARCH_PRODUCT, 100, "Product research complete")

    async def _generate_description_step(self) -> ProductDescription:
        await self._emit(TaskId.GENERATE_DESCRIPTION, 0, "Generating structured description")
        summary = self._artifacts.product_search_summary or "Well-presented second-hand item."
        facts = self._artifacts.product_search_facts
        title = (facts[0] if facts else "Pre-owned Product").strip()[:80]
        short = summary[:220]
        long_desc = (
            f"{summary}\n\n"
            "Highlights:\n"
            + ("\n".join(f"- {fact}" for fact in facts[:8]) if facts else "- Quality item in good shape.")
        )

        description = build_product_description(
            title=title or "Pre-owned Product",
            short_description=short,
            long_description=long_desc,
            key_features=facts[:8] or ["Reliable condition", "Carefully photographed"],
            condition="Used - Good",
            estimated_category="Consumer Goods",
            search_facts=facts,
            confidence=0.75 if facts else 0.55,
            source_image_path=self._artifacts.source_image_path,
            polished_image_url=self._artifacts.polished_image_url,
            polished_image_path=self._artifacts.polished_image_path,
        )
        await self._emit(TaskId.GENERATE_DESCRIPTION, 100, "Structured description ready")
        return description


def _extension_for_mime(image_mime_type: str | None) -> str:
    if image_mime_type == "image/jpeg":
        return ".jpg"
    if image_mime_type == "image/webp":
        return ".webp"
    if image_mime_type == "image/gif":
        return ".gif"
    return ".png"

