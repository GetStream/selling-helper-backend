"""Step-based orchestration for product-page generation."""

from __future__ import annotations

import asyncio
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from integrations.nano_banana import NanoBananaClient
from processors.object_capture import CaptureStatus, ObjectCaptureProcessor
from schemas.product_description import ProductDescription, build_product_description
from workflow.events import EventEmitter
from workflow.models import TaskId, TaskProgress, TaskState, WorkflowArtifacts


class SupportsSimpleResponse(Protocol):
    """Small protocol to keep orchestrator decoupled from Agent concrete type."""

    async def simple_response(self, text: str) -> None: ...


@dataclass(slots=True)
class OrchestratorConfig:
    """Configurable runtime thresholds for the workflow."""

    capture_timeout_seconds: float = 10.0
    detect_retries: int = 4
    screenshot_dir: Path = Path("artifacts")
    nano_prompt: str = "Create polished ecommerce product photos on a clean background with soft lighting."
    search_query_template: str = (
        "Identify this product and key selling points: {title_hint}"
    )


class TaskWorkflowOrchestrator:
    """Coordinates all steps in the selling workflow."""

    def __init__(
        self,
        *,
        agent: SupportsSimpleResponse,
        progress_emitter: EventEmitter,
        capture_processor: ObjectCaptureProcessor,
        nano_banana_client: NanoBananaClient,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._agent = agent
        self._progress = progress_emitter
        self._capture = capture_processor
        self._nano_banana = nano_banana_client
        self._config = config or OrchestratorConfig()
        self._artifacts = WorkflowArtifacts()
        self._image_polish_finished = False

    def _set_image_polish_finished(self, finished: bool) -> None:
        self._image_polish_finished = finished

    async def run(self) -> ProductDescription:
        print("[Orchestrator] Running workflow")
        await self._detect_object_step()
        print("[Orchestrator] Object detected")
        await self._capture_screenshot_step()
        print("[Orchestrator] Screenshot captured")
        await self._polish_image_step()
        print("[Orchestrator] Image polished")
        await self._research_step()
        print("[Orchestrator] Product research complete")
        description = await self._generate_description_step()
        print("[Orchestrator] Description generated")
        return description

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
            "Ask the user to hold an item in front of the webcam."
        )

        await self._capture.update_capture_status(CaptureStatus.DETECTING)
        for attempt in range(self._config.detect_retries + 1):
            print(
                f"[Orchestrator] Detecting object in frame, attempt {attempt + 1} / {self._config.detect_retries + 1} (Processor: {self._capture.capture_status})"
            )
            ready = await self._capture.wait_until_ready(
                timeout_seconds=self._config.capture_timeout_seconds
            )
            if ready:
                print("[Orchestrator] Object clearly detected in frame")
                await self._emit(
                    TaskId.DETECT_OBJECT, 100, "Object clearly detected in frame"
                )
                return
            else:
                self._capture.min_ready_score -= 0.05

            await self._emit(
                TaskId.DETECT_OBJECT,
                min(85, 30 + (attempt * 20)),
                "Still analyzing frames",
            )
            await self._agent.simple_response(self._capture.guidance_message)

        raise TimeoutError(
            "Object was not detected clearly enough within timeout window."
        )

    async def _capture_screenshot_step(self) -> None:
        print("[Orchestrator] Capturing screenshot")
        await self._capture.update_capture_status(CaptureStatus.CAPTURING)
        await self._emit(TaskId.CAPTURE_SCREENSHOT, 0, "Capturing screenshot")
        screenshot_path = await self._capture.capture_best_frame(
            self._config.screenshot_dir
        )

        if not screenshot_path:
            raise RuntimeError("Screenshot path is missing after capture step.")

        # Keep a canonical absolute path so every integration resolves the same file.
        print(f"[Orchestrator] Screenshot path: {screenshot_path}")
        self._artifacts.source_image_path = str(screenshot_path.resolve())
        print(f"[Orchestrator] Source image path: {self._artifacts.source_image_path}")
        await self._emit(TaskId.CAPTURE_SCREENSHOT, 100, "Screenshot captured")

    async def _polish_image_step(self) -> None:
        if not self._artifacts.source_image_path:
            raise RuntimeError("Source image path is missing before polish step.")

        await self._emit(TaskId.POLISH_IMAGE, 0, "Polishing product image")
        await self._agent.simple_response(
            "Please tell the user that the image will now be polished and perform that step in the background using the nano_banana_polish tool."
        )
        print(
            f"[Orchestrator] Polishing product image with prompt: {self._config.nano_prompt}"
        )

        for attempt in range(self._config.polish_retries + 1):
            print(
                f"[Orchestrator] Waiting for image polish to finish (Attempt {attempt + 1} / {self._config.polish_retries + 1})"
            )
            ready = await self._nano_banana.wait_until_ready(
                timeout_seconds=self._config.capture_timeout_seconds
            )
            if ready:
                print("[Orchestrator] Object clearly detected in frame")
                await self._emit(
                    TaskId.POLISH_IMAGE, 100, "Polished product image created"
                )
                return

            await self._emit(
                TaskId.POLISH_IMAGE,
                min(85, 30 + (attempt * 20)),
                "Still waiting for polished image to be created",
            )
        raise TimeoutError(
            "Object was not detected clearly enough within timeout window."
        )

        while not self._image_polish_finished:
            print(
                f"[Orchestrator] Waiting for image polish to finish (Status: {self._image_polish_finished})"
            )
            await asyncio.sleep(1)

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
        output_path = (
            output_dir / f"polished_product{_extension_for_mime(image_mime_type)}"
        )

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

        await self._agent.simple_response(
            "Confirm that the image is ready and tell the user that the product research is starting."
        )

        await self._emit(TaskId.RESEARCH_PRODUCT, 100, "Product research complete")

    async def _generate_description_step(self) -> ProductDescription:
        await self._emit(
            TaskId.GENERATE_DESCRIPTION, 0, "Generating structured description"
        )
        summary = (
            self._artifacts.product_search_summary or "Well-presented second-hand item."
        )
        facts = self._artifacts.product_search_facts
        title = (facts[0] if facts else "Pre-owned Product").strip()[:80]
        short = summary[:220]
        long_desc = f"{summary}\n\nHighlights:\n" + (
            "\n".join(f"- {fact}" for fact in facts[:8])
            if facts
            else "- Quality item in good shape."
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
        await self._emit(
            TaskId.GENERATE_DESCRIPTION, 100, "Structured description ready"
        )
        return description


def _extension_for_mime(image_mime_type: str | None) -> str:
    if image_mime_type == "image/jpeg":
        return ".jpg"
    if image_mime_type == "image/webp":
        return ".webp"
    if image_mime_type == "image/gif":
        return ".gif"
    return ".png"
