"""Video processor that detects readiness and captures a source frame."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import aiortc
import av
import numpy as np
from vision_agents.core.processors import VideoProcessor
from vision_agents.core.utils.video_forwarder import VideoForwarder


class ObjectCaptureProcessor(VideoProcessor):
    """Detect when object is clearly visible and capture the best frame."""

    name = "object_capture"

    def __init__(
        self,
        fps: int = 8,
        min_ready_score: float = 0.62,
        consecutive_ready_frames: int = 6,
        source_frame_filename: str = "captured_product_frame.png",
    ) -> None:
        self.fps = fps
        self.min_ready_score = min_ready_score
        self.consecutive_ready_frames = consecutive_ready_frames
        self.source_frame_filename = source_frame_filename

        self._forwarder: VideoForwarder | None = None
        self._ready_event = asyncio.Event()
        self._ready_streak = 0
        self._latest_guidance = "Center the item and hold it steady."
        self._best_score = 0.0
        self._best_frame: av.VideoFrame | None = None

    @property
    def guidance_message(self) -> str:
        return self._latest_guidance

    async def process_video(
        self,
        track: aiortc.VideoStreamTrack,
        participant_id: str | None,
        shared_forwarder: VideoForwarder | None = None,
    ) -> None:
        if shared_forwarder is None:
            raise RuntimeError("ObjectCaptureProcessor requires a shared VideoForwarder.")

        if self._forwarder is not None:
            await self._forwarder.remove_frame_handler(self._handle_frame)

        self._forwarder = shared_forwarder
        self._forwarder.add_frame_handler(
            self._handle_frame,
            fps=float(self.fps),
            name=self.name,
        )

    async def _handle_frame(self, frame: av.VideoFrame) -> None:
        score, guidance = self._analyze_frame(frame)
        self._latest_guidance = guidance

        if score > self._best_score:
            self._best_score = score
            self._best_frame = frame

        if score >= self.min_ready_score:
            self._ready_streak += 1
        else:
            self._ready_streak = 0

        if self._ready_streak >= self.consecutive_ready_frames:
            self._ready_event.set()

    def _analyze_frame(self, frame: av.VideoFrame) -> tuple[float, str]:
        rgb = frame.to_ndarray(format="rgb24").astype(np.float32)
        gray = rgb.mean(axis=2)

        brightness = float(gray.mean())
        contrast = float(gray.std())

        dx = np.abs(np.diff(gray, axis=1)).mean()
        dy = np.abs(np.diff(gray, axis=0)).mean()
        edge_strength = float((dx + dy) / 2.0)

        brightness_score = 1.0 - min(abs(brightness - 130.0) / 130.0, 1.0)
        contrast_score = min(contrast / 55.0, 1.0)
        edge_score = min(edge_strength / 18.0, 1.0)

        score = (0.3 * brightness_score) + (0.35 * contrast_score) + (0.35 * edge_score)

        if brightness < 65:
            return score, "Increase lighting so the product is clearly visible."
        if brightness > 210:
            return score, "Reduce glare or move away from strong direct light."
        if contrast < 28:
            return score, "Move closer and keep the item sharp in focus."
        if edge_strength < 8:
            return score, "Hold still for one second and center the product in frame."
        return score, "Great framing. Hold still while I capture the photo."

    async def wait_until_ready(self, timeout_seconds: float = 20.0) -> bool:
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout_seconds)
            return True
        except TimeoutError:
            return False

    async def capture_best_frame(self, output_dir: Path) -> Path:
        if self._best_frame is None:
            raise RuntimeError("No candidate frame available for screenshot capture.")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.source_frame_filename
        image = self._best_frame.to_image()
        image.save(output_path)
        return output_path

    async def stop_processing(self) -> None:
        if self._forwarder is not None:
            await self._forwarder.remove_frame_handler(self._handle_frame)
            self._forwarder = None

    async def close(self) -> None:
        await self.stop_processing()


class StubObjectCaptureProcessor:

    """Fallback processor used in tests or non-video environments."""

    def __init__(self) -> None:
        self.guidance_message = "Center the item and hold still."
        self._is_ready = True
        self.source_frame_filename = "captured_product_frame.png"

    async def wait_until_ready(self, timeout_seconds: float = 20.0) -> bool:  # noqa: ARG002
        return self._is_ready

    async def capture_best_frame(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.source_frame_filename
        # 1x1 transparent PNG so downstream image tooling receives a valid image.
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+XWfQAAAAASUVORK5CYII="
        )
        output_path.write_bytes(png_bytes)
        return output_path

    async def stop_processing(self) -> None:
        return None

    async def close(self) -> None:
        return None

