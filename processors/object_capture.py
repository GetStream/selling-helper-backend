"""Video processor that detects readiness and captures a source frame."""

from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path

import aiortc
import av
import numpy as np
from vision_agents.core.processors import VideoProcessor
from vision_agents.core.utils.video_forwarder import VideoForwarder

from helpers import get_date_and_time_now


class CaptureStatus(Enum):
    """Status of the capture process."""

    INACTIVE = "inactive"
    DETECTING = "detecting"
    CAPTURING = "capturing"


class ObjectCaptureProcessor(VideoProcessor):
    """Detect when object is clearly visible and capture the best frame."""

    name = "object_capture"

    def __init__(
        self,
        fps: int = 8,
        min_ready_score: float = 0.42,
        consecutive_ready_frames: int = 6,
        source_frame_filename: str = "captured_product_frame",
    ) -> None:
        self.fps = fps
        self.capture_status = CaptureStatus.INACTIVE
        self.min_ready_score = min_ready_score
        self.consecutive_ready_frames = consecutive_ready_frames
        self.source_frame_filename = source_frame_filename

        self._forwarder: VideoForwarder | None = None
        self._ready_event = asyncio.Event()
        self._ready_streak = 0
        self._latest_guidance = "Tell the user to hold the item in the center of the webcam and hold it steady."
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
        # If the video forwarder is not provided, raise an error.
        if shared_forwarder is None:
            raise RuntimeError(
                "ObjectCaptureProcessor requires a shared VideoForwarder."
            )

        # If the forwarder is already set, remove the existing frame handler.
        if self._forwarder is not None:
            await self._forwarder.remove_frame_handler(self._handle_frame)

        # Set the new forwarder and add the frame handler.
        self._forwarder = shared_forwarder
        self._forwarder.add_frame_handler(
            self._handle_frame,
            fps=float(self.fps),
            name=self.name,
        )

    async def update_capture_status(self, capture_status: CaptureStatus) -> None:
        print(f"[ObjectCaptureProcessor] Updating capture status to {capture_status}")
        self.capture_status = capture_status

    async def _handle_frame(self, frame: av.VideoFrame) -> None:
        print(
            f"[ObjectCaptureProcessor] Handling frame (Capture status: {self.capture_status})"
        )
        # If the capture is inactive, skip frame analysis.
        if self.capture_status == CaptureStatus.INACTIVE:
            print(
                "[ObjectCaptureProcessor] Capture is inactive, skipping frame analysis"
            )
            return

        if self.capture_status == CaptureStatus.DETECTING:
            print("[ObjectCaptureProcessor] Analyzing frame")
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
                print(
                    "[ObjectCaptureProcessor] Ready streak reached, setting ready event"
                )
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
        print(f"[ObjectCaptureProcessor] Score: {score}")
        if brightness < 65:
            return score, "Increase lighting so the product is clearly visible."
        if brightness > 210:
            return score, "Reduce glare or move away from strong direct light."
        if contrast < 28:
            return score, "Move closer and keep the item sharp in focus."
        if edge_strength < 8:
            return score, "Hold still for one second and center the product in frame."
        return score, "Great framing. Hold still while I capture the photo."

    async def wait_until_ready(self, timeout_seconds: float = 30.0) -> bool:
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout_seconds)
            return True
        except TimeoutError:
            return False

    async def capture_best_frame(self, output_dir: Path) -> Path:
        if self._best_frame is None:
            raise RuntimeError("No candidate frame available for screenshot capture.")

        if self.capture_status != CaptureStatus.CAPTURING:
            print(
                "[ObjectCaptureProcessor] Capture is not active, skipping frame capture"
            )
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir / f"{self.source_frame_filename}-{get_date_and_time_now()}.png"
        )
        image = self._best_frame.to_image()
        image.save(output_path)
        print(f"[ObjectCaptureProcessor] Frame captured, output path: {output_path}")
        return output_path

    async def stop_processing(self) -> None:
        if self._forwarder is not None:
            await self._forwarder.remove_frame_handler(self._handle_frame)
            self._forwarder = None

    async def close(self) -> None:
        await self.stop_processing()
