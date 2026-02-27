"""Typed models for the product-page task workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TaskId(StrEnum):
    """Stable task identifiers shared with the frontend."""

    DETECT_OBJECT = "detect-object"
    CAPTURE_SCREENSHOT = "capture-screenshot"
    POLISH_IMAGE = "polish-image"
    RESEARCH_PRODUCT = "research-product"
    GENERATE_DESCRIPTION = "generate-description"


class TaskState(StrEnum):
    """Task execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class TaskProgress:
    """Progress update emitted during workflow execution."""

    task_id: TaskId
    progress: int
    message: str
    state: TaskState = TaskState.RUNNING


@dataclass(slots=True)
class WorkflowArtifacts:
    """Artifacts accumulated across workflow steps."""

    source_image_path: str | None = None
    polished_image_url: str | None = None
    product_search_facts: list[str] = field(default_factory=list)
    product_search_summary: str | None = None

