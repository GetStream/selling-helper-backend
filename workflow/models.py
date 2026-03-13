"""Typed models for the product-page task workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

# from enum import StrEnum
from enum import Enum

from pydantic import BaseModel, Field


class ProductDetails(BaseModel):
    name: str = Field(description="The name of the product.")
    description: str = Field(description="The description of the product.")
    specifications: str = Field(description="The specifications of the product.")
    pricing: float = Field(description="The pricing of the product in USD.")


class TaskId(Enum):
    """Stable task identifiers shared with the frontend."""

    DETECT_OBJECT = "detect-object"
    CAPTURE_SCREENSHOT = "capture-screenshot"
    POLISH_IMAGE = "polish-image"
    RESEARCH_PRODUCT = "research-product"
    GENERATE_DESCRIPTION = "generate-description"


class TaskState(Enum):
    """Task execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskProgress:
    """Progress update emitted during workflow execution."""

    task_id: TaskId
    progress: int
    message: str
    state: TaskState = TaskState.RUNNING


@dataclass
class WorkflowArtifacts:
    """Artifacts accumulated across workflow steps."""

    source_image_path: str | None = None
    polished_image_url: str | None = None
    polished_image_path: str | None = None
    product_search_facts: list[str] = field(default_factory=list)
    product_search_summary: str | None = None
