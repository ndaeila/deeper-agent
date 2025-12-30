"""Contracts for the Planner node."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from odr.agents.retriever.shared.contracts import WorkerType


class EvidenceRequirement(BaseModel):
    """Evidence requirements associated with a task or overall run."""

    urls_required: bool = True
    min_unique_urls: int = 1


class TaskSpec(BaseModel):
    """A structured task assigned by the planner."""

    # Planner may omit; Retriever will assign stable ids (worker_1..worker_N).
    worker_id: str | None = None
    worker_type: WorkerType = WorkerType.BROWSER_USE
    task_description: str
    deliverables: list[str] = Field(default_factory=list)
    requirements: EvidenceRequirement = Field(default_factory=EvidenceRequirement)
    # Optional runtime knobs for workers (e.g., browser search budget). This is merged into
    # WorkerTask["requirements"] to keep worker contracts stable/simple.
    runtime: dict[str, Any] = Field(default_factory=dict)


class ResearchPlan(BaseModel):
    """A structured plan produced by the planner."""

    tasks: list[TaskSpec] = Field(min_length=1)



