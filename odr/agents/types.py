"""Shared data types used across agents and workers.

This module exists to avoid circular imports between `odr.agents.retriever`
and worker implementations/factories.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, Required, TypedDict


class WorkerTask(TypedDict):
    """A task assigned to a worker.

    Required keys form the stable contract across the graph. Optional keys allow
    the planner to evolve (e.g., routing hints) without breaking older callers.
    """

    worker_id: Required[str]
    task_description: Required[str]
    context: Required[str]
    # Preferred: deterministic routing by worker type (e.g., 'browser_use', 'llm')
    worker_type: NotRequired[str]
    # Optional: planner-specified deliverables/requirements (kept for traceability)
    deliverables: NotRequired[list[str]]
    requirements: NotRequired[dict[str, Any]]


class WorkerResult(TypedDict):
    """Base result from a worker's execution."""

    worker_id: str
    findings: str
    success: bool
    iteration: int


class WorkerResultExtended(WorkerResult, total=False):
    """Optional extended fields attached by specialized workers."""

    worker_type: str
    evidence: list[dict[str, Any]]
    context_window: dict[str, Any]
    raw_history: Any
    error: str


