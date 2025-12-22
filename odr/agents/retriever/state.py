"""State definitions for the Retriever agent graph.

This module contains the TypedDicts that flow through the graph as state.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage

from odr.agents.judge_counsel import JudgmentDecision
from odr.agents.types import WorkerResultExtended, WorkerTask


class RetrieverState(TypedDict):
    """State for the Retriever agent graph.

    Attributes:
        input: The original input query or task.
        messages: Conversation history for context.
        worker_tasks: List of tasks assigned to workers.
        worker_results: Results collected from workers (uses operator.add for parallel merge).
        compiled_output: The aggregated output from all worker results.
        judgment_decision: The decision from the judgment step.
        iteration_count: Number of iterations through the loop.
        max_iterations: Maximum allowed iterations before forced exit.
        max_workers: Maximum number of workers per iteration.
    """

    input: str
    messages: Annotated[list[BaseMessage], operator.add]
    worker_tasks: list[WorkerTask]
    worker_results: Annotated[list[WorkerResultExtended], operator.add]
    compiled_output: str
    judgment_decision: JudgmentDecision | None
    iteration_count: int
    max_iterations: int
    max_workers: int
    # Closed-loop fields (v2)
    judge_feedback: str | None
    compiled_report: dict | None
    next_action: str | None
    next_worker_guidance: str | None
    final_status: str | None


class WorkerState(TypedDict):
    """State passed to individual worker nodes."""

    task: WorkerTask
    input: str
    iteration: int

