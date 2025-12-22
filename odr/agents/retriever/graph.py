"""LangGraph wiring for the Retriever agent.

This module contains only graph construction logic: adding nodes, edges,
and conditional routing. Node implementations live in `nodes.py`.
"""

from __future__ import annotations

import random
from typing import Any, Sequence, cast

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from odr.agents.judge_counsel import JudgeCounsel
from odr.agents.workers.base import WorkerFactory
from odr.agents.workers.llm_worker import LLMWorkerFactory

from .nodes import (
    choose_workers,
    decide_next_action,
    judgment,
    observe_and_compile,
    route_after_decision,
    worker_node,
)
from .state import RetrieverState, WorkerState


def fan_out_to_workers(state: RetrieverState) -> list[Send]:
    """Create Send commands for parallel worker execution.

    Args:
        state: Current graph state with worker tasks.

    Returns:
        List of Send commands, one for each worker task.
    """
    worker_tasks = state.get("worker_tasks", [])
    iteration = state.get("iteration_count", 1)

    return [
        Send(
            "worker",
            WorkerState(task=task, input=state["input"], iteration=iteration),
        )
        for task in worker_tasks
    ]


def create_retriever_graph(
    llm: BaseChatModel,
    judge_counsel: JudgeCounsel | None = None,
    worker_factories: Sequence[WorkerFactory] | None = None,
    worker_factory_seed: int | None = None,
) -> CompiledStateGraph:
    """Create the Retriever agent graph.

    The graph implements the following flow:
    1. choose_workers: Supervisor assigns tasks to workers
    2. worker (fan-out): Multiple workers process tasks in parallel
    3. observe_and_compile: Aggregate worker findings
    4. judgment: Judge Counsel evaluates via multi-judge voting
    5. Either loop back to choose_workers or end

    Args:
        llm: Language model for supervisor, workers, and observer.
        judge_counsel: Judge Counsel agent for judgment (uses default model if not provided).
        worker_factories: Factories to create workers (defaults to LLMWorkerFactory).
        worker_factory_seed: Seed for random worker factory selection.

    Returns:
        Compiled StateGraph ready for execution.
    """
    # Create Judge Counsel if not provided
    if judge_counsel is None:
        judge_counsel = JudgeCounsel()

    factories: list[WorkerFactory] = (
        list(worker_factories) if worker_factories else [LLMWorkerFactory(llm)]
    )
    rng = random.Random(worker_factory_seed)

    # Create graph with state schema
    graph = StateGraph(RetrieverState)

    # Add nodes with LLM bound
    # Using explicit functions to avoid type narrowing issues with lambdas
    def _choose_workers(state: Any) -> dict[str, Any]:
        return choose_workers(cast(RetrieverState, state), llm)

    def _worker(state: Any) -> dict[str, Any]:
        return worker_node(cast(WorkerState, state), llm, factories, rng)

    def _observe_and_compile(state: Any) -> dict[str, Any]:
        return observe_and_compile(cast(RetrieverState, state), llm)

    def _judgment(state: Any) -> dict[str, Any]:
        return judgment(cast(RetrieverState, state), judge_counsel)

    def _decide_next_action(state: Any) -> dict[str, Any]:
        return decide_next_action(cast(RetrieverState, state), llm)

    graph.add_node("choose_workers", _choose_workers)
    graph.add_node("worker", _worker)
    graph.add_node("observe_and_compile", _observe_and_compile)
    graph.add_node("judgment", _judgment)
    graph.add_node("decide_next_action", _decide_next_action)

    # Set entry point
    graph.set_entry_point("choose_workers")

    # Add edges
    # After choosing workers, fan out to all workers in parallel
    # When using Send, the path_map is not needed - targets come from Send objects
    graph.add_conditional_edges(
        "choose_workers",
        fan_out_to_workers,  # type: ignore[arg-type]
    )

    # Workers converge to observe_and_compile
    graph.add_edge("worker", "observe_and_compile")

    # After compilation, go to judgment
    graph.add_edge("observe_and_compile", "judgment")

    # Judgment feeds into the closed-loop controller
    graph.add_edge("judgment", "decide_next_action")

    # Decision node determines whether to continue or end
    graph.add_conditional_edges(
        "decide_next_action",
        route_after_decision,
        {
            "choose_workers": "choose_workers",
            END: END,
        },
    )

    return graph.compile()

