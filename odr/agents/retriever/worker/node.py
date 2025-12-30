"""Worker execution node for the Retriever agent.

Contains only the per-worker execution logic (`worker_node`). Graph wiring lives in `graph.py`.
"""

from __future__ import annotations

import random
from typing import Any, Sequence, cast

from langchain_core.language_models import BaseChatModel

from odr.agents.types import WorkerResultExtended
from odr.agents.workers.base import WorkerFactory
from odr.agents.workers.llm_worker import LLMWorkerFactory
from odr.agents.retriever.shared.utils import normalize_worker_result
from odr.agents.retriever.state import WorkerState


def worker_node(
    state: WorkerState,
    llm: BaseChatModel,
    worker_factories: Sequence[WorkerFactory],
    rng: random.Random,
) -> dict[str, Any]:
    """Individual worker node that processes an assigned task."""
    task = state["task"]
    iteration = state.get("iteration", 1)

    factories: list[WorkerFactory] = (
        list(worker_factories) if worker_factories else [LLMWorkerFactory(llm)]
    )
    requested_type = (
        task.get("worker_type")
        if isinstance(task.get("worker_type"), str) and task.get("worker_type")
        else None
    )
    factory: WorkerFactory | None = None
    if requested_type:
        for f in factories:
            if getattr(f, "worker_type", None) == requested_type:
                factory = f
                break
    # Backward-compatible default (no worker_type set): pick first factory.
    if factory is None and factories:
        factory = factories[0]

    try:
        if factory is None:
            raise RuntimeError("No worker factories configured")
        worker = factory.create(task["worker_id"])
        result = worker.run(task=task, input_text=state["input"], iteration=iteration)
    except Exception as e:
        result = WorkerResultExtended(
            worker_id=task["worker_id"],
            findings=f"Worker failed: {e}",
            success=False,
            iteration=iteration,
            worker_type=requested_type or getattr(factory, "worker_type", "unknown"),
            error=str(e),
        )

    # Keep downstream nodes simple: ensure required keys exist even for misbehaving workers.
    result.setdefault("worker_id", task["worker_id"])
    result.setdefault("iteration", iteration)
    result.setdefault("success", True)
    result.setdefault("findings", "")

    # Normalize evidence so downstream compilation/judgment can rely on URLs when available.
    result = normalize_worker_result(cast(WorkerResultExtended, result))

    return {"worker_results": [cast(WorkerResultExtended, result)]}



