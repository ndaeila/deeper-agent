"""Base interfaces for Retriever worker implementations."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from odr.agents.types import WorkerResultExtended, WorkerTask


class Worker(Protocol):
    """A runnable worker implementation.

    Workers should focus on producing evidence-backed findings for their assigned task.
    """

    worker_type: str

    def run(self, task: WorkerTask, input_text: str, iteration: int) -> WorkerResultExtended:
        """Run the worker for a task.

        Args:
            task: The assigned worker task.
            input_text: The original user input.
            iteration: Current retriever iteration.

        Returns:
            A WorkerResultExtended containing findings and (optionally) structured evidence/context.
        """
        ...


class WorkerFactory(Protocol):
    """Factory for creating workers."""

    worker_type: str

    def create(self, worker_id: str) -> Worker:
        """Create a new worker instance.

        Args:
            worker_id: The worker id for traceability.

        Returns:
            A Worker instance.
        """
        ...


