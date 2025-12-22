"""Retriever agent wrapper class.

This module contains the high-level `Retriever` class that wraps the
LangGraph workflow. It handles configuration, initial state construction,
and provides convenient run/stream methods.
"""

from __future__ import annotations

from typing import Sequence, cast

from langchain_core.language_models import BaseChatModel

from odr.agents.judge_counsel import JudgeCounsel
from odr.agents.workers.base import WorkerFactory

from .graph import create_retriever_graph
from .state import RetrieverState


class Retriever:
    """Retriever agent with supervisor pattern and worker fan-out.

    This agent coordinates multiple workers to investigate a query,
    compiles their findings, and uses a Judge Counsel (multi-judge voting)
    to determine if the results are satisfactory or need refinement.

    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from odr.agents.judge_counsel import JudgeCounsel

        llm = ChatOpenAI(model="gpt-4")
        # Optional: customize the Judge Counsel model
        counsel = JudgeCounsel()  # Uses default model

        retriever = Retriever(llm=llm, judge_counsel=counsel)

        result = retriever.run("What is the capital of France?")
        print(result["compiled_output"])
        ```
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_iterations: int = 3,
        max_workers: int = 2,
        judge_counsel: JudgeCounsel | None = None,
        worker_factories: Sequence[WorkerFactory] | None = None,
        worker_factory_seed: int | None = None,
    ):
        """Initialize the Retriever agent.

        Args:
            llm: Language model for supervisor, workers, and observer.
            max_iterations: Maximum number of retry iterations allowed.
            max_workers: Maximum number of workers per iteration (1-5, default 2 for browser workloads).
            judge_counsel: Judge Counsel agent for judgment.
                          Defaults to default model if not provided.
            worker_factories: Factories for creating workers (defaults to LLMWorkerFactory).
            worker_factory_seed: Seed for deterministic worker factory selection.
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_workers = max_workers
        self.judge_counsel = judge_counsel or JudgeCounsel()
        self.worker_factories = worker_factories
        self.worker_factory_seed = worker_factory_seed
        self.graph = create_retriever_graph(
            llm,
            self.judge_counsel,
            worker_factories=self.worker_factories,
            worker_factory_seed=self.worker_factory_seed,
        )

    def _build_initial_state(self, input_text: str) -> RetrieverState:
        """Build the initial state for a run."""
        return RetrieverState(
            input=input_text,
            messages=[],
            worker_tasks=[],
            worker_results=[],
            compiled_output="",
            judgment_decision=None,
            judge_feedback=None,
            compiled_report=None,
            next_action=None,
            next_worker_guidance=None,
            final_status=None,
            iteration_count=0,
            max_iterations=self.max_iterations,
            max_workers=self.max_workers,
        )

    def run(self, input_text: str) -> RetrieverState:
        """Execute the retriever workflow.

        Args:
            input_text: The query or task to investigate.

        Returns:
            Final state containing compiled output and all intermediate results.
        """
        initial_state = self._build_initial_state(input_text)
        return cast(RetrieverState, self.graph.invoke(initial_state))

    async def arun(self, input_text: str) -> RetrieverState:
        """Execute the retriever workflow asynchronously.

        Args:
            input_text: The query or task to investigate.

        Returns:
            Final state containing compiled output and all intermediate results.
        """
        initial_state = self._build_initial_state(input_text)
        return cast(RetrieverState, await self.graph.ainvoke(initial_state))

    def stream(self, input_text: str):
        """Stream the retriever workflow execution.

        Args:
            input_text: The query or task to investigate.

        Yields:
            State updates as the graph progresses through nodes.
        """
        initial_state = self._build_initial_state(input_text)
        yield from self.graph.stream(initial_state)

