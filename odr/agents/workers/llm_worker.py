"""Default LLM-backed worker used by Retriever when no specialized workers are provided."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from typing import Any, Mapping, cast

from odr.agents.types import WorkerResultExtended, WorkerTask
from odr.agents.workers.base import WorkerFactory


class LLMResearchWorker:
    """A simple worker that uses a provided LLM to complete the assigned task."""

    worker_type = "llm"

    def __init__(self, worker_id: str, llm: BaseChatModel):
        self.worker_id = worker_id
        self.llm = llm

    def run(self, task: Mapping[str, Any], input_text: str, iteration: int) -> WorkerResultExtended:
        task_typed = cast(WorkerTask, task)
        system_prompt = f"""You are a research worker with ID: {task['worker_id']}.
Your job is to investigate the assigned task and provide detailed findings.
Be thorough and factual in your research.

IMPORTANT: Prefer evidence-backed statements and cite URLs/snippets if available.
If you cannot verify a claim, explicitly mark it as unverified."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Task: {task_typed['task_description']}\n\n"
                    f"Context from supervisor: {task_typed['context']}"
                )
            ),
        ]

        # Prefer structured output (findings + evidence[] with URLs), with fallback to plain text.
        content: str
        evidence: list[dict[str, Any]]
        try:
            with_structured = getattr(self.llm, "with_structured_output", None)
            if callable(with_structured):
                # Lazy import to avoid circular dependency
                from odr.agents.retriever.contracts import WorkerFinding

                try:
                    runner = cast(Any, with_structured(WorkerFinding, method="function_calling"))
                except TypeError:
                    runner = cast(Any, with_structured(WorkerFinding))
                wf = cast(WorkerFinding, cast(Any, runner).invoke(messages))
                content = wf.findings
                evidence = [e.model_dump() for e in wf.evidence]
            else:
                raise RuntimeError("LLM does not support structured output")
        except Exception:
            response = self.llm.invoke(messages)
            content = response.content if isinstance(response.content, str) else str(response.content)
            evidence = []

        return WorkerResultExtended(
            worker_id=task_typed["worker_id"],
            findings=content,
            success=True,
            iteration=iteration,
            worker_type=self.worker_type,
            evidence=evidence,
            context_window={
                "system_prompt": system_prompt,
                "user_prompt": messages[1].content,
                "model_response": content,
            },
        )


class LLMWorkerFactory(WorkerFactory):
    """Factory for creating LLMResearchWorker instances."""

    worker_type = "llm"

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def create(self, worker_id: str) -> LLMResearchWorker:
        return LLMResearchWorker(worker_id=worker_id, llm=self.llm)


