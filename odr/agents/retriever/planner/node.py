"""Planning utilities for the Retriever agent.

This module contains the supervisor/planning logic that turns an input query plus
prior context into a set of worker tasks.
"""

from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from odr.agents.types import WorkerTask
from odr.agents.retriever.state import RetrieverState
from odr.agents.retriever.shared.contracts import WorkerType
from odr.factory import DefaultLLMFactory

from .contracts import ResearchPlan
from .utils import plan_to_worker_tasks


_CHOOSE_WORKERS_SYSTEM_PROMPT = """You are a supervisor coordinating research workers.
Analyze the input request and create task descriptions for workers to investigate.
Each worker should focus on a specific, non-overlapping aspect of the SAME research task.

CRITICAL SCOPE LOCK / ANTI-DRIFT:
- You MUST stay aligned to the user's original query and any embedded structure (paper title/section/goal/questions).
- Do NOT change the topic, do NOT change the section, and no matter what, do NOT broaden into generic OSINT theory, even if the user query explicitly asks for it.
- If "Planning guidance" is provided, treat it as authoritative and follow it verbatim.

TASK QUALITY BAR (atomic + unambiguous):
- Each task_description must be concrete and execution-oriented (not a vague theme); 
  though instead of prescribing how to do the task, just focus on describing what you want from the task, and trust the workers to deliver. 
- Each task MUST include 2-5 explicit sub-steps (e.g. what to extract).
- Each task MUST include evidence requirements with URLs, and why you believe the task has been fulfilled by that URL evidence or by multiple URLs.
- Prefer primary/official sources over aggregators.
- Provide more detail and context to the workers to help them understand the task and the user's original query.

Return ONLY valid JSON in the following schema:
{
  "tasks": [
    {
      "worker_type": "browser_use|llm",
      "task_description": "string (what to do)",
      "deliverables": ["string", "..."],  // optional
      "requirements": {                   // optional
        "urls_required": true,
        "min_unique_urls": 1
      }
    }
  ]
}

Rules:
- Create between 1 and MAX_WORKERS tasks.
- Keep tasks focused and non-overlapping.
"""


def _build_previous_context(state: RetrieverState) -> str:
    """Build planning context from previous attempts (pure helper for tests)."""
    iteration = state.get("iteration_count", 0)
    ctx = ""
    if iteration > 0 and state.get("compiled_output"):
        ctx = f"\n\nPrevious attempt output:\n{state['compiled_output']}"
    if iteration > 0 and state.get("compiled_report"):
        ctx += "\n\nPrevious structured report:\n" + json.dumps(
            state["compiled_report"], indent=2, default=str
        )
    judge_feedback = state.get("judge_feedback")
    if judge_feedback:
        ctx += f"\n\nJudge feedback from last iteration:\n{judge_feedback}"
    next_guidance = state.get("next_worker_guidance")
    if next_guidance:
        ctx += f"\n\nPlanning guidance:\n{next_guidance}"
    return ctx


def _build_choose_workers_human_prompt(*, input_text: str, previous_context: str, max_workers: int) -> str:
    """Build the human prompt (pure helper for tests)."""
    return (
        f"Research task:\n{input_text}{previous_context}\n\n"
        f"Create focused task descriptions for up to {max_workers} parallel workers."
    )


def choose_workers(state: RetrieverState, llm: BaseChatModel | None = None, llm_factory: DefaultLLMFactory | None = None) -> dict[str, Any]:
    """Supervisor node that decides which workers should handle the task.

    This node analyzes the input and creates task assignments for workers.

    DATA FLOW (deliverables generation):
    1. This LLM prompt generates JSON with `deliverables: ["...", ...]`
    2. `plan_to_worker_tasks` (contracts.py) formats them into WorkerTask.context
    3. BrowserUseWorker extracts them from context and embeds into agent_task prompt

    TUNING: To improve deliverable quality, edit the system_prompt below.

    Args:
        state: Current graph state.
        llm: Language model for decision making.
        llm_factory: Factory for creating LLMs.

    Returns:
        Updated state with worker_tasks assigned.
    """
    if llm_factory:
        llm = llm_factory.get_llm(name="retriever-supervisor")
    
    if not llm:
        raise ValueError("Either llm or llm_factory must be provided")

    input_text = state["input"]  # ← Usually: user's original query
    iteration = state.get("iteration_count", 0)  # ← Usually: 0, 1, 2, ...
    max_workers = state.get("max_workers", 5)  # ← Usually: set in Retriever.__init__

    # Build context from previous attempts if any (← Usually: empty on iteration 0)
    previous_context = _build_previous_context(state)

    # ========================================================================
    # THIS PROMPT GENERATES THE DELIVERABLES
    # ========================================================================
    messages = [
        SystemMessage(content=_CHOOSE_WORKERS_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_choose_workers_human_prompt(
                input_text=input_text,
                previous_context=previous_context,
                max_workers=max_workers,
            )
        ),
    ]

    # Prefer structured output when supported; fall back to JSON parsing.
    worker_tasks: list[WorkerTask]
    response_message: AIMessage
    try:
        with_structured = getattr(llm, "with_structured_output", None)
        if callable(with_structured):
            # Use function calling for broad compatibility with OpenAI-compatible endpoints.
            try:
                planner = cast(Any, with_structured(ResearchPlan, method="function_calling"))
            except TypeError:
                planner = cast(Any, with_structured(ResearchPlan))
            plan = cast(ResearchPlan, cast(Any, planner).invoke(messages, config={"run_name": "choose_workers"}))
            plan.tasks = plan.tasks[: max(1, int(max_workers))]
            response_message = AIMessage(content=plan.model_dump_json())
            worker_tasks = plan_to_worker_tasks(plan=plan, raw_context=str(response_message.content))
        else:
            raise RuntimeError("LLM does not support structured output")
    except Exception:
        response = llm.invoke(messages, config={"run_name": "choose_workers"})
        response_text = response.content if isinstance(response.content, str) else str(response.content)
        response_message = (
            cast(AIMessage, response) if isinstance(response, AIMessage) else AIMessage(content=response_text)
        )
        worker_tasks = _parse_worker_tasks_from_supervisor_json(
            response_text=response_text,
            input_text=input_text,
            max_workers=max_workers,
        )

    return {
        "worker_tasks": worker_tasks,
        "messages": [response_message],
        "iteration_count": iteration + 1,
    }


def _parse_worker_tasks_from_supervisor_json(
    response_text: str,
    input_text: str,
    max_workers: int,
) -> list[WorkerTask]:
    """Parse worker tasks from supervisor JSON output.

    Falls back to a simple 1-2 worker setup if parsing fails.
    """
    max_workers = max(1, min(5, int(max_workers)))

    def _fallback() -> list[WorkerTask]:
        # Preserve previous behavior but respect max_workers.
        tasks: list[WorkerTask] = [
            WorkerTask(
                worker_id="worker_1",
                task_description=f"Primary investigation: {input_text}",
                context=response_text,
                worker_type=WorkerType.BROWSER_USE.value,
            )
        ]
        if max_workers >= 2:
            tasks.append(
                WorkerTask(
                    worker_id="worker_2",
                    task_description=f"Secondary verification: {input_text}",
                    context=response_text,
                    worker_type=WorkerType.BROWSER_USE.value,
                )
            )
        return tasks

    try:
        parsed = json.loads(response_text)
    except Exception:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return _fallback()
        try:
            parsed = json.loads(response_text[start : end + 1])
        except Exception:
            return _fallback()

    tasks_raw = parsed.get("tasks") if isinstance(parsed, dict) else None
    if not isinstance(tasks_raw, list) or not tasks_raw:
        return _fallback()

    worker_tasks: list[WorkerTask] = []
    for idx, item in enumerate(tasks_raw[:max_workers], start=1):
        if not isinstance(item, dict):
            continue
        desc = item.get("task_description")
        if not isinstance(desc, str) or not desc.strip():
            continue
        worker_type_raw = item.get("worker_type")
        worker_type = (
            worker_type_raw.strip()
            if isinstance(worker_type_raw, str) and worker_type_raw.strip()
            else WorkerType.BROWSER_USE.value
        )
        if worker_type not in (WorkerType.BROWSER_USE.value, WorkerType.LLM.value):
            worker_type = WorkerType.BROWSER_USE.value
        deliverables = item.get("deliverables")
        deliverables_txt = ""
        deliverables_list: list[str] = []
        if isinstance(deliverables, list):
            deliverables_strs = [d for d in deliverables if isinstance(d, str) and d.strip()]
            if deliverables_strs:
                deliverables_list = deliverables_strs
                deliverables_txt = "\nDeliverables:\n- " + "\n- ".join(deliverables_strs)
        requirements = item.get("requirements") if isinstance(item.get("requirements"), dict) else {}

        worker_tasks.append(
            WorkerTask(
                worker_id=f"worker_{idx}",
                task_description=desc.strip(),
                context=(response_text + deliverables_txt).strip(),
                worker_type=worker_type,
                deliverables=deliverables_list,
                requirements=cast(dict[str, Any], requirements),
            )
        )

    return worker_tasks if worker_tasks else _fallback()

