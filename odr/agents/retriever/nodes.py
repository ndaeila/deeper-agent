"""Node functions for the Retriever agent graph.

Each function in this module represents a single step (node) in the
LangGraph workflow. Nodes are pure-ish functions: they take state + deps
and return a minimal patch to be merged into state.
"""

from __future__ import annotations

import json
import random
from typing import Any, Sequence, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from odr.agents.judge_counsel import JudgeCounsel, JudgmentDecision
from odr.agents.types import WorkerResultExtended, WorkerTask
from odr.agents.workers.base import WorkerFactory
from odr.agents.workers.llm_worker import LLMWorkerFactory

from .contracts import (
    CompiledReport,
    NextAction,
    NextActionDecision,
    ResearchPlan,
    WorkerType,
    normalize_worker_result,
    plan_to_worker_tasks,
    render_compiled_report,
    unique_urls_from_results,
 )
from .state import RetrieverState, WorkerState


# ---------------------------------------------------------------------------
# choose_workers node
# ---------------------------------------------------------------------------


def choose_workers(state: RetrieverState, llm: BaseChatModel) -> dict[str, Any]:
    """Supervisor node that decides which workers should handle the task.

    This node analyzes the input and creates task assignments for workers.

    Args:
        state: Current graph state.
        llm: Language model for decision making.

    Returns:
        Updated state with worker_tasks assigned.
    """
    input_text = state["input"]
    iteration = state.get("iteration_count", 0)
    max_workers = state.get("max_workers", 5)

    # Build context from previous attempts if any
    previous_context = ""
    if iteration > 0 and state.get("compiled_output"):
        previous_context = f"\n\nPrevious attempt output:\n{state['compiled_output']}"
    if iteration > 0 and state.get("compiled_report"):
        previous_context += (
            "\n\nPrevious structured report:\n"
            f"{json.dumps(state['compiled_report'], indent=2, default=str)}"
        )
    judge_feedback = state.get("judge_feedback")
    if judge_feedback:
        previous_context += f"\n\nJudge feedback from last iteration:\n{judge_feedback}"
    next_guidance = state.get("next_worker_guidance")
    if next_guidance:
        previous_context += f"\n\nPlanning guidance:\n{next_guidance}"

    system_prompt = """You are a supervisor coordinating research workers.
Analyze the input request and create task descriptions for workers to investigate.
Each worker should focus on a specific aspect of the research task.
Be specific about what each worker should find and return, including evidence requirements (URLs) where relevant.

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
- Prefer worker_type=browser_use unless the task is purely internal analysis.
- Do not include markdown fences or extra commentary—JSON only."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Research task: {input_text}{previous_context}\n\n"
            f"Create focused task descriptions for up to {max_workers} parallel workers."
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
            plan = cast(ResearchPlan, cast(Any, planner).invoke(messages))
            # Enforce max_workers + stable ids.
            plan.tasks = plan.tasks[: max(1, min(5, int(max_workers)))]
            response_message = AIMessage(content=plan.model_dump_json())
            worker_tasks = plan_to_worker_tasks(plan=plan, raw_context=str(response_message.content))
        else:
            raise RuntimeError("LLM does not support structured output")
    except Exception:
        response = llm.invoke(messages)
        response_text = response.content if isinstance(response.content, str) else str(response.content)
        response_message = cast(AIMessage, response) if isinstance(response, AIMessage) else AIMessage(content=response_text)
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


# ---------------------------------------------------------------------------
# worker_node
# ---------------------------------------------------------------------------


def worker_node(
    state: WorkerState,
    llm: BaseChatModel,
    worker_factories: Sequence[WorkerFactory],
    rng: random.Random,
) -> dict[str, Any]:
    """Individual worker node that processes an assigned task.

    Args:
        state: Worker-specific state with task details.
        llm: Language model for task execution.
        worker_factories: Available factories to create workers.
        rng: Random generator for factory selection.

    Returns:
        Worker result to be merged into parent state.
    """
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


# ---------------------------------------------------------------------------
# observe_and_compile node
# ---------------------------------------------------------------------------


def observe_and_compile(state: RetrieverState, llm: BaseChatModel) -> dict[str, Any]:
    """Observer node that aggregates findings from all workers.

    Args:
        state: Current graph state with worker results.
        llm: Language model for synthesis.

    Returns:
        Updated state with compiled output.
    """
    all_worker_results = state.get("worker_results", [])
    current_iteration = state.get("iteration_count", 1)
    original_input = state["input"]

    # Filter to only current iteration's results
    worker_results = [r for r in all_worker_results if r.get("iteration") == current_iteration]

    # Format worker findings for synthesis
    findings_text = "\n\n".join(
        _format_worker_result_for_consolidation(result) for result in worker_results
    )

    urls = sorted(unique_urls_from_results(worker_results))
    urls_block = "\n".join(f"- {u}" for u in urls[:25]) if urls else "(none)"
    evidence_items: list[dict[str, Any]] = []
    for r in worker_results:
        for ev in cast(list[dict[str, Any]], r.get("evidence") or []):
            url = ev.get("url")
            if not url:
                continue
            evidence_items.append(
                {
                    "worker_id": r.get("worker_id"),
                    "url": url,
                    "title": ev.get("title"),
                    "excerpt": ev.get("excerpt"),
                }
            )
    evidence_items = evidence_items[:30]

    system_prompt = """You are an analyst synthesizing research findings from multiple workers.
Produce a structured report with:
- answer: concise but complete, grounded in the findings
- claims: a list of factual claims. EACH claim must include explicit evidence items with URL + excerpt/snippet.
- citations: URLs (must be valid http(s)) that support the answer (a deduped convenience list)
- open_questions: remaining gaps
- limitations: constraints/uncertainty

CRITICAL:
- Preserve and include source URLs when present.
- Do not invent URLs; only use the provided URLs or URLs clearly present in worker findings."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Original query: {original_input}\n\n"
            f"Known source URLs from workers:\n{urls_block}\n\n"
            f"Normalized evidence items (url + excerpt):\n{json.dumps(evidence_items, indent=2, default=str)}\n\n"
            f"Worker findings to synthesize:\n{findings_text}"
        ),
    ]

    report: CompiledReport
    response_msg: AIMessage
    try:
        with_structured = getattr(llm, "with_structured_output", None)
        if callable(with_structured):
            try:
                runner = cast(Any, with_structured(CompiledReport, method="function_calling"))
            except TypeError:
                runner = cast(Any, with_structured(CompiledReport))
            report = cast(CompiledReport, cast(Any, runner).invoke(messages))
            if not report.citations:
                # Prefer citations derived from claim evidence; fall back to known URLs.
                claim_urls: list[str] = []
                for c in report.claims:
                    for ev in c.evidence:
                        claim_urls.append(str(ev.url))
                dedup = []
                seen: set[str] = set()
                for u in claim_urls:
                    if u not in seen:
                        seen.add(u)
                        dedup.append(u)
                use_urls = dedup or urls
                if use_urls:
                    report.citations = cast(Any, [{"url": u} for u in use_urls[:25]])
            response_msg = AIMessage(content=report.model_dump_json())
        else:
            raise RuntimeError("LLM does not support structured output")
    except Exception:
        response = llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        # Best-effort: keep URLs deterministically as citations if the model didn't structure them.
        citations = [{"url": u} for u in urls[:25]]
        report = CompiledReport(answer=content, citations=cast(Any, citations))
        response_msg = cast(AIMessage, response) if isinstance(response, AIMessage) else AIMessage(content=content)

    return {
        "compiled_report": report.model_dump(mode="json"),
        "compiled_output": render_compiled_report(report),
        "messages": [response_msg],
    }


def _format_worker_result_for_consolidation(result: WorkerResultExtended) -> str:
    """Format a single worker result for the observe_and_compile synthesis prompt."""
    worker_id = result.get("worker_id", "unknown").upper()
    worker_type = result.get("worker_type", "unknown")
    success = result.get("success", False)

    header = f"=== {worker_id} ({worker_type}) | success={success} ==="
    body = result.get("findings", "")

    evidence = result.get("evidence") or []
    evidence_block = ""
    if evidence:
        lines: list[str] = ["Evidence artifacts:"]
        for idx, ev in enumerate(evidence[:10], start=1):
            url = ev.get("url") or ev.get("source_url") or ""
            title = ev.get("title") or ""
            excerpt = ev.get("excerpt") or ""
            line = f"- [{idx}] {title} {url}".strip()
            if excerpt:
                line += f"\n  excerpt: {excerpt}"
            lines.append(line)
        evidence_block = "\n" + "\n".join(lines)

    return f"{header}\n{body}\n{evidence_block}".rstrip()


# ---------------------------------------------------------------------------
# judgment node
# ---------------------------------------------------------------------------


def judgment(state: RetrieverState, judge_counsel: JudgeCounsel) -> dict[str, Any]:
    """Judgment node that evaluates output via Judge Counsel voting.

    This node delegates evaluation to a multi-judge counsel that votes on
    whether to approve the output or request another iteration.

    Args:
        state: Current graph state with compiled output.
        judge_counsel: The Judge Counsel agent for evaluation.

    Returns:
        Updated state with judgment decision from counsel vote.
    """
    compiled_output = state.get("compiled_output", "")
    original_input = state["input"]
    iteration = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 3)

    # Invoke the Judge Counsel for multi-judge evaluation
    counsel_result = judge_counsel.evaluate(
        original_query=original_input,
        compiled_output=compiled_output,
        iteration_count=iteration,
        max_iterations=max_iterations,
    )

    decision = counsel_result.get("final_decision", JudgmentDecision.APPROVE)
    deliberation = counsel_result.get("deliberation_summary", "")

    return {
        "judgment_decision": decision,
        "judge_feedback": deliberation,
        "messages": [AIMessage(content=f"Judge Counsel Decision:\n{deliberation}")],
    }


# ---------------------------------------------------------------------------
# route_after_judgment
# ---------------------------------------------------------------------------


def route_after_judgment(state: RetrieverState) -> str:
    """Routing function to determine next step after judgment.

    Args:
        state: Current graph state.

    Returns:
        Next node name: 'choose_workers' for retry, END for approval.
    """
    from langgraph.graph import END

    decision = state.get("judgment_decision")

    if decision == JudgmentDecision.RETRY:
        return "choose_workers"

    return END


# ---------------------------------------------------------------------------
# decide_next_action node (closed-loop controller)
# ---------------------------------------------------------------------------


def decide_next_action(state: RetrieverState, llm: BaseChatModel) -> dict[str, Any]:
    """Decide whether to continue searching, finish, stop best-effort, or fail.

    This node is the closed-loop controller. It uses the compiled report,
    judge feedback, and iteration budget to decide what to do next. If it
    chooses to continue, it can also provide guidance for the next planning step.
    """
    iteration = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 3)

    compiled_report = state.get("compiled_report") or {}
    compiled_output = state.get("compiled_output", "")
    judge_feedback = state.get("judge_feedback") or ""
    judgment_decision = state.get("judgment_decision")

    # Progress signal: how many unique URLs we have so far vs this iteration.
    all_results = state.get("worker_results", [])
    urls_total = sorted(unique_urls_from_results(all_results))
    urls_this_iter = sorted(
        unique_urls_from_results([r for r in all_results if r.get("iteration") == iteration])
    )

    system_prompt = """You are a deep research controller.
Given the current report and judge feedback, decide the next action:
- continue: more research is needed and worth it; provide guidance for next iteration
- finish: output is good enough to deliver now
- stop_best_effort: diminishing returns; deliver best-effort with limitations
- fail: cannot proceed (e.g., missing access/deps) and should admit failure

You must consider:
- the judge feedback (if it says RETRY, identify what to fix)
- the iteration budget (if at/over max iterations, prefer stop_best_effort unless clearly finished)
- whether new progress is being made (e.g., new sources/URLs)
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Original query:\n{state['input']}\n\n"
                f"Iteration: {iteration}/{max_iterations}\n"
                f"Judgment decision: {judgment_decision}\n\n"
                f"Judge feedback:\n{judge_feedback}\n\n"
                f"URLs this iteration ({len(urls_this_iter)}):\n"
                + "\n".join(f"- {u}" for u in urls_this_iter[:20])
                + "\n\n"
                f"Total unique URLs so far ({len(urls_total)}):\n"
                + "\n".join(f"- {u}" for u in urls_total[:20])
                + "\n\n"
                f"Compiled report (structured):\n{json.dumps(compiled_report, indent=2, default=str)}\n\n"
                f"Compiled output (rendered):\n{compiled_output}"
            )
        ),
    ]

    decision: NextActionDecision
    decision_msg: AIMessage
    try:
        with_structured = getattr(llm, "with_structured_output", None)
        if callable(with_structured):
            try:
                runner = cast(Any, with_structured(NextActionDecision, method="function_calling"))
            except TypeError:
                runner = cast(Any, with_structured(NextActionDecision))
            decision = cast(NextActionDecision, cast(Any, runner).invoke(messages))
            decision_msg = AIMessage(content=decision.model_dump_json())
        else:
            raise RuntimeError("LLM does not support structured output")
    except Exception:
        # Deterministic fallback:
        if iteration >= max_iterations:
            action = NextAction.STOP_BEST_EFFORT
            rationale = "Iteration budget reached; stopping with best-effort output."
        elif judgment_decision == JudgmentDecision.RETRY:
            action = NextAction.CONTINUE
            rationale = "Judge requested retry; continuing to address gaps."
        else:
            action = NextAction.FINISH
            rationale = "No retry requested; finishing."
        decision = NextActionDecision(action=action, rationale=rationale, gaps=[], next_worker_guidance=None)
        decision_msg = AIMessage(content=decision.model_dump_json())

    final_status: str | None = None
    if decision.action == NextAction.FINISH:
        final_status = "success"
    elif decision.action == NextAction.STOP_BEST_EFFORT:
        final_status = "best_effort"
    elif decision.action == NextAction.FAIL:
        final_status = "failed"

    return {
        "next_action": decision.action.value,
        "next_worker_guidance": decision.next_worker_guidance,
        "final_status": final_status,
        "messages": [decision_msg],
    }


def route_after_decision(state: RetrieverState) -> str:
    """Route after decide_next_action."""
    from langgraph.graph import END

    action = state.get("next_action")
    if action == NextAction.CONTINUE.value:
        return "choose_workers"
    return END

