"""Closed-loop controller node(s) for the Retriever agent.

Decides whether to continue researching or finish.
"""

from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from odr.agents.judge_counsel import JudgmentDecision
from odr.agents.retriever.state import RetrieverState
from odr.agents.retriever.shared.utils import unique_urls_from_results
from odr.factory import DefaultLLMFactory

from .contracts import NextAction, NextActionDecision


_DECIDE_NEXT_ACTION_SYSTEM_PROMPT = """You are a deep research controller.
Given the current report and judge feedback, decide the next action:
- continue: more research is needed and worth it; provide guidance for next iteration
- finish: output is good enough to deliver now
- stop_best_effort: diminishing returns; deliver best-effort with limitations
- fail: cannot proceed (e.g., missing access/deps) and should admit failure

You must consider:
- the judge feedback (if it says RETRY, identify what to fix)
- if the judgment decision is APPROVE, you should almost always choose finish (do not continue merely to
  answer optional open questions; only continue if the user explicitly asked for something that has not yet been answered,
  more research is needed or there is a clear correctness issue that must be fixed to safely deliver)
- the iteration budget (if at/over max iterations, prefer stop_best_effort unless clearly finished)
- whether new progress is being made (e.g., new sources/URLs)

CRITICAL ANTI-DRIFT:
- You MUST stay aligned to the user's original query and any embedded structure (e.g., paper title/section/goal/questions).
- Do NOT switch to a different section/topic or write a generic OSINT methodology primer unless the original query
  explicitly asks for that as the next step.

If action=continue, you MUST set next_worker_guidance and it MUST include:
- Recap (3-6 bullets): what was covered and which sources were used (URLs ok).
- Gaps (1-5 bullets): what is missing, with crisp acceptance criteria (what would "done" look like).
- Next iteration plan: 1-2 atomic worker tasks, each with concrete steps + evidence requirements.

If action != continue, set next_worker_guidance to null.
"""


def _compact_recap_from_compiled_report(compiled_report: dict[str, Any]) -> str:
    """Build a short, deterministic recap to reduce LLM drift between iterations."""
    try:
        claims = compiled_report.get("claims") if isinstance(compiled_report, dict) else None
    except Exception:
        claims = None
    if not isinstance(claims, list) or not claims:
        return "(no structured claims available)"

    lines: list[str] = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        stmt = c.get("statement")
        if not isinstance(stmt, str) or not stmt.strip():
            continue
        url = None
        ev = c.get("evidence")
        if isinstance(ev, list) and ev:
            first = ev[0]
            if isinstance(first, dict):
                url = first.get("url")
        url_txt = f" ({url})" if isinstance(url, str) and url.startswith("http") else ""
        lines.append(f"- {stmt.strip()}{url_txt}")

    return "\n".join(lines) if lines else "(no structured claims available)"


def decide_next_action(state: RetrieverState, llm: BaseChatModel | None = None, llm_factory: DefaultLLMFactory | None = None) -> dict[str, Any]:
    """Decide whether to continue searching, finish, stop best-effort, or fail."""
    if llm_factory:
        llm = llm_factory.get_llm(name="retriever-controller")
    
    if not llm:
        raise ValueError("Either llm or llm_factory must be provided")

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
    recap = _compact_recap_from_compiled_report(cast(dict[str, Any], compiled_report))

    messages = [
        SystemMessage(content=_DECIDE_NEXT_ACTION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original query:\n{state['input']}\n\n"
                f"Iteration: {iteration}/{max_iterations}\n"
                f"Judgment decision: {judgment_decision}\n\n"
                f"Judge feedback:\n{judge_feedback}\n\n"
                f"Deterministic recap (from structured claims):\n{recap}\n\n"
                f"URLs this iteration ({len(urls_this_iter)}):\n"
                + "\n".join(f"- {u}" for u in urls_this_iter)
                + "\n\n"
                f"Total unique URLs so far ({len(urls_total)}):\n"
                + "\n".join(f"- {u}" for u in urls_total)
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
            decision = cast(NextActionDecision, cast(Any, runner).invoke(messages, config={"run_name": "decide_next_action"}))
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
    # Safety override: if the judge already approved, default to finishing.
    #
    # In practice, "open questions" are often optional nice-to-haves; continuing after APPROVE
    # can waste budget and produce controller loops. We still allow FAIL if something truly prevents
    # delivering output (e.g., missing deps/access).
    if judgment_decision == JudgmentDecision.APPROVE and decision.action in {
        NextAction.CONTINUE,
        NextAction.STOP_BEST_EFFORT,
    }:
        decision = NextActionDecision(
            action=NextAction.FINISH,
            rationale=f"{decision.rationale} (Judge approved; finishing.)".strip(),
            gaps=list(decision.gaps or []),
            next_worker_guidance=None,
        )

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

