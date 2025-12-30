"""Judgment node(s) for the Retriever agent.

Wraps JudgeCounsel evaluation and routing helpers.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from odr.agents.judge_counsel import JudgeCounsel, JudgmentDecision
from odr.agents.retriever.state import RetrieverState


def judgment(state: RetrieverState, judge_counsel: JudgeCounsel) -> dict[str, Any]:
    """Judgment node that evaluates output via Judge Counsel voting."""
    compiled_output = state.get("compiled_output", "")
    original_input = state["input"]
    iteration = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 3)

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


def route_after_judgment(state: RetrieverState) -> str:
    """Routing function to determine next step after judgment."""
    from langgraph.graph import END

    decision = state.get("judgment_decision")
    if decision == JudgmentDecision.RETRY:
        return "choose_workers"
    return END



