from __future__ import annotations

from odr.agents.retriever.planner import _build_choose_workers_human_prompt, _build_previous_context


def test_build_previous_context_includes_planning_guidance_and_feedback() -> None:
    state = {
        "input": "Original query: X",
        "iteration_count": 1,
        "compiled_output": "Prev output",
        "compiled_report": {"answer": "Prev report"},
        "judge_feedback": "RETRY: missing primary source",
        "next_worker_guidance": "Stay in section A. Get 2 more primary URLs.",
    }
    ctx = _build_previous_context(state)  # type: ignore[arg-type]
    assert "Previous attempt output" in ctx
    assert "Previous structured report" in ctx
    assert "Judge feedback from last iteration" in ctx
    assert "Planning guidance" in ctx
    assert "Stay in section A" in ctx


def test_build_choose_workers_human_prompt_is_stable_and_includes_context() -> None:
    prompt = _build_choose_workers_human_prompt(
        input_text="User query: Who is X?\nSection to research: Geographic footprint",
        previous_context="\n\nPlanning guidance:\nDo NOT change the section.",
        max_workers=2,
    )
    assert prompt.startswith("Research task:\n")
    assert "Section to research: Geographic footprint" in prompt
    assert "Planning guidance" in prompt
    assert "Do NOT change the section." in prompt
    assert "up to 2 parallel workers" in prompt



