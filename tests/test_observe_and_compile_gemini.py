"""Live integration test for Retriever observe_and_compile with Gemini.

This test is intentionally env-gated and will be skipped unless GEMINI credentials are present.
It is meant for local verification that the Gemini client works end-to-end, not for CI.
"""

from __future__ import annotations

import os

import pytest

from odr.agents.retriever.compiler import observe_and_compile
from odr.integrations.observability import get_observed_gemini_llm


@pytest.mark.integration
def test_observe_and_compile_with_gemini_live() -> None:
    """Call observe_and_compile using a real Gemini LLM (skips if not configured)."""
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")

    if not api_key or not model:
        pytest.skip("Requires GEMINI_API_KEY and GEMINI_MODEL in environment")

    # Optional extra guard so CI never accidentally runs a live test.
    if os.getenv("RUN_LIVE_LLM_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_LIVE_LLM_TESTS=true to run live Gemini integration tests")

    llm = get_observed_gemini_llm(model=model, api_key=api_key, temperature=0)

    # Minimal state required by observe_and_compile.
    state = {
        "input": "What is the capital of France? Answer in one sentence.",
        "iteration_count": 1,
        "worker_results": [
            {
                "worker_id": "worker_1",
                "worker_type": "llm",
                "success": True,
                "iteration": 1,
                "findings": (
                    "France's capital city is Paris. Source: https://en.wikipedia.org/wiki/Paris"
                ),
                "evidence": [
                    {
                        "url": "https://en.wikipedia.org/wiki/Paris",
                        "title": "Paris - Wikipedia",
                        "excerpt": "Paris is the capital and most populous city of France.",
                    }
                ],
            }
        ],
    }

    out = observe_and_compile(state, llm)

    assert isinstance(out, dict)
    assert "compiled_output" in out
    assert isinstance(out["compiled_output"], str)
    assert out["compiled_output"].strip()
    assert "compiled_report" in out
    assert isinstance(out["compiled_report"], dict)



