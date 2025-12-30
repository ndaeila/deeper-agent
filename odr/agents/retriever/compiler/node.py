"""Compilation/synthesis node for the Retriever agent.

Responsible for aggregating worker results into a structured report and rendered output.
"""

from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from odr.agents.types import WorkerResultExtended
from odr.agents.retriever.state import RetrieverState
from odr.agents.retriever.shared.utils import unique_urls_from_results
from odr.factory import DefaultLLMFactory

from .contracts import Citation, CompiledReport
from .utils import render_compiled_report


def observe_and_compile(state: RetrieverState, llm: BaseChatModel | None = None, llm_factory: DefaultLLMFactory | None = None, sys_prompt: str | None = None) -> dict[str, Any]:
    """Observer node that aggregates findings from all workers."""
    if llm_factory:
        llm = llm_factory.get_llm(name="retriever-compiler", provider="gemini")
    
    if not llm:
        raise ValueError("Either llm or llm_factory must be provided")

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
    urls_block = "\n".join(f"- {u}" for u in urls) if urls else "(none)"
    evidence_items: list[dict[str, Any]] = []
    for r in worker_results:
        for ev in cast(list[dict[str, Any]], r.get("evidence") or []):
            url = ev.get("url")
            if not url:
                continue
            # Avoid flooding the synthesis prompt with tool logs or gigantic blobs.
            excerpt = ev.get("excerpt")
            if isinstance(excerpt, str):
                excerpt = excerpt.strip()
            evidence_items.append(
                {
                    "worker_id": r.get("worker_id"),
                    "url": url,
                    "title": ev.get("title"),
                    "excerpt": excerpt,
                }
            )

    system_prompt = sys_prompt or """You are an analyst synthesizing research findings from multiple workers.
Produce a structured report with:
- answer: very completely and accurately answer the original query, grounded in the findings
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
                # Gemini (langchain-google-genai) supports native structured output via json_schema.
                # Using function_calling can emit warnings like:
                #   Key '$defs' is not supported in schema, ignoring
                # because Gemini's tool schema is a restricted JSON Schema subset.
                if getattr(llm.__class__, "__module__", "").startswith("langchain_google_genai"):
                    runner = cast(
                        Any,
                        with_structured(
                            CompiledReport.model_json_schema(),
                            method="json_schema",
                        ),
                    )
                else:
                    runner = cast(Any, with_structured(CompiledReport, method="function_calling"))
            except TypeError:
                runner = cast(Any, with_structured(CompiledReport))
            
            # Using run_name here creates the trace entry for this operation.
            # If Langfuse callbacks are already attached to 'llm' (which they are),
            # invoking 'runner' will trigger them. The model trace is a child of this invocation.
            report = cast(CompiledReport, cast(Any, runner).invoke(messages, config={"run_name": "observe_and_compile"}))
            if not report.citations:
                # Prefer citations derived from claim evidence; fall back to known URLs.
                claim_urls: list[str] = []
                for c in report.claims:
                    for ev in c.evidence:
                        claim_urls.append(str(ev.url))
                dedup: list[str] = []
                seen: set[str] = set()
                for u in claim_urls:
                    if u not in seen:
                        seen.add(u)
                        dedup.append(u)
                use_urls = dedup or urls
                if use_urls:
                    # IMPORTANT: Pydantic v2 does not validate assignments by default.
                    # Ensure we set the correct model type, not raw dicts, since
                    # render_compiled_report expects `Citation` objects.
                    report.citations = [Citation(url=u) for u in use_urls]
            response_msg = AIMessage(content=report.model_dump_json())
        else:
            raise RuntimeError("LLM does not support structured output")
    except Exception as e:
        # Prevent double-tracing if the structured attempt failed but left a trace.
        # However, we can't easily suppress the previous trace.
        # We can at least ensure this fallback run is clearly marked or reuses the intent.
        
        # Check if the error is likely a refusal/API error or just a parsing error.
        # If it's an API error (e.g. 404, overload), the fallback might also fail, but we try anyway.
        
        response = llm.invoke(messages, config={"run_name": "observe_and_compile_fallback"})
        content = response.content if isinstance(response.content, str) else str(response.content)
        # Best-effort: keep URLs deterministically as citations if the model didn't structure them.
        citations = [{"url": u} for u in urls]
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
        for idx, ev in enumerate(evidence, start=1):
            url = ev.get("url") or ev.get("source_url") or ""
            title = ev.get("title") or ""
            excerpt = ev.get("excerpt") or ""
            line = f"- [{idx}] {title} {url}".strip()
            if excerpt:
                line += f"\n  excerpt: {excerpt}"
            lines.append(line)
        evidence_block = "\n" + "\n".join(lines)

    return f"{header}\n{body}\n{evidence_block}".rstrip()

