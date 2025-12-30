"""Node implementations for the research paper writing workflow."""

from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from odr.agents.retriever.contracts import coerce_evidence_items, unique_urls_from_results
from odr.agents.types import WorkerResultExtended

from .state import ResearchPaperState


def _extract_text_from_content(content: Any) -> str:
    """Extract text from LangChain message content.
    
    Handles both string content and Gemini's structured content blocks
    (list of dicts with 'type' and 'text' keys).
    
    Args:
        content: The content from a LangChain message response.
        
    Returns:
        Extracted text as a string.
    """
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        # Handle Gemini's structured content blocks: [{'type': 'text', 'text': '...'}]
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                # Extract text from content blocks
                if block.get("type") == "text" and "text" in block:
                    text_parts.append(str(block["text"]))
                # Also handle direct 'text' key without 'type'
                elif "text" in block:
                    text_parts.append(str(block["text"]))
            elif isinstance(block, str):
                text_parts.append(block)
        if text_parts:
            return "\n".join(text_parts)
    
    # Fallback: convert to string
    return str(content)


def route_after_section(state: ResearchPaperState) -> str:
    """Route through the section-writing loop, then assemble."""
    outline = state.get("outline") or {}
    sections = cast(list[dict[str, Any]], outline.get("sections") or [])
    idx = int(state.get("section_index") or 0)
    if idx < len(sections):
        return "write_next_section"
    return "assemble_paper"


def write_next_section(state: ResearchPaperState, llm: BaseChatModel) -> dict[str, Any]:
    """Write the next section of the paper based on its section packet (retrieval artifacts)."""
    outline = state.get("outline") or {}
    sections = cast(list[dict[str, Any]], outline.get("sections") or [])
    packets = state.get("section_packets") or []
    idx = int(state.get("section_index") or 0)

    if not sections or idx >= len(sections):
        return {"section_index": idx}

    section = cast(dict[str, Any], sections[idx])
    packet = cast(dict[str, Any], packets[idx] if idx < len(packets) else {})

    # Prepare cleaner context
    # 1. Simplified Outline (for context/flow)
    simplified_outline = {
        "paper_title": outline.get("title"),
        "paper_goal": outline.get("abstract_goal"),
        "section_sequence": [s.get("title") for s in sections],
    }

    # 2. Current Section Spec
    section_spec = {
        "title": section.get("title"),
        "goal": section.get("goal"),
        "research_questions": section.get("questions"),
    }

    # 3. Research Findings
    retrieval_data = {
        "summary": packet.get("retriever_compiled_output"),
        "evidence": packet.get("evidence_items"),
    }

    # 4. Relevant Citations Only
    all_refs = cast(list[dict[str, Any]], (outline.get("references") or []))
    packet_urls = set(packet.get("unique_urls") or [])
    relevant_refs = [r for r in all_refs if r.get("url") in packet_urls]

    system_prompt = """You are writing ONE section of an OSINT-style research paper in Markdown.

You will be given:
- The original query
- The paper context (title, goal, section list)
- The current section specification (title + goal + questions)
- Research findings (summary + evidence)
- A list of available citations (numbered)

Write ONLY the section content for this section:
- Start with an H2 header matching the section title.
- Then write 2–6 paragraphs grounded ONLY in the provided research findings.
- If you cite evidence, use the numbered reference style [n] matching the available citations list.
- Do not invent facts, dates, quotes, or sources.
- If evidence is weak or first-party-only, explicitly say so."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=json.dumps(
                {
                    "original_query": state.get("input"),
                    "paper_context": simplified_outline,
                    "section_to_write": section_spec,
                    "research_findings": retrieval_data,
                    "available_citations": relevant_refs,
                },
                indent=2,
                default=str,
            )
        ),
    ]
    response = llm.invoke(messages, config={"run_name": "write_next_section"})
    section_md = _extract_text_from_content(response.content)
    return {
        "drafted_sections": [section_md],
        "section_index": idx + 1,
        "messages": [cast(AIMessage, response) if isinstance(response, AIMessage) else AIMessage(content=section_md)],
    }


def assemble_paper(state: ResearchPaperState) -> dict[str, Any]:
    """Assemble a full paper from drafted sections and the outline programmatically.

    This is a deterministic assembly step - no LLM needed since sections are already drafted.
    """
    outline = state.get("outline") or {}
    drafted = state.get("drafted_sections") or []
    references = cast(list[dict[str, Any]], (outline.get("references") or []))

    if not drafted:
        paper = _render_skipped_paper(reason="No drafted sections; cannot assemble paper.")
        return {"paper_markdown": paper, "final_status": "skipped", "messages": [AIMessage(content=paper)]}

    parts: list[str] = []

    # Title
    title = outline.get("title") or "Research Paper"
    parts.append(f"# {title}")
    parts.append("")

    # Abstract
    abstract_goal = outline.get("abstract_goal")
    if abstract_goal:
        parts.append("## Abstract")
        parts.append("")
        parts.append(abstract_goal)
        parts.append("")

    # Drafted sections (already have H2 headers)
    for section_md in drafted:
        parts.append(section_md.strip())
        parts.append("")

    # References
    if references:
        parts.append("## References")
        parts.append("")
        for ref in references:
            idx = ref.get("index", "?")
            url = ref.get("url", "")
            title_text = ref.get("title") or url
            parts.append(f"[{idx}] {title_text} - {url}")
        parts.append("")

    paper = "\n".join(parts)
    return {
        "paper_markdown": paper,
        "final_status": "success",
        "messages": [AIMessage(content=paper)],
    } 


def _collect_evidence_items(
    results: list[WorkerResultExtended],
    limit: int = 100000,
) -> list[dict[str, Any]]:
    """Collect normalized evidence items from all worker results.

    Note: We intentionally use the retriever's own evidence coercion helper to handle
    common worker artifacts (strings/dicts/lists) while keeping this agent separate
    from retriever's control loop.
    """
    out: list[dict[str, Any]] = []
    for r in results:
        for ev in coerce_evidence_items(r.get("evidence")):
            url = ev.get("url") or ev.get("source_url")
            if not url:
                continue
            excerpt = ev.get("excerpt")
            if isinstance(excerpt, str):
                excerpt = excerpt.strip()
            out.append(
                {
                    "worker_id": r.get("worker_id"),
                    "iteration": r.get("iteration"),
                    "url": url,
                    "title": ev.get("title"),
                    "excerpt": excerpt,
                }
            )
            if len(out) >= limit:
                return out
    return out


def _render_skipped_paper(reason: str) -> str:
    return (
        "# Research Paper (Skipped)\n\n"
        f"**Reason**: {reason}\n\n"
        "This step requires an outline and per-section retrieval packets.\n"
    )


