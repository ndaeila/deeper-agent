"""State definitions for the research paper writing workflow.

This agent is intentionally *downstream* from retrieval. It does NOT run the Retriever.
Instead, it consumes:
- an outline (sections + research goals/questions), and
- per-section retrieval artifacts (compiled reports, evidence excerpts, URLs),
then writes a markdown research paper section-by-section.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage


class ResearchPaperState(TypedDict):
    """State for the research paper writer graph.

    Attributes:
        input: The original query/task.
        outline: The planned paper outline (structured dict or model_dump).
        section_packets: Per-section retrieval artifacts (structured dicts).
        section_index: Current section index being drafted.
        drafted_sections: Drafted markdown sections (appended in order).
        paper_markdown: The final markdown research paper.
        final_status: Status for this step ("success" or "skipped").
        messages: Message history for observability/debugging.
    """

    input: str
    outline: dict[str, Any] | None
    section_packets: list[dict[str, Any]]
    section_index: int
    drafted_sections: Annotated[list[str], operator.add]
    paper_markdown: str
    final_status: str | None
    messages: Annotated[list[BaseMessage], operator.add]


