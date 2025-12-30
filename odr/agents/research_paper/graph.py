"""LangGraph wiring for the research paper writing workflow."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import assemble_paper, route_after_section, write_next_section
from .state import ResearchPaperState


def create_research_paper_graph(
    llm: BaseChatModel,
) -> CompiledStateGraph:
    """Create a graph of agents that work together to generate a final markdown research paper.

    Flow:
    - write_next_section: write sections sequentially from outline + packets
    - assemble_paper: assemble final markdown with references
    - END
    """
    graph = StateGraph(ResearchPaperState)

    def _write_next(state: Any) -> dict[str, Any]:
        return write_next_section(cast(ResearchPaperState, state), llm)

    def _assemble(state: Any) -> dict[str, Any]:
        return assemble_paper(cast(ResearchPaperState, state))

    graph.add_node("write_next_section", _write_next)
    graph.add_node("assemble_paper", _assemble)

    graph.set_entry_point("write_next_section")
    graph.add_conditional_edges(
        "write_next_section",
        route_after_section,
        {
            "write_next_section": "write_next_section",
            "assemble_paper": "assemble_paper",
            END: END,
        },
    )
    graph.add_edge("assemble_paper", END)

    return graph.compile()


