"""Research paper writer agent (downstream from retrieval)."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models import BaseChatModel

from odr.factory import DefaultLLMFactory

from .graph import create_research_paper_graph
from .state import ResearchPaperState


class ResearchPaperWriter:
    """Research paper writer that drafts sections sequentially from outline + retrieval packets."""

    def __init__(
        self,
        llm_factory: DefaultLLMFactory | None = None,
        llm: BaseChatModel | None = None,
        recursion_limit: int = 50,
    ):
        """Initialize the paper writer.

        Args:
            llm_factory: Factory for creating LLMs.
            llm: Language model used to write the markdown research paper.
            recursion_limit: Maximum number of graph steps allowed.
        """
        if llm_factory:
            self.llm = llm_factory.get_llm(name="paper-writer", provider="gemini")
        else:
            self.llm = llm
        
        if not self.llm:
            raise ValueError("Either llm_factory or llm must be provided")

        self.recursion_limit = recursion_limit
        self.graph = create_research_paper_graph(llm=self.llm)

    def run(
        self,
        input_text: str,
        outline: dict[str, Any] | None,
        section_packets: list[dict[str, Any]],
    ) -> ResearchPaperState:
        """Write a paper from a planned outline and per-section retrieval packets.

        Returns:
            Final ResearchPaperState containing `paper_markdown`.
        """
        initial_state = ResearchPaperState(
            input=input_text,
            outline=outline,
            section_packets=section_packets,
            section_index=0,
            drafted_sections=[],
            paper_markdown="",
            final_status=None,
            messages=[],
        )
        config = {"recursion_limit": getattr(self, "recursion_limit", 50)}
        return cast(ResearchPaperState, self.graph.invoke(initial_state, config=config))


