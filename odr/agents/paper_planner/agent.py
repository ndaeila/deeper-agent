"""PaperPlanner agent wrapper."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from odr.agents.paper_planner.contracts import PaperOutline
from odr.agents.paper_planner.planner import plan_outline
from odr.factory import DefaultLLMFactory


class PaperPlanner:
    """High-level planner that produces a research-paper outline for a query."""

    def __init__(self, llm_factory: DefaultLLMFactory | None = None, llm: BaseChatModel | None = None):
        if llm_factory:
            self.llm = llm_factory.get_llm(name="paper-planner", provider="gemini")
        else:
            self.llm = llm
        
        if not self.llm:
            raise ValueError("Either llm_factory or llm must be provided")

    def plan(self, input_text: str) -> PaperOutline:
        """Plan the outline for a query."""
        return plan_outline(input_text=input_text, llm=self.llm)



