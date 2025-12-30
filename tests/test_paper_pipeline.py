"""Tests for Planner -> Retriever loop -> ResearchPaperWriter pipeline."""

import json
from unittest.mock import MagicMock

import pytest

from odr.agents.paper_pipeline import DeepResearchPaper
from odr.agents.paper_planner.contracts import PaperOutline, PaperSection
from odr.agents.retriever.state import RetrieverState


@pytest.fixture
def mock_llm():
    """Mock LLM used by planner and writer.

    We use the fallback `.invoke()` paths by disabling structured output.
    """
    llm = MagicMock()
    llm.with_structured_output = None
    # Planner fallback expects JSON for PaperOutline. Writer invokes LLM too; return simple strings.
    llm.invoke.side_effect = [
        MagicMock(
            content=json.dumps(
                {
                    "title": "Test Paper",
                    "abstract_goal": "Summarize the subject with evidence.",
                    "sections": [
                        {
                            "section_id": "section_1",
                            "title": "Background",
                            "goal": "Establish basic identity and context.",
                            "questions": ["Who is the person?", "What are notable works?"],
                            "retrieval_query": "Identify the person and notable public artifacts.",
                            "requirements": {},
                        },
                        {
                            "section_id": "section_2",
                            "title": "Public Writing",
                            "goal": "Summarize authored posts with dates and key themes.",
                            "questions": ["What posts exist?", "What themes appear?"],
                            "retrieval_query": "Find authored posts and dates; extract excerpts.",
                            "requirements": {},
                        },
                    ],
                }
            )
        ),
        # write_next_section call 1
        MagicMock(content="## Background\n\nDrafted background.\n"),
        # write_next_section call 2
        MagicMock(content="## Public Writing\n\nDrafted writing.\n"),
        # assemble_paper
        MagicMock(content="# Test Paper\n\nAbstract...\n\n## Background\n\n...\n\n## Public Writing\n\n...\n\n"),
    ]
    return llm


class _FakeRetriever:
    def __init__(self):
        self.calls: list[str] = []

    def run(self, query: str) -> RetrieverState:  # noqa: D401
        """Return a deterministic retriever state; record calls."""
        self.calls.append(query)
        return RetrieverState(
            input=query,
            messages=[],
            worker_tasks=[],
            worker_results=[],
            compiled_output="compiled",
            judgment_decision=None,
            judge_feedback=None,
            compiled_report={"answer": "compiled", "citations": []},
            next_action="finish",
            next_worker_guidance=None,
            final_status="success",
            iteration_count=1,
            max_iterations=3,
            max_workers=2,
        )


def test_pipeline_runs_retriever_once_per_section_and_writes_paper(mock_llm):
    retriever = _FakeRetriever()
    pipeline = DeepResearchPaper(llm=mock_llm, retriever=retriever)

    out = pipeline.run("Who is Nathan Daeila?")

    # Two sections -> two retriever calls
    assert len(retriever.calls) == 2
    assert "Section to research:\nBackground" in retriever.calls[0]
    assert "Section to research:\nPublic Writing" in retriever.calls[1]

    outline = out["outline"]
    assert outline["title"] == "Test Paper"
    assert len(outline["sections"]) == 2
    assert "references" in outline

    assert out["paper_status"] == "success"
    assert out["paper_markdown"].startswith("# Test Paper")


