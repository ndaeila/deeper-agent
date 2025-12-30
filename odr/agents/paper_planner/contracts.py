"""Pydantic contracts for paper planning (high-level outline).

This planner sits above the Retriever and produces an outline of sections/goals/questions
to investigate for a given user query.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class EvidenceRequirements(BaseModel):
    """Requirements for evidence gathering."""

    min_unique_urls: Optional[int] = Field(None, description="Minimum number of unique URLs to find (e.g. 2).")
    urls_required: bool = Field(True, description="Whether URLs are required.")


class PaperSection(BaseModel):
    """A single planned section of the final paper."""

    section_id: str = Field(..., description="Stable id, e.g. section_1")
    title: str
    goal: str = Field(..., description="What this section should establish/answer")
    questions: list[str] = Field(default_factory=list, description="Concrete research questions")
    retrieval_query: str = Field(
        ...,
        description="A focused query to pass to the Retriever to gather evidence for this section",
    )
    requirements: EvidenceRequirements = Field(
        default_factory=EvidenceRequirements,
        description="Optional requirements for evidence depth.",
    )


class PaperOutline(BaseModel):
    """Planned outline for a markdown research paper."""

    title: str
    abstract_goal: str = Field(..., description="What the abstract should communicate")
    sections: list[PaperSection] = Field(min_length=2)



