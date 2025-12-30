"""Tools for refining the paper outline based on research findings."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AddSection(BaseModel):
    """Add a new section to the research paper outline."""

    title: str = Field(..., description="Title of the new section")
    goal: str = Field(..., description="What this section should establish/answer")
    questions: list[str] = Field(..., description="Concrete research questions for this section")
    retrieval_query: str = Field(
        ...,
        description="Focused query to pass to the Retriever to gather evidence for this section",
    )
    after_section_id: Optional[str] = Field(
        None,
        description="ID of the section after which to insert the new section. If None, appends to the end.",
    )


class RemoveSection(BaseModel):
    """Remove a section from the research paper outline."""

    section_id: str = Field(..., description="ID of the section to remove")


class UpdateSection(BaseModel):
    """Update an existing section in the research paper outline."""

    section_id: str = Field(..., description="ID of the section to update")
    title: Optional[str] = Field(None, description="New title")
    goal: Optional[str] = Field(None, description="New goal")
    questions: Optional[list[str]] = Field(None, description="New questions")
    retrieval_query: Optional[str] = Field(None, description="New retrieval query")


class NoChange(BaseModel):
    """Indicate that no changes are needed to the outline."""

    reason: str = Field(..., description="Why no changes are needed")



