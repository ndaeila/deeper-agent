"""Contracts for the Compiler node."""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl

from odr.agents.retriever.shared.contracts import EvidenceItem


class Claim(BaseModel):
    """A single factual claim with explicit supporting evidence."""

    statement: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float | None = None


class Citation(BaseModel):
    """A citation entry in the compiled report."""

    url: HttpUrl
    title: str | None = None
    note: str | None = None


class CompiledReport(BaseModel):
    """Structured compiled output preserving sources."""

    answer: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)



