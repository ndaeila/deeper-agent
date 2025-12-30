"""Shared contracts for the Retriever agent."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class WorkerType(str, Enum):
    """Supported worker types."""

    BROWSER_USE = "browser_use"
    LLM = "llm"


class EvidenceItem(BaseModel):
    """A normalized evidence artifact."""

    url: HttpUrl
    title: str | None = None
    excerpt: str | None = None


class WorkerFinding(BaseModel):
    """Structured worker output."""

    findings: str
    evidence: list[EvidenceItem] = Field(default_factory=list)


