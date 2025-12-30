"""Contracts for the Controller node."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class NextAction(str, Enum):
    """Next action after judgment/decision."""

    CONTINUE = "continue"
    FINISH = "finish"
    STOP_BEST_EFFORT = "stop_best_effort"
    FAIL = "fail"


class NextActionDecision(BaseModel):
    """Structured decision after judgment."""

    action: NextAction
    rationale: str
    gaps: list[str] = Field(default_factory=list)
    next_worker_guidance: str | None = None



