"""Pydantic contracts for the Retriever deep-research loop.

Re-exports contracts from the specific node submodules.
"""

from __future__ import annotations

from odr.agents.retriever.shared.contracts import WorkerType, EvidenceItem, WorkerFinding
from odr.agents.retriever.planner.contracts import ResearchPlan, TaskSpec, EvidenceRequirement
from odr.agents.retriever.compiler.contracts import CompiledReport, Claim, Citation
from odr.agents.retriever.controller.contracts import NextAction, NextActionDecision
from odr.agents.retriever.shared.utils import (
    extract_urls,
    normalize_worker_result,
    unique_urls_from_results,
    coerce_evidence_items,
)
from odr.agents.retriever.compiler.utils import render_compiled_report
from odr.agents.retriever.planner.utils import plan_to_worker_tasks

__all__ = [
    "WorkerType",
    "EvidenceItem",
    "WorkerFinding",
    "ResearchPlan",
    "TaskSpec",
    "EvidenceRequirement",
    "CompiledReport",
    "Claim",
    "Citation",
    "NextAction",
    "NextActionDecision",
    "extract_urls",
    "normalize_worker_result",
    "unique_urls_from_results",
    "coerce_evidence_items",
    "render_compiled_report",
    "plan_to_worker_tasks",
]
