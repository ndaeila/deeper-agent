"""Pydantic contracts for the Retriever deep-research loop.

These models define the data contracts between graph nodes:
- Planner emits a ResearchPlan with typed tasks (including worker_type).
- Workers emit findings with normalized evidence items (including URLs).
- Compiler emits a structured report that preserves citations/sources.
- Decision node emits the next action (continue/finish/stop_best_effort/fail).

The module also includes small pure helpers used by nodes and tests to prevent
context drift between steps (e.g., planner asks for URLs but outputs omit them).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, Field, HttpUrl

from odr.agents.types import WorkerResultExtended, WorkerTask


class WorkerType(str, Enum):
    """Supported worker types."""

    BROWSER_USE = "browser_use"
    LLM = "llm"


class EvidenceRequirement(BaseModel):
    """Evidence requirements associated with a task or overall run."""

    urls_required: bool = True
    min_unique_urls: int = 1


class TaskSpec(BaseModel):
    """A structured task assigned by the planner."""

    # Planner may omit; Retriever will assign stable ids (worker_1..worker_N).
    worker_id: str | None = None
    worker_type: WorkerType = WorkerType.BROWSER_USE
    task_description: str
    deliverables: list[str] = Field(default_factory=list)
    requirements: EvidenceRequirement = Field(default_factory=EvidenceRequirement)


class ResearchPlan(BaseModel):
    """A structured plan produced by the planner."""

    tasks: list[TaskSpec] = Field(min_length=1)


class EvidenceItem(BaseModel):
    """A normalized evidence artifact."""

    url: HttpUrl
    title: str | None = None
    excerpt: str | None = None


class WorkerFinding(BaseModel):
    """Structured worker output."""

    findings: str
    evidence: list[EvidenceItem] = Field(default_factory=list)


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


_URL_RE = re.compile(r"https?://[^\s\]\)\}>,\"']+")


def extract_urls(text: str) -> set[str]:
    """Extract URLs from a string using a conservative regex."""
    urls: set[str] = set()
    for m in _URL_RE.finditer(text or ""):
        urls.add(m.group(0).rstrip(".,);]"))
    return urls


def coerce_evidence_items(raw: Any) -> list[dict[str, Any]]:
    """Coerce a raw evidence value into a list of dicts with at least url/excerpt/title keys."""
    if not raw:
        return []
    if isinstance(raw, list):
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, str):
                urls = list(extract_urls(item))
                if urls:
                    out.append({"url": urls[0], "excerpt": item[:500]})
        return out
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, str):
        urls = list(extract_urls(raw))
        return [{"url": urls[0], "excerpt": raw[:500]}] if urls else []
    return []


def normalize_worker_result(result: WorkerResultExtended) -> WorkerResultExtended:
    """Ensure a worker result has normalized evidence and best-effort URLs.

    - If evidence is missing, attempt to extract URLs from findings.
    - If evidence exists but items lack URLs, attempt to populate from excerpts/findings.
    """
    findings = str(result.get("findings", "") or "")
    evidence = coerce_evidence_items(result.get("evidence"))

    normalized: list[dict[str, Any]] = []
    findings_urls = extract_urls(findings)

    for item in evidence:
        url = item.get("url") or item.get("source_url")
        if not url and isinstance(item.get("excerpt"), str):
            urls = extract_urls(item["excerpt"])
            url = next(iter(urls), None)
        if not url:
            url = next(iter(findings_urls), None)
        if url:
            normalized.append(
                {
                    "url": str(url),
                    "title": item.get("title"),
                    "excerpt": item.get("excerpt"),
                }
            )

    # If still empty, attach URLs from findings as bare citations.
    if not normalized and findings_urls:
        normalized = [{"url": u, "excerpt": None, "title": None} for u in sorted(findings_urls)]

    result["evidence"] = normalized
    return result


def plan_to_worker_tasks(plan: ResearchPlan, raw_context: str) -> list[WorkerTask]:
    """Convert a structured plan into WorkerTask TypedDicts."""
    tasks: list[WorkerTask] = []
    for idx, t in enumerate(plan.tasks, start=1):
        deliverables_txt = ""
        if t.deliverables:
            deliverables_txt = "\nDeliverables:\n- " + "\n- ".join(t.deliverables)
        req_txt = (
            f"\nEvidence requirements:\n- urls_required={t.requirements.urls_required}\n"
            f"- min_unique_urls={t.requirements.min_unique_urls}"
        )
        tasks.append(
            WorkerTask(
                worker_id=t.worker_id or f"worker_{idx}",
                task_description=t.task_description,
                context=(raw_context + deliverables_txt + req_txt).strip(),
                worker_type=t.worker_type.value,
            )
        )
    return tasks


def unique_urls_from_results(results: Iterable[WorkerResultExtended]) -> set[str]:
    """Collect unique URLs from worker results (findings + evidence)."""
    urls: set[str] = set()
    for r in results:
        urls |= extract_urls(str(r.get("findings", "")))
        for ev in coerce_evidence_items(r.get("evidence")):
            url = ev.get("url") or ev.get("source_url")
            if isinstance(url, str) and url.startswith("http"):
                urls.add(url.rstrip(".,);]"))
    return urls


def render_compiled_report(report: CompiledReport) -> str:
    """Render a structured report into a stable, human-readable answer."""
    lines: list[str] = [report.answer.strip()]
    if report.claims:
        lines.append("\nClaims (with evidence):")
        for idx, claim in enumerate(report.claims, start=1):
            lines.append(f"- [{idx}] {claim.statement}".strip())
            for ev in claim.evidence[:5]:
                title = f"{ev.title} " if ev.title else ""
                excerpt = f' "{ev.excerpt}"' if ev.excerpt else ""
                lines.append(f"  - {title}{ev.url}{excerpt}".strip())
    if report.citations:
        lines.append("\nSources:")
        for idx, c in enumerate(report.citations, start=1):
            title = f"{c.title} " if c.title else ""
            note = f" — {c.note}" if c.note else ""
            lines.append(f"- [{idx}] {title}{c.url}{note}".strip())
    if report.open_questions:
        lines.append("\nOpen questions:")
        for q in report.open_questions:
            lines.append(f"- {q}")
    if report.limitations:
        lines.append("\nLimitations:")
        for lim in report.limitations:
            lines.append(f"- {lim}")
    return "\n".join(lines).strip()


