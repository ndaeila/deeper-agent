"""Utilities for the Compiler node."""

from __future__ import annotations

from .contracts import CompiledReport


def render_compiled_report(report: CompiledReport) -> str:
    """Render a structured report into a stable, human-readable answer."""
    lines: list[str] = [report.answer.strip()]
    if report.claims:
        lines.append("\nClaims (with evidence):")
        for idx, claim in enumerate(report.claims, start=1):
            lines.append(f"- [{idx}] {claim.statement}".strip())
            for ev in claim.evidence:
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

