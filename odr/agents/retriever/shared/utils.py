"""Shared utilities for the Retriever agent."""

from __future__ import annotations

import re
from typing import Any, Iterable
from urllib.parse import urlparse

from odr.agents.types import WorkerResultExtended
from .contracts import EvidenceItem


_URL_RE = re.compile(r"https?://[^\s\]\)\}>,\"']+")


def _clean_url(raw: Any) -> str | None:
    """Best-effort URL cleanup for common worker/tool artifacts.

    Workers sometimes return URLs with:
    - trailing backslashes (e.g., 'https://x.com/\\')
    - embedded escape sequences (e.g., '...\\n')
    - surrounding quotes
    """
    if not raw:
        return None
    if not isinstance(raw, str):
        return None

    url = raw.strip().strip('"').strip("'")
    # Remove common trailing escape artifacts.
    url = url.replace("\\n", "").replace("\\t", "").replace("\\r", "")
    url = url.rstrip("\\").rstrip(".,);]").strip()

    if not url.startswith(("http://", "https://")):
        return None

    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    # Drop common "stub" profile URLs that are technically valid but not useful evidence.
    host = parsed.netloc.lower()
    path = (parsed.path or "").rstrip("/")
    if host.endswith("linkedin.com") and path == "/in":
        return None
    return url


def _is_low_signal_evidence(url: str, excerpt: str | None) -> bool:
    """Filter evidence artifacts that are likely tool logs or non-content."""
    if "cf.browser-use.com/logo.svg" in url:
        return True
    if excerpt is None:
        return False
    ex = excerpt.strip()
    if not ex:
        return False
    # Tool/log style artifacts (not page content).
    if ex.startswith("🔗") or ex.startswith("SEARCH_QUOTA_EXHAUSTED"):
        return True
    if ex.startswith("[") and "results" in ex and "{'query':" in ex:
        # Serialized search results blob.
        return True
    if "<url>" in ex and "about:blank" in ex and "logo" in ex:
        return True
    return False


def extract_urls(text: str) -> set[str]:
    """Extract URLs from a string using a conservative regex."""
    urls: set[str] = set()
    for m in _URL_RE.finditer(text or ""):
        cleaned = _clean_url(m.group(0))
        if cleaned:
            urls.add(cleaned)
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
                    out.append({"url": urls[0], "excerpt": item})
        return out
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, str):
        urls = list(extract_urls(raw))
        return [{"url": urls[0], "excerpt": raw}] if urls else []
    return []


def normalize_worker_result(result: WorkerResultExtended) -> WorkerResultExtended:
    """Ensure a worker result has normalized evidence and best-effort URLs.

    - If evidence is missing, attempt to extract URLs from findings.
    - If evidence exists but items lack URLs, attempt to populate from excerpts/findings.
    """
    findings = str(result.get("findings", "") or "")
    evidence = coerce_evidence_items(result.get("evidence"))

    normalized: list[dict[str, Any]] = []
    # findings_urls = extract_urls(findings)  <-- Disable extraction from findings (too noisy)

    for item in evidence:
        raw_url = item.get("url") or item.get("source_url")
        excerpt = item.get("excerpt") if isinstance(item.get("excerpt"), str) else None

        url = _clean_url(raw_url)
        if not url and excerpt:
            # excessive URL extraction from excerpts is also noisy; trust explicit URLs first.
            # url = next(iter(extract_urls(excerpt)), None)
            pass
        
        # If still no URL, skip backfilling from findings to avoid "hallucinated" URLs
        if not url:
            continue

        if _is_low_signal_evidence(url=url, excerpt=excerpt):
            continue

        normalized.append({"url": url, "title": item.get("title"), "excerpt": excerpt})

    result["evidence"] = normalized
    return result


def unique_urls_from_results(results: Iterable[WorkerResultExtended]) -> set[str]:
    """Collect unique URLs from worker results (findings + evidence)."""
    urls: set[str] = set()
    for r in results:
        # Don't extract from findings text; rely on structured evidence to avoid garbage artifacts
        # urls |= extract_urls(str(r.get("findings", "")))
        for ev in coerce_evidence_items(r.get("evidence")):
            url = _clean_url(ev.get("url") or ev.get("source_url"))
            if url:
                urls.add(url)
    return urls

