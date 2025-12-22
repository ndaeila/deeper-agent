"""Contract tests for Retriever v2 helpers."""

from odr.agents.retriever.contracts import (
    Citation,
    Claim,
    CompiledReport,
    EvidenceItem,
    extract_urls,
    normalize_worker_result,
    render_compiled_report,
)


def test_extract_urls_finds_http_urls():
    text = "See https://example.com/a and http://example.org/x."
    urls = extract_urls(text)
    assert "https://example.com/a" in urls
    assert "http://example.org/x" in urls


def test_normalize_worker_result_recovers_urls_from_findings():
    result = {
        "worker_id": "worker_1",
        "findings": "Evidence at https://example.com/source",
        "success": True,
        "iteration": 1,
    }
    normalized = normalize_worker_result(result)  # type: ignore[arg-type]
    assert normalized["evidence"]
    assert normalized["evidence"][0]["url"].startswith("https://example.com")


def test_render_compiled_report_includes_sources_section():
    report = CompiledReport(answer="Hello", citations=[Citation(url="https://example.com")])
    report_text = render_compiled_report(report)
    assert "Sources:" in report_text
    assert "https://example.com" in report_text


def test_render_compiled_report_includes_claims_section():
    report = CompiledReport(
        answer="Hello",
        claims=[
            Claim(
                statement="X is true",
                evidence=[EvidenceItem(url="https://example.com", excerpt="supporting snippet")],
            )
        ],
    )
    report_text = render_compiled_report(report)
    assert "Claims (with evidence):" in report_text
    assert "X is true" in report_text
    assert "https://example.com" in report_text


