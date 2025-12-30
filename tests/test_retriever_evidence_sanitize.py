from odr.agents.retriever.contracts import normalize_worker_result, unique_urls_from_results


def test_normalize_worker_result_cleans_trailing_backslash_urls():
    result = {
        "worker_id": "worker_1",
        "findings": "",
        "success": True,
        "iteration": 1,
        "evidence": [{"url": "https://www.youtube.com/@natedaeilauw\\", "excerpt": "ok"}],
    }
    normalized = normalize_worker_result(result)  # type: ignore[arg-type]
    assert normalized["evidence"][0]["url"] == "https://www.youtube.com/@natedaeilauw"


def test_unique_urls_from_results_ignores_findings_text():
    results = [
        {
            "worker_id": "w1",
            "findings": "bad: https://www.linkedin.com/in\\n and ok https://www.daeila.com/",
            "success": True,
            "iteration": 1,
            "evidence": [{"url": "about:blank"}, {"url": "https://rocketreach.co/nathan-daeila-email_403393211\\"}],
        }
    ]
    urls = unique_urls_from_results(results)  # type: ignore[arg-type]
    # We no longer extract URLs from unstructured findings to avoid garbage artifacts.
    assert "https://www.daeila.com/" not in urls
    # The cleaned RocketReach URL (from structured evidence) should be present without the trailing backslash.
    assert "https://rocketreach.co/nathan-daeila-email_403393211" in urls
    # The truncated linkedin '/in' should not be included (from structured evidence cleaning rules).
    assert "https://www.linkedin.com/in" not in urls


