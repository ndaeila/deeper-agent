from __future__ import annotations

from odr.agents.workers import browser_use_worker


def test_normalize_search_queries_dedupes_and_strips() -> None:
    queries = ["  Nathan Daeila  ", "nathan   daeila", "", "   ", "Nathan Daeila UW  "]
    normalized = browser_use_worker._normalize_search_queries(queries)  # type: ignore[attr-defined]
    assert normalized == ["Nathan Daeila", "Nathan Daeila UW"]


def test_search_budget_allows_one_call_by_default() -> None:
    budget = browser_use_worker._SearchBudget(max_calls=1, max_queries_per_call=3)  # type: ignore[attr-defined]

    accepted1, status1 = budget.consume(["foo", "bar"])
    assert status1 == browser_use_worker._SearchStatus.OK  # type: ignore[attr-defined]
    assert accepted1 == ["foo", "bar"]

    accepted2, status2 = budget.consume(["baz"])
    assert status2 == browser_use_worker._SearchStatus.BLOCKED  # type: ignore[attr-defined]
    assert accepted2 == []


def test_search_budget_dedupes_queries_across_calls() -> None:
    budget = browser_use_worker._SearchBudget(max_calls=2, max_queries_per_call=3)  # type: ignore[attr-defined]

    accepted1, status1 = budget.consume(["Nathan Daeila", "Nathan Daeila UW"])
    assert status1 == browser_use_worker._SearchStatus.OK  # type: ignore[attr-defined]
    assert accepted1 == ["Nathan Daeila", "Nathan Daeila UW"]

    # Repeat should be ignored; only the new query is accepted.
    accepted2, status2 = budget.consume(["nathan   daeila", "New Query"])
    assert status2 == browser_use_worker._SearchStatus.OK  # type: ignore[attr-defined]
    assert accepted2 == ["New Query"]


def test_plan_to_worker_tasks_propagates_runtime_requirements() -> None:
    from odr.agents.retriever.contracts import ResearchPlan, TaskSpec, WorkerType
    from odr.agents.retriever.contracts import EvidenceRequirement
    from odr.agents.retriever.contracts import plan_to_worker_tasks

    plan = ResearchPlan(
        tasks=[
            TaskSpec(
                worker_type=WorkerType.BROWSER_USE,
                task_description="Test task",
                requirements=EvidenceRequirement(urls_required=True, min_unique_urls=4),
                runtime={"exploration_mode": "depth", "max_search_calls": 0},
            )
        ]
    )
    tasks = plan_to_worker_tasks(plan=plan, raw_context="ctx")
    assert len(tasks) == 1
    req = tasks[0].get("requirements") or {}
    assert req["min_unique_urls"] == 4
    assert req["exploration_mode"] == "depth"
    assert req["max_search_calls"] == 0


