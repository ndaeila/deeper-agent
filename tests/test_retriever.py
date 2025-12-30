"""Tests for the Retriever agent."""

import random
from unittest.mock import MagicMock, patch

import pytest

from odr.agents.retriever import (
    JudgmentDecision,
    Retriever,
    RetrieverState,
    WorkerResult,
    WorkerTask,
    WorkerState,
    choose_workers,
    decide_next_action,
    fan_out_to_workers,
    judgment,
    observe_and_compile,
    route_after_decision,
    route_after_judgment,
    worker_node,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns predictable responses."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Mock LLM response")
    # Force fallback paths that use .invoke() (MagicMock otherwise pretends to support everything).
    llm.with_structured_output = None
    return llm


@pytest.fixture
def sample_state() -> RetrieverState:
    """Create a sample initial state."""
    return RetrieverState(
        input="What is OSINT?",
        messages=[],
        worker_tasks=[],
        worker_results=[],
        compiled_output="",
        judgment_decision=None,
        judge_feedback=None,
        compiled_report=None,
        next_action=None,
        next_worker_guidance=None,
        final_status=None,
        iteration_count=0,
        max_iterations=3,
        max_workers=5,
    )


@pytest.fixture
def sample_worker_task() -> WorkerTask:
    """Create a sample worker task."""
    return WorkerTask(
        worker_id="worker_1",
        task_description="Research OSINT definition",
        context="Focus on open source intelligence",
    )


class TestWorkerTask:
    """Tests for WorkerTask TypedDict."""

    def test_worker_task_creation(self):
        """Test creating a WorkerTask."""
        task = WorkerTask(
            worker_id="test_worker",
            task_description="Test task",
            context="Test context",
        )
        assert task["worker_id"] == "test_worker"
        assert task["task_description"] == "Test task"
        assert task["context"] == "Test context"


class TestWorkerResult:
    """Tests for WorkerResult TypedDict."""

    def test_worker_result_creation(self):
        """Test creating a WorkerResult."""
        result = WorkerResult(
            worker_id="test_worker",
            findings="Found important information",
            success=True,
            iteration=1,
        )
        assert result["worker_id"] == "test_worker"
        assert result["findings"] == "Found important information"
        assert result["success"] is True
        assert result["iteration"] == 1


class TestJudgmentDecision:
    """Tests for JudgmentDecision enum."""

    def test_approve_value(self):
        """Test APPROVE enum value."""
        assert JudgmentDecision.APPROVE.value == "approve"

    def test_retry_value(self):
        """Test RETRY enum value."""
        assert JudgmentDecision.RETRY.value == "retry"


class TestChooseWorkers:
    """Tests for the choose_workers node."""

    def test_creates_up_to_max_workers(self, sample_state, mock_llm):
        """Test that choose_workers creates between 1 and max_workers tasks."""
        result = choose_workers(sample_state, mock_llm)

        assert "worker_tasks" in result
        assert 1 <= len(result["worker_tasks"]) <= sample_state["max_workers"]

    def test_increments_iteration_count(self, sample_state, mock_llm):
        """Test that iteration count is incremented."""
        result = choose_workers(sample_state, mock_llm)

        assert result["iteration_count"] == 1

    def test_worker_ids_are_unique(self, sample_state, mock_llm):
        """Test that worker IDs are unique."""
        result = choose_workers(sample_state, mock_llm)

        worker_ids = [task["worker_id"] for task in result["worker_tasks"]]
        assert len(worker_ids) == len(set(worker_ids))

    def test_calls_llm(self, sample_state, mock_llm):
        """Test that the LLM is called."""
        choose_workers(sample_state, mock_llm)

        mock_llm.invoke.assert_called_once()


class TestWorkerNode:
    """Tests for the worker_node function."""

    def test_returns_worker_result(self, sample_worker_task, mock_llm):
        """Test that worker_node returns a proper result."""
        worker_state = WorkerState(
            task=sample_worker_task,
            input="Test input",
            iteration=1,
        )

        result = worker_node(worker_state, mock_llm, worker_factories=[], rng=random.Random(0))

        assert "worker_results" in result
        assert len(result["worker_results"]) == 1
        assert result["worker_results"][0]["worker_id"] == "worker_1"
        assert result["worker_results"][0]["success"] is True
        assert result["worker_results"][0]["iteration"] == 1

    def test_calls_llm_with_task(self, sample_worker_task, mock_llm):
        """Test that worker calls LLM with task details."""
        worker_state = WorkerState(
            task=sample_worker_task,
            input="Test input",
            iteration=1,
        )

        worker_node(worker_state, mock_llm, worker_factories=[], rng=random.Random(0))

        mock_llm.invoke.assert_called_once()


class TestObserveAndCompile:
    """Tests for the observe_and_compile node."""

    def test_compiles_worker_results(self, sample_state, mock_llm):
        """Test that results are compiled from workers."""
        sample_state["iteration_count"] = 1
        sample_state["worker_results"] = [
            WorkerResult(worker_id="worker_1", findings="Finding 1", success=True, iteration=1),
            WorkerResult(worker_id="worker_2", findings="Finding 2", success=True, iteration=1),
        ]

        result = observe_and_compile(sample_state, mock_llm)

        assert "compiled_output" in result
        assert result["compiled_output"] == "Mock LLM response"

    def test_handles_empty_results(self, sample_state, mock_llm):
        """Test handling of empty worker results."""
        sample_state["worker_results"] = []
        sample_state["iteration_count"] = 1

        result = observe_and_compile(sample_state, mock_llm)

        assert "compiled_output" in result

    def test_filters_by_current_iteration(self, sample_state, mock_llm):
        """Test that only current iteration results are used."""
        sample_state["iteration_count"] = 2
        sample_state["worker_results"] = [
            # Previous iteration results (should be ignored)
            WorkerResult(worker_id="worker_1", findings="Old Finding", success=True, iteration=1),
            # Current iteration results
            WorkerResult(worker_id="worker_1", findings="New Finding", success=True, iteration=2),
        ]

        observe_and_compile(sample_state, mock_llm)

        # Verify LLM was called - the exact content depends on filtering
        mock_llm.invoke.assert_called_once()

    def test_structured_output_fills_citations_as_models_not_dicts(self, sample_state):
        """Regression test: filling citations after structured output must not leave dicts (pydantic v2)."""
        from odr.agents.retriever.contracts import CompiledReport

        class StructuredLLM:
            def with_structured_output(self, _schema, method=None):  # noqa: ANN001
                class Runner:
                    def invoke(self, _messages):  # noqa: ANN001
                        # Structured report but missing citations (common), so compiler fills them.
                        return CompiledReport(answer="hello", citations=[])

                return Runner()

        llm = StructuredLLM()
        sample_state["iteration_count"] = 1
        sample_state["worker_results"] = [
            WorkerResult(
                worker_id="worker_1",
                findings="See https://example.com/source for details",
                success=True,
                iteration=1,
                evidence=[{"url": "https://example.com/source", "excerpt": "details"}],
            )
        ]

        result = observe_and_compile(sample_state, llm)  # type: ignore[arg-type]
        # If citations were dicts, render_compiled_report would raise AttributeError.
        assert "compiled_output" in result
        assert "https://example.com/source" in result["compiled_output"]


class TestJudgment:
    """Tests for the judgment node."""

    def test_returns_approve_decision(self, sample_state, mock_llm):
        """Test that judgment returns an APPROVE decision (via counsel)."""
        sample_state["compiled_output"] = "Good output"

        counsel = MagicMock()
        counsel.evaluate.return_value = {
            "final_decision": JudgmentDecision.APPROVE,
            "deliberation_summary": "approve",
        }
        result = judgment(sample_state, counsel)

        assert result["judgment_decision"] == JudgmentDecision.APPROVE
        assert result["judge_feedback"] == "approve"

    def test_returns_retry_decision(self, sample_state, mock_llm):
        """Test that judgment returns a RETRY decision (via counsel)."""
        sample_state["compiled_output"] = "Incomplete output"

        counsel = MagicMock()
        counsel.evaluate.return_value = {
            "final_decision": JudgmentDecision.RETRY,
            "deliberation_summary": "retry",
        }
        result = judgment(sample_state, counsel)

        assert result["judgment_decision"] == JudgmentDecision.RETRY
        assert result["judge_feedback"] == "retry"

class TestDecideNextAction:
    """Tests for the decide_next_action node."""

    def test_stops_best_effort_at_iteration_cap(self, sample_state, mock_llm):
        sample_state["iteration_count"] = 3
        sample_state["max_iterations"] = 3
        sample_state["compiled_output"] = "Output"
        sample_state["judgment_decision"] = JudgmentDecision.RETRY
        sample_state["judge_feedback"] = "needs more sources"
        result = decide_next_action(sample_state, mock_llm)
        assert result["next_action"] == "stop_best_effort"
        assert result["final_status"] == "best_effort"

    def test_continues_when_judge_retries(self, sample_state, mock_llm):
        sample_state["iteration_count"] = 1
        sample_state["max_iterations"] = 3
        sample_state["compiled_output"] = "Output"
        sample_state["judgment_decision"] = JudgmentDecision.RETRY
        result = decide_next_action(sample_state, mock_llm)
        assert result["next_action"] == "continue"

    def test_finishes_when_no_retry(self, sample_state, mock_llm):
        sample_state["iteration_count"] = 1
        sample_state["max_iterations"] = 3
        sample_state["compiled_output"] = "Output"
        sample_state["judgment_decision"] = JudgmentDecision.APPROVE
        result = decide_next_action(sample_state, mock_llm)
        assert result["next_action"] == "finish"
        assert result["final_status"] == "success"

    def test_overrides_continue_to_finish_when_judge_approves_with_structured_output(self, sample_state):
        """If the judge already APPROVEs, we should not keep looping even if the LLM suggests continue."""
        from odr.agents.retriever.contracts import NextAction, NextActionDecision

        class StructuredLLM:
            def with_structured_output(self, _schema, method=None):  # noqa: ANN001
                class Runner:
                    def invoke(self, _messages):  # noqa: ANN001
                        return NextActionDecision(
                            action=NextAction.CONTINUE,
                            rationale="More research is needed.",
                            gaps=["optional verification"],
                            next_worker_guidance="do more",
                        )

                return Runner()

        llm = StructuredLLM()
        sample_state["iteration_count"] = 1
        sample_state["max_iterations"] = 10
        sample_state["compiled_output"] = "Output"
        sample_state["judgment_decision"] = JudgmentDecision.APPROVE

        result = decide_next_action(sample_state, llm)  # type: ignore[arg-type]
        assert result["next_action"] == "finish"
        assert result["final_status"] == "success"


class TestRouteAfterJudgment:
    """Tests for the route_after_judgment function."""

    def test_routes_to_choose_workers_on_retry(self, sample_state):
        """Test routing to choose_workers on RETRY decision."""
        sample_state["judgment_decision"] = JudgmentDecision.RETRY

        result = route_after_judgment(sample_state)

        assert result == "choose_workers"

    def test_routes_to_end_on_approve(self, sample_state):
        """Test routing to END on APPROVE decision."""
        sample_state["judgment_decision"] = JudgmentDecision.APPROVE

        result = route_after_judgment(sample_state)

        assert result == "__end__"  # END constant value

    def test_routes_to_end_on_none(self, sample_state):
        """Test routing to END when decision is None."""
        sample_state["judgment_decision"] = None

        result = route_after_judgment(sample_state)

        assert result == "__end__"


class TestRouteAfterDecision:
    """Tests for the route_after_decision function."""

    def test_routes_to_choose_workers_on_continue(self, sample_state):
        sample_state["next_action"] = "continue"
        assert route_after_decision(sample_state) == "choose_workers"

    def test_routes_to_end_on_finish(self, sample_state):
        sample_state["next_action"] = "finish"
        assert route_after_decision(sample_state) == "__end__"


class TestFanOutToWorkers:
    """Tests for the fan_out_to_workers function."""

    def test_creates_send_for_each_task(self, sample_state):
        """Test that Send is created for each worker task."""
        sample_state["iteration_count"] = 1
        sample_state["worker_tasks"] = [
            WorkerTask(worker_id="w1", task_description="Task 1", context=""),
            WorkerTask(worker_id="w2", task_description="Task 2", context=""),
        ]

        sends = fan_out_to_workers(sample_state)

        assert len(sends) == 2

    def test_empty_tasks_returns_empty_list(self, sample_state):
        """Test that empty tasks returns empty list."""
        sample_state["worker_tasks"] = []
        sample_state["iteration_count"] = 1

        sends = fan_out_to_workers(sample_state)

        assert len(sends) == 0

    def test_sends_include_iteration(self, sample_state):
        """Test that Send includes iteration in worker state."""
        sample_state["iteration_count"] = 2
        sample_state["worker_tasks"] = [
            WorkerTask(worker_id="w1", task_description="Task 1", context=""),
        ]

        sends = fan_out_to_workers(sample_state)

        assert len(sends) == 1
        # The Send object contains the WorkerState with iteration
        assert sends[0].arg["iteration"] == 2


class TestRetrieverClass:
    """Tests for the Retriever class."""

    def test_initialization(self, mock_llm):
        """Test Retriever initialization."""
        retriever = Retriever(llm=mock_llm, max_iterations=5)

        assert retriever.llm == mock_llm
        assert retriever.max_iterations == 5
        assert retriever.graph is not None

    def test_default_max_iterations(self, mock_llm):
        """Test default max_iterations value."""
        retriever = Retriever(llm=mock_llm)

        assert retriever.max_iterations == 3

    def test_run_creates_initial_state(self, mock_llm):
        """Test that run creates proper initial state."""
        # Create a mock that simulates graph execution
        with patch.object(Retriever, '__init__', lambda x, llm, max_iterations=3: None):
            retriever = Retriever.__new__(Retriever)
            retriever.llm = mock_llm
            retriever.max_iterations = 3
            retriever.max_workers = 5
            retriever.graph = MagicMock()
            retriever.graph.invoke.return_value = {"compiled_output": "Final result"}

            result = retriever.run("Test input")

            # Verify graph.invoke was called with initial state
            retriever.graph.invoke.assert_called_once()
            call_args = retriever.graph.invoke.call_args[0][0]
            assert call_args["input"] == "Test input"
            assert call_args["iteration_count"] == 0
            assert call_args["max_iterations"] == 3
            assert call_args["max_workers"] == 5
            assert call_args["judge_feedback"] is None
            assert call_args["compiled_report"] is None
            assert call_args["next_action"] is None
            assert call_args["next_worker_guidance"] is None
            assert call_args["final_status"] is None

