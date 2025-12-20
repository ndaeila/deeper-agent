"""Retriever Agent - A LangGraph agent with supervisor pattern and worker fan-out.

This agent implements a hierarchical workflow:
1. Input → Choose Workers: Supervisor decides which workers should handle the task
2. Fan-out to Workers: Work is distributed to multiple worker instances in parallel
3. Observe & Compile: Results from workers are aggregated into a cohesive output
4. Judgment: Evaluate if output is satisfactory or needs another iteration
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send


class JudgmentDecision(str, Enum):
    """Possible decisions from the judgment step."""

    APPROVE = "approve"
    RETRY = "retry"


class WorkerTask(TypedDict):
    """A task assigned to a worker."""

    worker_id: str
    task_description: str
    context: str


class WorkerResult(TypedDict):
    """Result from a worker's execution."""

    worker_id: str
    findings: str
    success: bool
    iteration: int


class RetrieverState(TypedDict):
    """State for the Retriever agent graph.

    Attributes:
        input: The original input query or task.
        messages: Conversation history for context.
        worker_tasks: List of tasks assigned to workers.
        worker_results: Results collected from workers (uses operator.add for parallel merge).
        compiled_output: The aggregated output from all worker results.
        judgment_decision: The decision from the judgment step.
        iteration_count: Number of iterations through the loop.
        max_iterations: Maximum allowed iterations before forced exit.
    """

    input: str
    messages: Annotated[list[BaseMessage], operator.add]
    worker_tasks: list[WorkerTask]
    worker_results: Annotated[list[WorkerResult], operator.add]
    compiled_output: str
    judgment_decision: JudgmentDecision | None
    iteration_count: int
    max_iterations: int


class WorkerState(TypedDict):
    """State passed to individual worker nodes."""

    task: WorkerTask
    input: str
    iteration: int


# --- Node Functions ---


def choose_workers(state: RetrieverState, llm: BaseChatModel) -> dict[str, Any]:
    """Supervisor node that decides which workers should handle the task.

    This node analyzes the input and creates task assignments for workers.
    Currently spawns 2 workers with the same role for parallel processing.

    Args:
        state: Current graph state.
        llm: Language model for decision making.

    Returns:
        Updated state with worker_tasks assigned.
    """
    input_text = state["input"]
    iteration = state.get("iteration_count", 0)

    # Build context from previous attempts if any
    previous_context = ""
    if iteration > 0 and state.get("compiled_output"):
        previous_context = f"\n\nPrevious attempt output:\n{state['compiled_output']}"

    system_prompt = """You are a supervisor coordinating research workers.
Analyze the input and create task descriptions for workers to investigate.
Each worker should focus on a specific aspect of the research task.
Be specific about what each worker should find and return."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Research task: {input_text}{previous_context}\n\n"
            "Create focused task descriptions for 2 parallel workers."
        ),
    ]

    response = llm.invoke(messages)

    # Create 2 worker tasks with slightly different focus areas
    worker_tasks = [
        WorkerTask(
            worker_id="worker_1",
            task_description=f"Primary investigation: {input_text}",
            context=response.content if isinstance(response.content, str) else str(response.content),
        ),
        WorkerTask(
            worker_id="worker_2",
            task_description=f"Secondary verification: {input_text}",
            context=response.content if isinstance(response.content, str) else str(response.content),
        ),
    ]

    return {
        "worker_tasks": worker_tasks,
        "messages": [response],
        "iteration_count": iteration + 1,
    }


def worker_node(state: WorkerState, llm: BaseChatModel) -> dict[str, Any]:
    """Individual worker node that processes an assigned task.

    Args:
        state: Worker-specific state with task details.
        llm: Language model for task execution.

    Returns:
        Worker result to be merged into parent state.
    """
    task = state["task"]
    iteration = state.get("iteration", 1)

    system_prompt = f"""You are a research worker with ID: {task['worker_id']}.
Your job is to investigate the assigned task and provide detailed findings.
Be thorough and factual in your research."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Task: {task['task_description']}\n\nContext from supervisor: {task['context']}"
        ),
    ]

    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "worker_results": [
            WorkerResult(
                worker_id=task["worker_id"],
                findings=content,
                success=True,
                iteration=iteration,
            )
        ]
    }


def observe_and_compile(state: RetrieverState, llm: BaseChatModel) -> dict[str, Any]:
    """Observer node that aggregates findings from all workers.

    Args:
        state: Current graph state with worker results.
        llm: Language model for synthesis.

    Returns:
        Updated state with compiled output.
    """
    all_worker_results = state.get("worker_results", [])
    current_iteration = state.get("iteration_count", 1)
    original_input = state["input"]

    # Filter to only current iteration's results
    worker_results = [r for r in all_worker_results if r.get("iteration") == current_iteration]

    # Format worker findings for synthesis
    findings_text = "\n\n".join(
        f"=== {result['worker_id'].upper()} FINDINGS ===\n{result['findings']}"
        for result in worker_results
    )

    system_prompt = """You are an analyst synthesizing research findings from multiple workers.
Combine the findings into a cohesive, comprehensive response.
Highlight agreements, note any discrepancies, and provide a unified conclusion."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Original query: {original_input}\n\n"
            f"Worker findings to synthesize:\n{findings_text}"
        ),
    ]

    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "compiled_output": content,
        "messages": [response],
    }


def judgment(state: RetrieverState, llm: BaseChatModel) -> dict[str, Any]:
    """Judgment node that evaluates if the output is satisfactory.

    This node decides whether to approve the output or request another iteration.

    Args:
        state: Current graph state with compiled output.
        llm: Language model for evaluation.

    Returns:
        Updated state with judgment decision.
    """
    compiled_output = state.get("compiled_output", "")
    original_input = state["input"]
    iteration = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 3)

    # Force approval if max iterations reached
    if iteration >= max_iterations:
        return {
            "judgment_decision": JudgmentDecision.APPROVE,
            "messages": [
                AIMessage(content=f"Max iterations ({max_iterations}) reached. Approving output.")
            ],
        }

    system_prompt = """You are a quality judge evaluating research output.
Determine if the output adequately addresses the original query.
Respond with exactly one of: APPROVE or RETRY

APPROVE: The output is comprehensive and accurate.
RETRY: The output needs more investigation or has gaps."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Original query: {original_input}\n\n"
            f"Compiled output to evaluate:\n{compiled_output}\n\n"
            f"Current iteration: {iteration}/{max_iterations}\n\n"
            "Decision (APPROVE or RETRY):"
        ),
    ]

    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)

    # Parse decision from response
    decision = JudgmentDecision.APPROVE
    if "RETRY" in content.upper():
        decision = JudgmentDecision.RETRY

    return {
        "judgment_decision": decision,
        "messages": [response],
    }


def route_after_judgment(state: RetrieverState) -> str:
    """Routing function to determine next step after judgment.

    Args:
        state: Current graph state.

    Returns:
        Next node name: 'choose_workers' for retry, END for approval.
    """
    decision = state.get("judgment_decision")

    if decision == JudgmentDecision.RETRY:
        return "choose_workers"

    return END


def fan_out_to_workers(state: RetrieverState) -> list[Send]:
    """Create Send commands for parallel worker execution.

    Args:
        state: Current graph state with worker tasks.

    Returns:
        List of Send commands, one for each worker task.
    """
    worker_tasks = state.get("worker_tasks", [])
    iteration = state.get("iteration_count", 1)

    return [
        Send(
            "worker",
            WorkerState(task=task, input=state["input"], iteration=iteration),
        )
        for task in worker_tasks
    ]


def create_retriever_graph(llm: BaseChatModel) -> CompiledStateGraph:
    """Create the Retriever agent graph.

    The graph implements the following flow:
    1. choose_workers: Supervisor assigns tasks to workers
    2. worker (fan-out): Multiple workers process tasks in parallel
    3. observe_and_compile: Aggregate worker findings
    4. judgment: Evaluate output quality
    5. Either loop back to choose_workers or end

    Args:
        llm: Language model to use for all nodes.

    Returns:
        Compiled StateGraph ready for execution.
    """
    # Create graph with state schema
    graph = StateGraph(RetrieverState)

    # Add nodes with LLM bound
    # Using explicit functions to avoid type narrowing issues with lambdas
    def _choose_workers(state: Any) -> dict[str, Any]:
        return choose_workers(cast(RetrieverState, state), llm)

    def _worker(state: Any) -> dict[str, Any]:
        return worker_node(cast(WorkerState, state), llm)

    def _observe_and_compile(state: Any) -> dict[str, Any]:
        return observe_and_compile(cast(RetrieverState, state), llm)

    def _judgment(state: Any) -> dict[str, Any]:
        return judgment(cast(RetrieverState, state), llm)

    graph.add_node("choose_workers", _choose_workers)
    graph.add_node("worker", _worker)
    graph.add_node("observe_and_compile", _observe_and_compile)
    graph.add_node("judgment", _judgment)

    # Set entry point
    graph.set_entry_point("choose_workers")

    # Add edges
    # After choosing workers, fan out to all workers in parallel
    # When using Send, the path_map is not needed - targets come from Send objects
    graph.add_conditional_edges(
        "choose_workers",
        fan_out_to_workers,  # type: ignore[arg-type]
    )

    # Workers converge to observe_and_compile
    graph.add_edge("worker", "observe_and_compile")

    # After compilation, go to judgment
    graph.add_edge("observe_and_compile", "judgment")

    # Judgment decides to retry or end
    graph.add_conditional_edges(
        "judgment",
        route_after_judgment,
        {
            "choose_workers": "choose_workers",
            END: END,
        },
    )

    return graph.compile()


class Retriever:
    """Retriever agent with supervisor pattern and worker fan-out.

    This agent coordinates multiple workers to investigate a query,
    compiles their findings, and uses a judgment step to determine
    if the results are satisfactory or need refinement.

    Example:
        ```python
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4")
        retriever = Retriever(llm=llm)

        result = retriever.run("What is the capital of France?")
        print(result["compiled_output"])
        ```
    """

    def __init__(self, llm: BaseChatModel, max_iterations: int = 3):
        """Initialize the Retriever agent.

        Args:
            llm: Language model to use for all operations.
            max_iterations: Maximum number of retry iterations allowed.
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.graph = create_retriever_graph(llm)

    def run(self, input_text: str) -> RetrieverState:
        """Execute the retriever workflow.

        Args:
            input_text: The query or task to investigate.

        Returns:
            Final state containing compiled output and all intermediate results.
        """
        initial_state: RetrieverState = {
            "input": input_text,
            "messages": [],
            "worker_tasks": [],
            "worker_results": [],
            "compiled_output": "",
            "judgment_decision": None,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
        }

        return cast(RetrieverState, self.graph.invoke(initial_state))

    async def arun(self, input_text: str) -> RetrieverState:
        """Execute the retriever workflow asynchronously.

        Args:
            input_text: The query or task to investigate.

        Returns:
            Final state containing compiled output and all intermediate results.
        """
        initial_state: RetrieverState = {
            "input": input_text,
            "messages": [],
            "worker_tasks": [],
            "worker_results": [],
            "compiled_output": "",
            "judgment_decision": None,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
        }

        return cast(RetrieverState, await self.graph.ainvoke(initial_state))

    def stream(self, input_text: str):
        """Stream the retriever workflow execution.

        Args:
            input_text: The query or task to investigate.

        Yields:
            State updates as the graph progresses through nodes.
        """
        initial_state: RetrieverState = {
            "input": input_text,
            "messages": [],
            "worker_tasks": [],
            "worker_results": [],
            "compiled_output": "",
            "judgment_decision": None,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
        }

        yield from self.graph.stream(initial_state)

