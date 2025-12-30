"""Retriever Agent - A LangGraph agent with supervisor pattern and worker fan-out.

This agent implements a hierarchical workflow:
1. Input -> Choose Workers: Supervisor decides which workers should handle the task
2. Fan-out to Workers: Work is distributed to multiple worker instances in parallel
3. Observe & Compile: Results from workers are aggregated into a cohesive output
4. Judgment: Judge Counsel evaluates via multi-judge voting system

Public API
----------
- Retriever: Main agent class with run/arun/stream methods
- RetrieverState, WorkerState: State TypedDicts
- create_retriever_graph: Low-level graph factory
- choose_workers, worker_node, observe_and_compile, judgment: Node functions
- fan_out_to_workers, route_after_judgment: Routing helpers
"""

from odr.agents.judge_counsel import JudgmentDecision
from odr.agents.types import WorkerResult, WorkerResultExtended, WorkerTask

from .agent import Retriever
from .compiler import observe_and_compile
from .controller import decide_next_action, route_after_decision
from .graph import create_retriever_graph, fan_out_to_workers
from .judgment import judgment, route_after_judgment
from .planner import choose_workers
from .state import RetrieverState, WorkerState
from .worker import worker_node

__all__ = [
    # Agent
    "Retriever",
    # State
    "RetrieverState",
    "WorkerState",
    # Graph factory
    "create_retriever_graph",
    # Nodes
    "choose_workers",
    "worker_node",
    "observe_and_compile",
    "judgment",
    "decide_next_action",
    # Routing
    "fan_out_to_workers",
    "route_after_judgment",
    "route_after_decision",
    # Types (re-exported for convenience)
    "WorkerTask",
    "WorkerResult",
    "WorkerResultExtended",
    "JudgmentDecision",
]

