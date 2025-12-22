"""Judge Counsel Agent - A multi-judge voting system for OSINT evidence validation.

This agent implements a fan-out voting pattern where multiple judge instances
evaluate evidence quality and reach a consensus through majority voting.

The counsel is specifically designed for OSINT research validation:
- Entity disambiguation (is this the right John Smith?)
- Source cross-referencing (do independent sources corroborate?)
- Temporal consistency (do dates/timelines align?)
- Citation chain validation (can claims be traced to primary sources?)
- Conflicting evidence resolution (how to handle contradictions?)

Each judge is randomly assigned a specialized persona focused on a different
aspect of evidence validation, providing diverse evaluation perspectives.
"""

from __future__ import annotations

import operator
import os
import random
from enum import Enum
from typing import Annotated, Any, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from odr.integrations.observability import get_observed_llm


class JudgmentDecision(str, Enum):
    """Possible decisions from the judgment step."""

    APPROVE = "approve"
    RETRY = "retry"


class JudgeVote(TypedDict):
    """A vote from an individual judge."""

    judge_id: str
    decision: JudgmentDecision
    reasoning: str
    confidence: float


class JudgeCounselState(TypedDict):
    """State for the Judge Counsel agent graph.

    Attributes:
        original_query: The original input query being evaluated.
        compiled_output: The output to evaluate.
        iteration_count: Current iteration number.
        max_iterations: Maximum allowed iterations.
        num_judges: Number of judges on the counsel.
        personas: List of available judge personas.
        judge_votes: Votes from all judges (merged via operator.add).
        final_decision: The consensus decision after vote aggregation.
        deliberation_summary: Summary of the voting deliberation.
    """

    original_query: str
    compiled_output: str
    iteration_count: int
    max_iterations: int
    num_judges: int
    personas: list[str]
    judge_votes: Annotated[list[JudgeVote], operator.add]
    final_decision: JudgmentDecision | None
    deliberation_summary: str


class JudgeWorkerState(TypedDict):
    """State passed to individual judge worker nodes."""

    judge_id: str
    persona: str
    original_query: str
    compiled_output: str
    iteration_count: int
    max_iterations: int


# Default judge personas for OSINT evidence validation
DEFAULT_PERSONAS: list[str] = [
    # Entity Disambiguation
    """You are an entity disambiguation specialist. Your focus is determining whether 
evidence actually refers to the correct subject (person, organization, event). 
Look for: unique identifiers, contextual clues that confirm identity, potential 
confusion with similarly-named entities. Flag when "John Smith" could be anyone.""",

    # Source Cross-Reference Validation
    """You are a cross-reference validator. Your focus is whether claims are supported 
by multiple independent sources. Look for: URLs that reference each other, citations 
that can be verified, whether sources are truly independent or just copying each other. 
Single-source claims are weak evidence.""",

    # Temporal Consistency
    """You are a temporal consistency analyst. Your focus is whether dates, timelines, 
and chronological claims align across sources. Look for: conflicting dates for the 
same event, anachronistic claims, timeline gaps, whether newer sources contradict 
older authoritative sources.""",

    # Primary Source Verification
    """You are a primary source verifier. Your focus is distinguishing primary sources 
(original documents, firsthand accounts) from secondary sources (news articles, 
summaries). Look for: original source citations, whether claims trace back to 
verifiable primary evidence, or are just repeated assertions.""",

    # Domain Authority Assessment
    """You are a domain authority assessor. Your focus is evaluating the credibility 
of source domains and URLs. Look for: official vs unofficial sources, known reliable 
domains vs questionable ones, archived vs live sources, potential for manipulation 
or spoofing.""",

    # Corroboration Checker
    """You are a corroboration specialist. Your focus is whether key claims have 
independent confirmation. Look for: triangulation from multiple unrelated sources, 
claims that only appear in one place, evidence of primary research vs aggregation, 
whether corroborating sources add new information or just repeat.""",

    # Conflicting Evidence Resolver
    """You are a conflict resolution analyst. Your focus is identifying and evaluating 
contradictory evidence. Look for: sources that disagree, which source is more 
authoritative when conflicts exist, whether conflicts indicate the wrong entity, 
outdated information, or genuine uncertainty.""",

    # Attribution Chain Validator
    """You are an attribution chain validator. Your focus is tracing claims back to 
their origin. Look for: broken citation chains, circular references, claims that 
cite each other without independent verification, original source accessibility, 
whether the attribution chain is complete or has gaps.""",

    # Bias and Agenda Detector
    """You are a bias and agenda detector. Your focus is identifying potential 
source bias that could affect accuracy. Look for: sources with known agendas, 
one-sided presentation, missing counter-evidence, promotional content disguised 
as information, potential for deliberate misinformation.""",

    # Evidence Sufficiency Evaluator
    """You are an evidence sufficiency evaluator. Your focus is whether there is 
enough quality evidence to support conclusions. Look for: thin evidence for strong 
claims, speculation presented as fact, logical leaps without supporting data, 
whether the evidence actually proves what it claims to prove.""",
]


def create_judge_task(state: JudgeCounselState) -> dict[str, Any]:
    """Initial node that prepares the judging task.

    Args:
        state: Current counsel state.

    Returns:
        State unchanged, serves as entry point before fan-out.
    """
    return {}


def judge_worker(state: JudgeWorkerState, llm: BaseChatModel) -> dict[str, Any]:
    """Individual judge worker that evaluates the output.

    Each judge has a randomly assigned persona for diverse evaluation perspectives.

    Args:
        state: Judge-specific state with evaluation context and assigned persona.
        llm: Language model for evaluation.

    Returns:
        Judge vote to be merged into counsel state.
    """
    judge_id = state["judge_id"]
    persona = state["persona"]

    original_query = state["original_query"]
    compiled_output = state["compiled_output"]
    iteration = state["iteration_count"]
    max_iterations = state["max_iterations"]

    system_prompt = f"""{persona}

You are a judge on an OSINT evidence validation counsel. Your role is to evaluate 
whether collected evidence is accurate, properly attributed, and actually supports 
the research query.

CRITICAL EVALUATION CRITERIA:
1. ENTITY ACCURACY: Does evidence clearly identify the correct subject? Could this 
   be confused with a different person/org/event with a similar name?
2. SOURCE VERIFICATION: Are claims backed by verifiable URLs/citations? Do sources 
   cross-reference each other independently?
3. TEMPORAL CONSISTENCY: Do dates and timelines align across sources? Are there 
   contradictions that suggest wrong entity or outdated info?
4. EVIDENCE CHAIN: Can claims be traced to primary sources? Or are they unsupported 
   assertions repeated without verification?
5. SUFFICIENCY: Is there enough quality evidence, or are conclusions based on 
   thin/speculative data?

DECISION CRITERIA:
- APPROVE: Evidence is well-sourced, entity is clearly identified, sources corroborate 
  each other, no major red flags for misidentification or unverified claims.
- RETRY: Evidence is thin, entity could be wrong person/org, sources don't cross-reference, 
  claims lack verification, or there are conflicting/contradictory sources that need resolution.

Respond in this exact format:
DECISION: [APPROVE or RETRY]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Explain your evidence assessment in 2-3 sentences, noting specific concerns]"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Original Query: {original_query}\n\n"
            f"Output to Evaluate:\n{compiled_output}\n\n"
            f"Current Iteration: {iteration}/{max_iterations}"
        ),
    ]

    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)

    # Parse the response
    decision = JudgmentDecision.APPROVE
    confidence = 0.5
    reasoning = content

    lines = content.strip().split("\n")
    for line in lines:
        line_upper = line.upper()
        if line_upper.startswith("DECISION:"):
            if "RETRY" in line_upper:
                decision = JudgmentDecision.RETRY
            else:
                decision = JudgmentDecision.APPROVE
        elif line_upper.startswith("CONFIDENCE:"):
            try:
                conf_str = line.split(":", 1)[1].strip()
                confidence = float(conf_str)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, IndexError):
                confidence = 0.5
        elif line_upper.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip() if ":" in line else reasoning

    return {
        "judge_votes": [
            JudgeVote(
                judge_id=judge_id,
                decision=decision,
                reasoning=reasoning,
                confidence=confidence,
            )
        ]
    }


def fan_out_to_judges(state: JudgeCounselState) -> list[Send]:
    """Create Send commands for parallel judge execution.

    Each judge is assigned a random persona from the available personas.

    Args:
        state: Current counsel state.

    Returns:
        List of Send commands, one for each judge on the counsel.
    """
    num_judges = state.get("num_judges", 10)
    personas = state.get("personas", DEFAULT_PERSONAS)

    # Assign a random persona to each judge
    assigned_personas = [random.choice(personas) for _ in range(num_judges)]

    return [
        Send(
            "judge_worker",
            JudgeWorkerState(
                judge_id=f"judge_{i}",
                persona=assigned_personas[i - 1],
                original_query=state["original_query"],
                compiled_output=state["compiled_output"],
                iteration_count=state["iteration_count"],
                max_iterations=state["max_iterations"],
            ),
        )
        for i in range(1, num_judges + 1)
    ]


def aggregate_votes(state: JudgeCounselState) -> dict[str, Any]:
    """Aggregate votes from all judges and determine final decision.

    Uses majority voting with confidence weighting.

    Args:
        state: Current counsel state with all judge votes.

    Returns:
        Updated state with final decision and deliberation summary.
    """
    votes = state.get("judge_votes", [])

    if not votes:
        return {
            "final_decision": JudgmentDecision.APPROVE,
            "deliberation_summary": "No votes received. Defaulting to APPROVE.",
        }

    # Count votes with confidence weighting
    approve_weight = 0.0
    retry_weight = 0.0
    approve_count = 0
    retry_count = 0

    for vote in votes:
        if vote["decision"] == JudgmentDecision.APPROVE:
            approve_weight += vote["confidence"]
            approve_count += 1
        else:
            retry_weight += vote["confidence"]
            retry_count += 1

    # Decision based on weighted majority
    if approve_weight >= retry_weight:
        final_decision = JudgmentDecision.APPROVE
    else:
        final_decision = JudgmentDecision.RETRY

    # Build deliberation summary
    vote_breakdown = f"APPROVE: {approve_count} votes (weight: {approve_weight:.2f}), "
    vote_breakdown += f"RETRY: {retry_count} votes (weight: {retry_weight:.2f})"

    reasoning_samples = []
    for vote in votes[:3]:  # Sample first 3 reasonings
        reasoning_samples.append(f"- Judge {vote['judge_id']}: {vote['reasoning'][:100]}...")

    deliberation_summary = (
        f"Judge Counsel Deliberation Complete\n"
        f"{'=' * 40}\n"
        f"Final Decision: {final_decision.value.upper()}\n"
        f"Vote Breakdown: {vote_breakdown}\n"
        f"\nSample Reasoning:\n" + "\n".join(reasoning_samples)
    )

    return {
        "final_decision": final_decision,
        "deliberation_summary": deliberation_summary,
    }


def create_judge_counsel_graph(counsel_llm: BaseChatModel) -> CompiledStateGraph:
    """Create the Judge Counsel agent graph.

    The graph implements the following flow:
    1. create_judge_task: Entry point, prepares evaluation context
    2. judge_worker (fan-out): Configurable number of judges evaluate in parallel
    3. aggregate_votes: Tally votes and determine consensus

    Args:
        counsel_llm: Language model for all judge workers.

    Returns:
        Compiled StateGraph ready for execution.
    """
    graph = StateGraph(JudgeCounselState)

    # Bind LLM to judge worker
    def _judge_worker(state: Any) -> dict[str, Any]:
        return judge_worker(cast(JudgeWorkerState, state), counsel_llm)

    # Add nodes
    graph.add_node("create_judge_task", create_judge_task)
    graph.add_node("judge_worker", _judge_worker)
    graph.add_node("aggregate_votes", aggregate_votes)

    # Set entry point
    graph.set_entry_point("create_judge_task")

    # Fan out from task creation to all judges
    graph.add_conditional_edges(
        "create_judge_task",
        fan_out_to_judges,  # type: ignore[arg-type]
    )

    # All judges converge to vote aggregation
    graph.add_edge("judge_worker", "aggregate_votes")

    # Aggregation ends the graph
    graph.add_edge("aggregate_votes", END)

    return graph.compile()


class JudgeCounsel:
    """Judge Counsel agent with configurable multi-judge voting system.

    This agent coordinates multiple judge workers to evaluate research output,
    each with a randomly assigned persona for diverse perspectives. Votes are
    aggregated using confidence-weighted majority voting.

    Uses the observability module for automatic Langfuse tracing when
    creating the default LLM.

    Example:
        ```python
        from odr.agents import JudgeCounsel

        # Default: 10 judges using gpt-5-nano-2025-08-07 with built-in personas
        counsel = JudgeCounsel()

        # Custom: 5 judges with custom personas
        counsel = JudgeCounsel(
            model="gpt-4o-mini",
            num_judges=5,
            personas=[
                "You are a security expert focused on threat analysis.",
                "You are a data analyst focused on statistical validity.",
                "You are a domain expert in the query's subject matter.",
            ],
            api_key="your-key",
        )

        result = counsel.evaluate(
            original_query="What is quantum computing?",
            compiled_output="Quantum computing uses qubits...",
        )
        print(result["final_decision"])
        ```
    """

    DEFAULT_MODEL = "gpt-5-nano-2025-08-07"
    DEFAULT_NUM_JUDGES = 10

    def __init__(
        self,
        counsel_llm: BaseChatModel | None = None,
        model: str | None = None,
        num_judges: int = DEFAULT_NUM_JUDGES,
        personas: list[str] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize the Judge Counsel.

        Args:
            counsel_llm: Language model for judge workers. If provided, model,
                        api_key, and base_url are ignored.
            model: Model name for the counsel (default: gpt-5-nano-2025-08-07).
            num_judges: Number of judges on the counsel (default: 10).
            personas: List of judge personas. Each judge is randomly assigned
                     one persona from this list. Defaults to built-in personas.
            api_key: API key for the counsel model. Falls back to API_KEY or
                    OPENAI_API_KEY environment variables.
            base_url: Base URL for the API endpoint.
        """
        self.num_judges = num_judges
        self.personas = personas or DEFAULT_PERSONAS
        self._model = model or self.DEFAULT_MODEL
        self._api_key = api_key
        self._base_url = base_url

        if counsel_llm is None:
            # Resolve API key from args or environment. If absent, defer initialization until evaluate().
            resolved_api_key = self._api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            resolved_base_url = self._base_url or os.getenv("MODEL_URL")

            if resolved_api_key:
                counsel_llm = get_observed_llm(
                    model=self._model,
                    api_key=resolved_api_key,
                    base_url=resolved_base_url,
                )

        self.counsel_llm = counsel_llm
        self.graph = create_judge_counsel_graph(counsel_llm) if counsel_llm is not None else None

    def _ensure_initialized(self) -> None:
        """Ensure the counsel has an LLM and compiled graph.

        This defers requiring API keys until evaluate-time, which keeps unit tests
        and offline usage from failing at import/instantiation.
        """
        if self.graph is not None and self.counsel_llm is not None:
            return

        resolved_api_key = self._api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        resolved_base_url = self._base_url or os.getenv("MODEL_URL")

        if not resolved_api_key:
            raise RuntimeError(
                "JudgeCounsel requires an API key to run. Set OPENAI_API_KEY (or API_KEY), "
                "or pass counsel_llm/api_key when constructing JudgeCounsel."
            )

        self.counsel_llm = get_observed_llm(
            model=self._model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        )
        self.graph = create_judge_counsel_graph(self.counsel_llm)

    def evaluate(
        self,
        original_query: str,
        compiled_output: str,
        iteration_count: int = 1,
        max_iterations: int = 3,
    ) -> JudgeCounselState:
        """Execute the judge counsel evaluation.

        Args:
            original_query: The original input query.
            compiled_output: The output to evaluate.
            iteration_count: Current iteration number.
            max_iterations: Maximum allowed iterations.

        Returns:
            Final state with decision and deliberation summary.
        """
        self._ensure_initialized()

        initial_state: JudgeCounselState = {
            "original_query": original_query,
            "compiled_output": compiled_output,
            "iteration_count": iteration_count,
            "max_iterations": max_iterations,
            "num_judges": self.num_judges,
            "personas": self.personas,
            "judge_votes": [],
            "final_decision": None,
            "deliberation_summary": "",
        }

        return cast(JudgeCounselState, cast(Any, self.graph).invoke(initial_state))

    async def aevaluate(
        self,
        original_query: str,
        compiled_output: str,
        iteration_count: int = 1,
        max_iterations: int = 3,
    ) -> JudgeCounselState:
        """Execute the judge counsel evaluation asynchronously.

        Args:
            original_query: The original input query.
            compiled_output: The output to evaluate.
            iteration_count: Current iteration number.
            max_iterations: Maximum allowed iterations.

        Returns:
            Final state with decision and deliberation summary.
        """
        self._ensure_initialized()

        initial_state: JudgeCounselState = {
            "original_query": original_query,
            "compiled_output": compiled_output,
            "iteration_count": iteration_count,
            "max_iterations": max_iterations,
            "num_judges": self.num_judges,
            "personas": self.personas,
            "judge_votes": [],
            "final_decision": None,
            "deliberation_summary": "",
        }

        return cast(JudgeCounselState, await cast(Any, self.graph).ainvoke(initial_state))

