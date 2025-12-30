"""Browser-use worker implementation (hybrid Tavily + browser agent).

This worker is designed to mirror the user's notebook prototype:
- Tavily for multi-query link discovery
- browser_use Agent for navigation + extraction

Notes:
- Dependencies are optional at import-time. If not installed, a clear ImportError is raised
  at runtime when attempting to create/run this worker.
- This worker runs an async browser_use agent under the hood. In sync contexts (CLI),
  it uses asyncio.run(). If you are already inside an event loop, prefer Retriever.arun().
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import time
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Mapping, cast

from odr.agents.types import WorkerResultExtended, WorkerTask
from odr.agents.workers.base import WorkerFactory
from odr.factory import DefaultLLMFactory
from odr.tools.search import search_for_links, search_provider_available


# ================================
# Install hints and warnings
# ================================


_WARNED_KEYS: set[str] = set()

_INSTALL_HINT = (
    "browser_use worker deps missing. Install:\n"
    "  poetry add selenium browser-use tavily-python langchain-openai\n"
    "Then install browser-use drivers:\n"
    "  uvx browser-use install"
)


def _print_once(key: str, message: str) -> None:
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    print(message)


def _browser_use_available() -> bool:
    return (
        find_spec("browser_use") is not None
        and find_spec("tavily") is not None
        and find_spec("selenium") is not None
    )


def _required_env_present(need_search: bool = True) -> bool:
    has_openai = bool(os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))
    if not need_search:
        return has_openai
    return has_openai and search_provider_available()


def _preflight_print_hints(need_search: bool = True) -> None:
    """Print actionable setup hints early (stdout), so failure isn't only visible at fan-in."""
    if not _browser_use_available():
        _print_once("missing_deps", _INSTALL_HINT)
    if not _required_env_present(need_search):
        msg = "browser_use worker missing OPENAI_API_KEY."
        if need_search:
            msg += " Also needs a search provider (Tavily or SerpApi)."
        _print_once("missing_env", msg)


def _require_browser_use() -> Any:
    try:
        from browser_use import Agent, Browser, Tools  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(_INSTALL_HINT) from e

    return Agent, Browser, Tools


# ================================
# Worker configuration
# ================================

@dataclass(frozen=True)
class BrowserUseWorkerConfig:
    """Configuration for the browser_use worker."""

    model: str = "gpt-5-nano-2025-08-07"
    reasoning_effort: str = "low"
    headless: bool = False
    use_vision: bool | str = "auto"  # Vision mode: "auto" (default), True (always), False (never)
    log_level: str = "WARNING"  # Log level for Browser Use: DEBUG, INFO, WARNING, ERROR, CRITICAL
    tavily_max_results: int = 5
    max_steps: int = 15  # Hard limit on browser agent steps
    max_actions: int = 15  # Some browser_use versions use this instead
    use_search_tool: bool = True  # Set False to disable Tavily/SerpApi search tool
    # Guardrails for Tavily/SerpApi spend: browser_use should search once, then browse/extract.
    max_search_calls: int = 1
    max_search_queries_per_call: int = 3
    # Bound how much history we keep in-memory / event bus payloads.
    max_history_chars: int = 30000
    max_raw_history_chars: int = 200000


# ================================
# Worker implementation
# ================================

class BrowserUseWorker:
    """A worker that uses Tavily + browser_use for evidence collection."""

    worker_type = "browser_use"

    def __init__(self, worker_id: str, llm_factory: DefaultLLMFactory, config: BrowserUseWorkerConfig | None = None):
        self.worker_id = worker_id
        self.llm_factory = llm_factory
        self.config = config or BrowserUseWorkerConfig()

    def run(self, task: Mapping[str, Any], input_text: str, iteration: int) -> WorkerResultExtended:
        task_typed = cast(WorkerTask, task)
        try:
            _preflight_print_hints(need_search=self.config.use_search_tool)
            return self._run_sync(task=task_typed, input_text=input_text, iteration=iteration)
        except ImportError:
            _print_once("missing_deps", _INSTALL_HINT)
            raise
        except Exception as e:
            return WorkerResultExtended(
                worker_id=task_typed["worker_id"],
                findings=f"browser_use worker failed: {e}",
                success=False,
                iteration=iteration,
                worker_type=self.worker_type,
                error=str(e),
            )

    def _run_sync(self, task: WorkerTask, input_text: str, iteration: int) -> WorkerResultExtended:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "browser_use worker cannot call asyncio.run() inside a running event loop. "
                "Use Retriever.arun() / Retriever.graph.ainvoke() in async contexts."
            )

        return asyncio.run(self._run_async(task=task, input_text=input_text, iteration=iteration))

    async def _run_async(self, task: WorkerTask, input_text: str, iteration: int) -> WorkerResultExtended:
        start_time = time.perf_counter()
        Agent, Browser, Tools = _require_browser_use()

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY (or API_KEY) for browser_use ChatOpenAI")

        req = task.get("requirements") if isinstance(task.get("requirements"), dict) else {}
        max_search_calls = self.config.max_search_calls
        max_search_queries_per_call = self.config.max_search_queries_per_call
        max_steps = self.config.max_steps
        max_actions = self.config.max_actions
        if isinstance(req.get("max_search_calls"), int):
            max_search_calls = int(req["max_search_calls"])
        if isinstance(req.get("max_search_queries_per_call"), int):
            max_search_queries_per_call = int(req["max_search_queries_per_call"])
        if isinstance(req.get("max_steps"), int):
            max_steps = int(req["max_steps"])
        if isinstance(req.get("max_actions"), int):
            max_actions = int(req["max_actions"])

        tools: Any = None
        # Enable search tool only if configured AND the per-task budget allows it.
        if self.config.use_search_tool and max_search_calls > 0:
            tools = Tools()
            search_budget = _SearchBudget(
                max_calls=max(0, int(max_search_calls)),
                max_queries_per_call=max(1, int(max_search_queries_per_call)),
            )

            @tools.action(
                description="Initial bulk web search for relevant links using multiple queries at once."
            )
            async def do_search(queries: list[str]) -> str:
                normalized = _normalize_search_queries(queries)
                accepted, status = search_budget.consume(normalized)
                if status == _SearchStatus.BLOCKED:
                    # The prompt tells the agent to STOP when quota is exhausted; reinforce here.
                    return (
                        "SEARCH_QUOTA_EXHAUSTED: `do_search` is only allowed once for this task. "
                        "Do NOT call `do_search` again. Proceed by opening 1-2 URLs from the previous "
                        "results (if any), extract facts with citations, then call done()."
                    )
                if not accepted:
                    return (
                        "NO_VALID_QUERIES: Provide 1-3 short, distinct queries (strings). "
                        "Then proceed to open 1-2 result URLs and extract facts with citations."
                    )
                return search_for_links(
                    queries=accepted,
                    max_results=self.config.tavily_max_results,
                )

        browser = Browser(headless=self.config.headless)
        llm_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "api_key": api_key,
        }
        # Some browser_use versions support reasoning_effort; keep best-effort compatibility.
        if self.config.reasoning_effort:
            llm_kwargs["reasoning_effort"] = self.config.reasoning_effort
        # Use Langfuse-wrapped LLM if observability is enabled
        
        llm = self.llm_factory.get_browser_use_llm(name="browser-use-worker", **llm_kwargs)

        

        # ================================
        # Prompt Sections
        # ================================
        base_q = task["task_description"]  # ← Usually: from choose_workers LLM (nodes.py)

        # Extract deliverables from context if present.
        deliverables_section = ""
        ctx = task.get("context") or ""
        if "Deliverables:" in ctx:
            start = ctx.find("Deliverables:")
            end = ctx.find("\n\n", start)
            if end == -1:
                end = len(ctx)
            deliverables_section = ctx[start:end].strip()

        workflow = (
            "1. First, call `do_search` one time with short query parameters you would use to search for the given topic. Do not overuse this tool.\n"
            "2. Read the returned context and use it to guide what sites you open and what information you extract.\n"
            "3. Then, using the browser, dig deeply into the sites you opened and extract the information you need.\n"
            "4. Stop and return findings with cited URLs.\n\n"

            "Advice (you don't have to follow these, but they may help you):\n"
            "Go deeper on the same domain to find more information about the subject.\n"
            "This means instead of skimming the surface of the first page, you should go deeper on the same domain when possible to find more information about the subject.\n"
            "Leverage the results from do_search to guide your browsing. They pull results like if you were searching on Google for the given topic. Oftentimes the answer can be found directly in the search result before going into the website, so you should use them to guide your browsing.\n"
            "Sometimes if a domain is relevant enough to the entity you're investigating, you should try to find more information about the entity on that domain.\n"
            "One example of this is if you're investigating a person, you can visit their blog or their school/institution/company's website and navigate through relevant pages to find more information about the entity.\n"
            "Sometimes, there is even a search bar on the page you can explore. Other times, you may see your entity is related to some other institution or organization, so you can use general search to find other websites so you can explore that affiliation further.\n"
            "Your choice. Just make sure you're looking as deep as possible into the entity you're investigating, including their other attributes, connnections, relationships and affiliations.\n\n"
            # "WORKFLOW (follow this order):\n"
            # "1. If web search is available, use it at most ONCE to find the best primary/official seed URLs.\n"
            # "   If web search is not available, reuse URLs already present in the task context.\n"
            # "2. Open relevant seed URLs and explore deeper by following internal links and buttons on that same page (or same domain).\n"
            # "3. Based on the possible relevance of a page you visit, deeply research specific facts/quotes/connections within the same domain to see if there is more to learn from the deeper pages.\n"
            # " - For instance, if you are investigating a person, their Twitter is relevant, but the entirety of the Twitter website should not be explored in depth. "
            # "   However, if you are investigating their workplace or school/institution, or even their blog or city of residence, you should explore that domain in depth."
            # "4. Stop and return findings with cited URLs.\n\n"
        )
        nav_constraint = (
            "- You can navigate to URLs present in the task/context and follow same-domain links.\n"
            "- Prefer going deeper on the same domain over adding new domains, unless you see another very relevant domain to your focus.\n"
            "- If you see a search bar on the page you are on, you can use it to find other websites to explore.\n"
            "- If you cannot close a paywall or authwall or if you see something like a captcha on a page, don't want to spend too much time on it and move on to the next site.\n"
        )
        hard_constraints = (
            "HARD CONSTRAINTS:\n"
            + f"- Max {max_steps} browser steps total.\n"
            + "- If search quota is exhausted or blocked, STOP IMMEDIATELY with whatever you have.\n"
            + "- Do NOT wait or retry indefinitely. Return partial results if needed.\n"
            + "- You MUST cite the exact URL for each fact. If uncertain, say so.\n\n"
            + "If blocked/quota exhausted: call done() with success=True and your best partial findings.\n\n"
        )


        # ================================
        # Full System Prompt (task)
        # ================================
        agent_task = (
            f"You are a browsing agent that hunts through the internet to the fullest depth for any information about any subject you are given.\n"
            + "MUST call the `do_search` tool BEFORE you open any webpage or start browsing with 5 concise, human-readable query parameters. The shorter the query, the better it tends to perform.\n\n"
            + workflow
            + nav_constraint
            + hard_constraints

            + "Task from your user:\n"
            + f"{base_q}\n\n"
            + f"Original query: {input_text}\n\n"
            + (f"{deliverables_section}\n\n" if deliverables_section else "")
        )

        # Enforce step budget when supported by the installed browser_use version.
        agent_kwargs: dict[str, Any] = {"task": agent_task, "llm": llm, "browser": browser}
        if tools is not None:
            agent_kwargs["tools"] = tools
        try:
            # `browser_use.Agent` is commonly a class; inspect __init__ for init kwargs.
            target = Agent.__init__ if inspect.isclass(Agent) else Agent
            sig = inspect.signature(target)
            params = sig.parameters
            # Try both parameter names (browser_use versions vary)
            if "max_steps" in params:
                agent_kwargs["max_steps"] = max_steps
            if "max_actions" in params:
                agent_kwargs["max_actions"] = max_actions
            # Add use_vision if supported (helps avoid Langfuse media upload issues)
            if "use_vision" in params:
                agent_kwargs["use_vision"] = self.config.use_vision
        except (TypeError, ValueError):
            # Best-effort; prompt-level constraint still applies.
            pass

        agent = Agent(**agent_kwargs)
        history = await agent.run()

        formatted = _format_browser_use_history(history)
        evidence = _extract_evidence_from_history(history)
        formatted = _truncate_text(formatted, max_chars=self.config.max_history_chars)

        duration = time.perf_counter() - start_time
        print(f"[browser_use worker {self.worker_id}] Completed in {duration:.2f} seconds")
        
        return WorkerResultExtended(
            worker_id=task["worker_id"],
            findings=formatted,
            success=True,
            iteration=iteration,
            worker_type=self.worker_type,
            evidence=evidence,
            context_window={
                "agent_task": agent_task,
                "history_summary": formatted,
            },
            raw_history=_truncate_jsonish(_safe_serialize_history(history), max_chars=self.config.max_raw_history_chars),
        )

# ================================
# Worker factory
# ================================

class BrowserUseWorkerFactory(WorkerFactory):
    """Factory for creating BrowserUseWorker instances."""

    worker_type = "browser_use"

    def __init__(self, llm_factory: DefaultLLMFactory, config: BrowserUseWorkerConfig | None = None):
        self.llm_factory = llm_factory
        self.config = config or BrowserUseWorkerConfig()
        _preflight_print_hints(need_search=self.config.use_search_tool)

    def create(self, worker_id: str) -> BrowserUseWorker:
        return BrowserUseWorker(worker_id=worker_id, llm_factory=self.llm_factory, config=self.config)


# ================================
# Helper functions
# ================================

def _safe_serialize_history(history: Any) -> Any:
    """Best-effort serialization of browser_use history to a JSON-safe object."""
    # browser_use history objects vary by version; we keep this defensive.
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(history, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return str(history)


def _truncate_text(text: str, *, max_chars: int) -> str:
    """No-op truncation: returns text as-is."""
    return text


def _truncate_jsonish(obj: Any, *, max_chars: int) -> Any:
    """No-op truncation: returns object as-is."""
    return str(obj)


class _SearchStatus:
    OK = "ok"
    BLOCKED = "blocked"


@dataclass
class _SearchBudget:
    """Simple quota + dedupe layer for do_search to prevent repeated Tavily spend."""

    max_calls: int
    max_queries_per_call: int
    calls: int = 0
    seen_queries: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.seen_queries is None:
            self.seen_queries = set()

    def consume(self, queries: list[str]) -> tuple[list[str], str]:
        if self.calls >= self.max_calls:
            return ([], _SearchStatus.BLOCKED)
        accepted: list[str] = []
        for q in queries:
            key = _normalize_query_key(q)
            if not key or key in self.seen_queries:
                continue
            self.seen_queries.add(key)
            accepted.append(q)
            if len(accepted) >= self.max_queries_per_call:
                break
        self.calls += 1
        return (accepted, _SearchStatus.OK)


_WS_RE = re.compile(r"\s+")


def _normalize_query_key(q: str) -> str:
    q2 = _WS_RE.sub(" ", (q or "").strip().lower())
    return q2


def _normalize_search_queries(queries: list[str]) -> list[str]:
    """Normalize and dedupe search queries while preserving original text."""
    out: list[str] = []
    seen: set[str] = set()
    for q in queries or []:
        if not isinstance(q, str):
            continue
        q_stripped = _WS_RE.sub(" ", q.strip())
        if not q_stripped:
            continue
        key = _normalize_query_key(q_stripped)
        if key in seen:
            continue
        seen.add(key)
        out.append(q_stripped)
    return out


def _format_browser_use_history(history: Any) -> str:
    """Extract full running context from browser_use history (v0.11.x).

    This captures all extractions, navigations, and intermediate findings so the
    compile step has the complete picture rather than just the agent's final summary.
    """
    sections: list[str] = []

    # Helper to safely get list attribute
    def _get_list_attr(obj: Any, *names: str) -> list[Any]:
        for name in names:
            try:
                val = getattr(obj, name, None)
                if val is not None:
                    if callable(val):
                        val = val()
                    if isinstance(val, (list, tuple)):
                        return list(val)
            except Exception:
                pass
        return []

    # 1. Extract all URLs visited (browser_use 0.11.x has a .urls property)
    urls_visited: list[str] = []
    try:
        urls_attr = getattr(history, "urls", None)
        if urls_attr:
            if callable(urls_attr):
                urls_attr = urls_attr()
            if isinstance(urls_attr, (list, tuple)):
                urls_visited = [str(u) for u in urls_attr if u]
    except Exception:
        pass
    if urls_visited:
        sections.append("=== URLs Visited ===\n" + "\n".join(f"  - {u}" for u in urls_visited))

    # 2. Extract all content extracted (browser_use 0.11.x has .extracted_content)
    extracted_content: list[str] = []
    try:
        ec_attr = getattr(history, "extracted_content", None)
        if ec_attr:
            if callable(ec_attr):
                ec_attr = ec_attr()
            if isinstance(ec_attr, (list, tuple)):
                extracted_content = [str(e) for e in ec_attr if e and str(e).strip()]
            elif isinstance(ec_attr, str) and ec_attr.strip():
                extracted_content = [ec_attr]
    except Exception:
        pass
    if extracted_content:
        content_section = "=== Extracted Content ===\n"
        for idx, content in enumerate(extracted_content, 1):
            # Bound each extraction
            excerpt = content
            content_section += f"\n--- Extraction {idx} ---\n{excerpt}\n"
        sections.append(content_section)

    # 3. Extract model thoughts/reasoning (browser_use 0.11.x has .model_thoughts)
    model_thoughts: list[Any] = _get_list_attr(history, "model_thoughts")
    if model_thoughts:
        thoughts_section = "=== Agent Reasoning ===\n"
        for idx, thought in enumerate(model_thoughts, 1):
            # Thought could be object or dict
            if hasattr(thought, "evaluation"):
                eval_text = getattr(thought, "evaluation", "") or ""
                memory_text = getattr(thought, "memory", "") or ""
                goal_text = getattr(thought, "next_goal", "") or ""
                thoughts_section += f"\n[Step {idx}]\n"
                if eval_text:
                    thoughts_section += f"  Eval: {eval_text}\n"
                if memory_text:
                    thoughts_section += f"  Memory: {memory_text}\n"
                if goal_text:
                    thoughts_section += f"  Goal: {goal_text}\n"
            elif isinstance(thought, dict):
                thoughts_section += f"\n[Step {idx}] {thought}\n"
            else:
                thoughts_section += f"\n[Step {idx}] {str(thought)}\n"
        sections.append(thoughts_section)

    # 4. Extract actions taken (browser_use 0.11.x has .model_actions)
    model_actions: list[Any] = _get_list_attr(history, "model_actions", "model_actions_filtered")
    if model_actions:
        actions_section = "=== Actions Taken ===\n"
        for idx, action in enumerate(model_actions, 1):
            if hasattr(action, "model_dump"):
                action_dict = action.model_dump()
            elif isinstance(action, dict):
                action_dict = action
            else:
                actions_section += f"  {idx}. {str(action)}\n"
                continue
            # Format action nicely
            for action_name in ["do_search", "go_to_url", "extract_page_content", "done", 
                                "click_element", "input_text", "search_google"]:
                if action_name in action_dict and action_dict[action_name]:
                    action_data = action_dict[action_name]
                    if isinstance(action_data, dict):
                        action_str = ", ".join(f"{k}={v!r}" for k, v in action_data.items() 
                                               if v is not None)
                    else:
                        action_str = str(action_data)
                    actions_section += f"  {idx}. {action_name}: {action_str}\n"
        sections.append(actions_section)

    # 5. Extract action results (browser_use 0.11.x has .action_results)
    action_results: list[Any] = _get_list_attr(history, "action_results")
    if action_results:
        results_section = "=== Action Results ===\n"
        for idx, result in enumerate(action_results, 1):
            if hasattr(result, "extracted_content"):
                extracted = getattr(result, "extracted_content", None)
                is_done = getattr(result, "is_done", False)
                error = getattr(result, "error", None)
                if extracted and str(extracted).strip():
                    excerpt = str(extracted)
                    results_section += f"\n[Result {idx}]\n{excerpt}\n"
                if error:
                    results_section += f"\n[Result {idx} ERROR] {error}\n"
                if is_done:
                    results_section += f"[Result {idx} DONE]\n"
            elif isinstance(result, dict):
                results_section += f"\n[Result {idx}] {result}\n"
        sections.append(results_section)

    # 6. Get the final result/done message if available (properly call if method)
    final_result = None
    try:
        final_result_attr = getattr(history, "final_result", None)
        if callable(final_result_attr):
            final_result = final_result_attr()
        elif final_result_attr is not None:
            final_result = final_result_attr
    except Exception:
        pass

    if final_result:
        text_attr = getattr(final_result, "text", None)
        if isinstance(text_attr, str):
            final_text: str | None = text_attr
        elif isinstance(final_result, dict):
            final_text = final_result.get("text") or str(final_result)
        elif isinstance(final_result, str):
            final_text = final_result
        else:
            final_text = None

        if final_text:
            sections.append(f"=== Final Result ===\n{final_text}")

    # 7. If we got nothing structured, fall back to a cleaned str(history)
    if not sections:
        # Try to extract just the useful parts
        text = str(history)
        # Don't include bound method garbage
        if "<bound method" in text:
            text = "[No structured content extracted from history]"
        return text

    full_context = "\n\n".join(sections)
    # Bound total output
    return full_context


def _extract_evidence_from_history(history: Any) -> list[dict[str, Any]]:
    """Extract URLs/snippets from browser_use 0.11.x history."""
    evidence: list[dict[str, Any]] = []
    url_re = re.compile(r"https?://[^\s\]\)\}>,\"']+")
    seen_urls: set[str] = set()

    # Helper to safely get list attribute
    def _get_list(obj: Any, *names: str) -> list[Any]:
        for name in names:
            try:
                val = getattr(obj, name, None)
                if val is not None:
                    if callable(val):
                        val = val()
                    if isinstance(val, (list, tuple)):
                        return list(val)
            except Exception:
                pass
        return []

    # 1. Get URLs directly from .urls property (browser_use 0.11.x)
    urls_visited = _get_list(history, "urls")
    for url in urls_visited:
        u = str(url).rstrip(".,);]")
        if u.startswith("http") and u not in seen_urls:
            seen_urls.add(u)
            evidence.append({"url": u, "excerpt": None})

    return evidence


