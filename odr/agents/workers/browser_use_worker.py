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
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Mapping, cast

from odr.agents.types import WorkerResultExtended, WorkerTask
from odr.agents.workers.base import WorkerFactory
from odr.tools.search import search_for_links, search_provider_available


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
        from browser_use.llm import ChatOpenAI as BrowserUseChatOpenAI  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(_INSTALL_HINT) from e

    return Agent, Browser, Tools, BrowserUseChatOpenAI


@dataclass(frozen=True)
class BrowserUseWorkerConfig:
    """Configuration for the browser_use worker."""

    model: str = "gpt-5-nano-2025-08-07"
    reasoning_effort: str = "low"
    headless: bool = False
    tavily_max_results: int = 5
    max_steps: int = 15  # Hard limit on browser agent steps
    max_actions: int = 15  # Some browser_use versions use this instead
    use_search_tool: bool = True  # Set False to disable Tavily/SerpApi search tool


class BrowserUseWorker:
    """A worker that uses Tavily + browser_use for evidence collection."""

    worker_type = "browser_use"

    def __init__(self, worker_id: str, config: BrowserUseWorkerConfig | None = None):
        self.worker_id = worker_id
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
        Agent, Browser, Tools, BrowserUseChatOpenAI = _require_browser_use()

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY (or API_KEY) for browser_use ChatOpenAI")

        tools: Any = None
        if self.config.use_search_tool:
            tools = Tools()

            @tools.action(
                description="Initial web search for relevant links using multiple queries at once."
            )
            async def do_search(queries: list[str]) -> str:
                return search_for_links(queries=queries, max_results=self.config.tavily_max_results)

        browser = Browser(headless=self.config.headless)
        llm_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "api_key": api_key,
        }
        # Some browser_use versions support reasoning_effort; keep best-effort compatibility.
        if self.config.reasoning_effort:
            llm_kwargs["reasoning_effort"] = self.config.reasoning_effort
        llm = BrowserUseChatOpenAI(**llm_kwargs)

        # Build a short task description without URLs to avoid browser_use auto-navigating
        # (it parses URLs from the task string and skips search).
        base_q = task["task_description"]

        # Extract deliverables from context if present (simple heuristic).
        deliverables_section = ""
        ctx = task.get("context") or ""
        if "Deliverables:" in ctx:
            start = ctx.find("Deliverables:")
            end = ctx.find("\n\n", start)
            if end == -1:
                end = len(ctx)
            deliverables_section = ctx[start:end].strip()

        agent_task = (
            f"Task: {base_q}\n\n"
            f"Original query: {input_text}\n\n"
            + (f"{deliverables_section}\n\n" if deliverables_section else "")
            + "WORKFLOW (follow this order):\n"
            f"1. FIRST use `do_search` with 2-3 queries to find relevant sources.\n"
            f"2. Open 1-2 of the best result URLs in the browser.\n"
            f"3. Extract the specific facts/quotes you need with their source URLs.\n"
            f"4. Stop and return findings with cited URLs.\n\n"
            f"HARD CONSTRAINTS:\n"
            f"- Max {self.config.max_steps} browser steps total.\n"
            f"- If search quota is exhausted or blocked, STOP IMMEDIATELY with whatever you have.\n"
            f"- Do NOT wait or retry indefinitely. Return partial results if needed.\n"
            f"- Do NOT navigate directly to URLs you haven't discovered via search.\n"
            f"- You MUST cite the exact URL for each fact. If uncertain, say so.\n\n"
            f"If blocked/quota exhausted: call done() with success=True and your best partial findings."
        )

        # Enforce step budget when supported by the installed browser_use version.
        agent_kwargs: dict[str, Any] = {"task": agent_task, "llm": llm, "browser": browser}
        if tools is not None:
            agent_kwargs["tools"] = tools
        try:
            sig = inspect.signature(Agent)  # some versions expose Agent as a class/callable
            params = sig.parameters
            # Try both parameter names (browser_use versions vary)
            if "max_steps" in params:
                agent_kwargs["max_steps"] = self.config.max_steps
            if "max_actions" in params:
                agent_kwargs["max_actions"] = self.config.max_actions
        except (TypeError, ValueError):
            # Best-effort; prompt-level constraint still applies.
            pass

        agent = Agent(**agent_kwargs)
        history = await agent.run()

        formatted = _format_browser_use_history(history)
        evidence = _extract_evidence_from_history(history)

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
            raw_history=_safe_serialize_history(history),
        )


class BrowserUseWorkerFactory(WorkerFactory):
    """Factory for creating BrowserUseWorker instances."""

    worker_type = "browser_use"

    def __init__(self, config: BrowserUseWorkerConfig | None = None):
        self.config = config or BrowserUseWorkerConfig()
        _preflight_print_hints(need_search=self.config.use_search_tool)

    def create(self, worker_id: str) -> BrowserUseWorker:
        return BrowserUseWorker(worker_id=worker_id, config=self.config)


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
            excerpt = content[:3000] if len(content) > 3000 else content
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
                    thoughts_section += f"  Eval: {eval_text[:500]}\n"
                if memory_text:
                    thoughts_section += f"  Memory: {memory_text[:500]}\n"
                if goal_text:
                    thoughts_section += f"  Goal: {goal_text[:300]}\n"
            elif isinstance(thought, dict):
                thoughts_section += f"\n[Step {idx}] {thought}\n"
            else:
                thoughts_section += f"\n[Step {idx}] {str(thought)[:500]}\n"
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
                actions_section += f"  {idx}. {str(action)[:200]}\n"
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
                        action_str = str(action_data)[:200]
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
                    excerpt = str(extracted)[:2000]
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
        if len(text) > 8000:
            text = text[:8000] + "\n\n[truncated]"
        return text

    full_context = "\n\n".join(sections)
    # Bound total output
    if len(full_context) > 100000:
        full_context = full_context[:100000] + "\n\n[truncated]"
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

    # 2. Get extracted content from .extracted_content property
    try:
        ec_attr = getattr(history, "extracted_content", None)
        if ec_attr:
            if callable(ec_attr):
                ec_attr = ec_attr()
            if isinstance(ec_attr, (list, tuple)):
                for content in ec_attr:
                    if isinstance(content, str) and content.strip():
                        # Extract URLs from content
                        found_urls = [m.group(0).rstrip(".,);]") for m in url_re.finditer(content)]
                        excerpt = content[:500]
                        for url in found_urls[:5]:
                            if url not in seen_urls:
                                seen_urls.add(url)
                                evidence.append({"url": url, "excerpt": excerpt})
    except Exception:
        pass

    # 3. Get from action_results
    action_results = _get_list(history, "action_results")
    for result in action_results:
        extracted = getattr(result, "extracted_content", None)
        if isinstance(extracted, str) and extracted.strip():
            excerpt = extracted[:500]
            found_urls = [m.group(0).rstrip(".,);]") for m in url_re.finditer(extracted)]
            for url in found_urls[:5]:
                if url not in seen_urls:
                    seen_urls.add(url)
                    evidence.append({"url": url, "excerpt": excerpt})

    return evidence[:50]


