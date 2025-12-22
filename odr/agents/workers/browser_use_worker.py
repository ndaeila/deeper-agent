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
    max_steps: int = 5
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
            f"Hard constraint: max {self.config.max_steps} browser steps.\n"
            "Do NOT navigate directly to URLs you haven't discovered via search.\n"
            "You MUST cite the exact URL for each fact. If uncertain, say so."
        )

        # Enforce step budget when supported by the installed browser_use version.
        agent_kwargs: dict[str, Any] = {"task": agent_task, "llm": llm, "browser": browser}
        if tools is not None:
            agent_kwargs["tools"] = tools
        try:
            sig = inspect.signature(Agent)  # some versions expose Agent as a class/callable
            if "max_steps" in sig.parameters:
                agent_kwargs["max_steps"] = self.config.max_steps
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
    """Extract full running context from browser_use history, not just final result.

    This captures all extractions, navigations, and intermediate findings so the
    compile step has the complete picture rather than just the agent's final summary.
    """
    sections: list[str] = []

    # 1. Collect all step-by-step actions and extractions
    all_results = getattr(history, "all_results", None) or []
    if isinstance(all_results, list):
        for idx, item in enumerate(all_results, start=1):
            step_lines: list[str] = [f"--- Step {idx} ---"]

            # Get URL for this step
            if isinstance(item, dict):
                page_url = item.get("url") or item.get("page_url") or item.get("current_url")
                extracted = item.get("extracted_content")
                action = item.get("action") or item.get("action_name")
            else:
                page_url = (
                    getattr(item, "url", None)
                    or getattr(item, "page_url", None)
                    or getattr(item, "current_url", None)
                )
                extracted = getattr(item, "extracted_content", None)
                action = getattr(item, "action", None) or getattr(item, "action_name", None)

            if page_url:
                step_lines.append(f"URL: {page_url}")
            if action:
                step_lines.append(f"Action: {action}")
            if isinstance(extracted, str) and extracted.strip():
                # Include full extraction (bounded per step)
                excerpt = extracted[:2000] if len(extracted) > 2000 else extracted
                step_lines.append(f"Extracted:\n{excerpt}")

            if len(step_lines) > 1:  # Has more than just the header
                sections.append("\n".join(step_lines))

    # 2. Get the final result/done message if available
    final_result = getattr(history, "final_result", None) or getattr(history, "result", None)
    if final_result:
        if hasattr(final_result, "text"):
            final_text = final_result.text
        elif isinstance(final_result, dict):
            final_text = final_result.get("text") or str(final_result)
        else:
            final_text = str(final_result)
        sections.append(f"--- Final Result ---\n{final_text}")

    # 3. If we got nothing structured, fall back to str(history)
    if not sections:
        text = str(history)
        if len(text) > 8000:
            text = text[:8000] + "\n\n[truncated]"
        return text

    full_context = "\n\n".join(sections)
    # Bound total output
    if len(full_context) > 100000:
        full_context = full_context[:100000] + "\n\n[truncated]"
    return full_context


def _extract_evidence_from_history(history: Any) -> list[dict[str, Any]]:
    """Extract URLs/snippets from history when possible."""
    evidence: list[dict[str, Any]] = []
    url_re = re.compile(r"https?://[^\s\]\)\}>,\"']+")
    seen_urls: set[str] = set()

    all_results = getattr(history, "all_results", None)
    if isinstance(all_results, list):
        for item in all_results:
            item_url = None
            if isinstance(item, dict):
                item_url = item.get("url") or item.get("page_url") or item.get("source_url")
            else:
                item_url = (
                    getattr(item, "url", None)
                    or getattr(item, "page_url", None)
                    or getattr(item, "source_url", None)
                )
            extracted = (
                item.get("extracted_content")
                if isinstance(item, dict)
                else getattr(item, "extracted_content", None)
            )
            excerpt = extracted[:500] if isinstance(extracted, str) else None

            if isinstance(item_url, str) and item_url.startswith("http"):
                u = item_url.rstrip(".,);]")
                if u not in seen_urls:
                    seen_urls.add(u)
                    evidence.append({"url": u, "excerpt": excerpt})

            if isinstance(extracted, str) and extracted.strip():
                urls = [m.group(0).rstrip(".,);]") for m in url_re.finditer(extracted)]
                for url in urls[:5]:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        evidence.append({"url": url, "excerpt": excerpt})
                if not urls and "http" in extracted:
                    # Keep a trace even if regex fails; normalizer may recover URLs from findings.
                    evidence.append({"excerpt": excerpt})

    return evidence[:50]


