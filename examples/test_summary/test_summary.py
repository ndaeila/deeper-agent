"""Test script for Planner -> Retriever loop -> ResearchPaperWriter (full pipeline).

Supports any OpenAI-compatible API endpoint via .env.
Configure MODEL_NAME, MODEL_URL, and API_KEY in .env file.

Edit the constants below to configure the agent behavior.

Usage:
    poetry run python examples/test_summary/test_summary.py
    poetry run python examples/test_summary/test_summary.py --query "Your question here"
"""

import argparse
import os

from dotenv import load_dotenv

from odr.agents import DeepResearchPaper, JudgeCounsel, Retriever
from odr.agents.workers import BrowserUseWorkerConfig, BrowserUseWorkerFactory
from odr.factory import DefaultLLMFactory
from odr.integrations import is_observability_enabled

# Configuration - edit these values directly
MAX_ITERATIONS = 10
NUM_JUDGES = 10
COUNSEL_MODEL = None  # None uses JudgeCounsel.DEFAULT_MODEL
WORKER_FACTORY_SEED = 42
EVALUATION_MODE = "best_effort"  # strict|balanced|best_effort

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)


def get_worker_factories(llm_factory):
    """Create worker factories configured from environment (easy local swapping).

    Browser-use worker env options:
      - BROWSER_USE_MODEL (default: o3)
      - BROWSER_USE_REASONING_EFFORT (default: medium)
      - BROWSER_USE_HEADLESS (true/false, default: false)
    """
    model = (os.getenv("BROWSER_USE_MODEL") or "o3").strip()
    reasoning_effort = (os.getenv("BROWSER_USE_REASONING_EFFORT") or "medium").strip()
    headless = (os.getenv("BROWSER_USE_HEADLESS") or "false").strip().lower() in {"1", "true", "yes"}

    print(
        "browser_use worker: "
        f"model={model} reasoning_effort={reasoning_effort} headless={headless}\n"
    )

    browser_use_config = BrowserUseWorkerConfig(
        model=model,
        reasoning_effort=reasoning_effort,
        headless=headless,
        use_search_tool=True,
        log_level="ERROR",
    )
    return [BrowserUseWorkerFactory(llm_factory=llm_factory, config=browser_use_config)]


def main():
    """Run Retriever, then generate a markdown "research paper" summary after success."""
    parser = argparse.ArgumentParser(
        description="Run Retriever then generate a post-summary research paper (Markdown)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Who is Nathan Daeila?",
        help="Query to run",
    )
    args = parser.parse_args()

    # Initialize Factory
    print(f"Langfuse tracing: {'enabled' if is_observability_enabled() else 'disabled'}\n")
    llm_factory = DefaultLLMFactory(prefer_gemini=True)

    # Create Judge Counsel (Factory-aware)
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("MODEL_URL")
    
    judge_counsel = JudgeCounsel(
        llm_factory=llm_factory,
        model=COUNSEL_MODEL,
        num_judges=NUM_JUDGES,
        # api_key=api_key, # Deprecated in favor of factory
        # base_url=base_url,
        evaluation_mode=EVALUATION_MODE,
    )
    counsel_model = COUNSEL_MODEL or JudgeCounsel.DEFAULT_MODEL
    print(f"Judge Counsel: {NUM_JUDGES} judges using {counsel_model}")

    # Initialize Retriever (Factory-aware)
    # Note: Retriever will use the factory to create its supervisor/compiler LLMs internally
    retriever = Retriever(
        llm_factory=llm_factory,
        max_iterations=MAX_ITERATIONS,
        judge_counsel=judge_counsel,
        worker_factories=get_worker_factories(llm_factory),
        worker_factory_seed=WORKER_FACTORY_SEED,
    )

    # Initialize Pipeline (Factory-aware)
    # DeepResearchPaper will use the factory to create Planner, Writer, and Refiner
    pipeline = DeepResearchPaper(
        llm_factory=llm_factory,
        retriever=retriever
    )

    print(f"Query: {args.query}\n")
    print("=" * 60)
    print("Running Planner -> Retriever loop -> ResearchPaperWriter...\n")

    out = pipeline.run(args.query)
    outline = out.get("outline") or {}
    packets = out.get("section_packets") or []

    print("Planned outline sections:")
    for s in (outline.get("sections") or []):
        print(f"  - {s.get('title')}")

    print("\nSection retrieval statuses:")
    for p in packets:
        sec = (p.get("section") or {}).get("title")
        status = p.get("retriever_final_status")
        urls = p.get("unique_urls") or []
        print(f"  - {sec}: status={status} urls={len(urls)}")

    print("\n" + "=" * 60)
    print("\nResearch Paper (Markdown):\n")
    print(out.get("paper_markdown") or "")


if __name__ == "__main__":
    main()
