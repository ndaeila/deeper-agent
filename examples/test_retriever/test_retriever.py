"""Test script for the Retriever agent.

Supports any OpenAI-compatible API endpoint via .env.
Configure MODEL_NAME, MODEL_URL, and API_KEY in .env file.

Edit the constants below to configure the agent behavior.

Usage:
    poetry run python examples/test_retriever/test_retriever.py
    poetry run python examples/test_retriever/test_retriever.py --query "Your question here"
"""

import argparse
import os

from dotenv import load_dotenv

from odr.agents import JudgeCounsel, Retriever
from odr.agents.workers import BrowserUseWorkerConfig, BrowserUseWorkerFactory
from odr.agents.retriever.contracts import unique_urls_from_results
from odr.integrations import get_observed_gemini_llm, get_observed_llm, is_observability_enabled

# Configuration - edit these values directly
MAX_ITERATIONS = 10
NUM_JUDGES = 10
COUNSEL_MODEL = None  # None uses JudgeCounsel.DEFAULT_MODEL
WORKER_FACTORY_SEED = 42
EVALUATION_MODE = "best_effort"  # strict|balanced|best_effort

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)


def get_llm():
    """Get LLM configured from environment.

    Uses get_observed_llm which automatically adds Langfuse tracing
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

    Returns:
        Configured ChatOpenAI instance with optional Langfuse observability.
    """
    name = os.getenv("MODEL_NAME", "none")
    url = os.getenv("MODEL_URL", "none")
    api_key = os.getenv("API_KEY", "none")

    print(f"Model: {name}")
    print(f"Endpoint: {url}")
    print(f"Langfuse tracing: {'enabled' if is_observability_enabled() else 'disabled'}\n")

    return get_observed_llm(
        model=name,
        base_url=url,
        api_key=api_key,
        temperature=0,
    )


def get_compile_llm(default_llm):
    """Get the LLM to use for observe_and_compile.

    Defaults to the main LLM, but can be overridden via env for easy swapping.

    Env options:
      - COMPILE_PROVIDER=main|openai_compat|gemini
      - If openai_compat:
          COMPILE_MODEL_NAME / COMPILE_MODEL_URL / COMPILE_API_KEY
      - If gemini:
          GEMINI_MODEL / (GOOGLE_API_KEY or GEMINI_API_KEY)
    """
    provider = (os.getenv("COMPILE_PROVIDER", "main") or "main").strip().lower()
    if provider in {"main", "same", "default"}:
        print("observe_and_compile LLM: (same as main)\n")
        return default_llm

    if provider in {"openai_compat", "openai", "oai"}:
        name = os.getenv("COMPILE_MODEL_NAME", os.getenv("MODEL_NAME", "none"))
        url = os.getenv("COMPILE_MODEL_URL", os.getenv("MODEL_URL", "none"))
        api_key = os.getenv("COMPILE_API_KEY", os.getenv("API_KEY", "none"))
        print(f"observe_and_compile LLM: openai_compat model={name} endpoint={url}\n")
        return get_observed_llm(
            model=name,
            base_url=url,
            api_key=api_key,
            temperature=0,
        )

    if provider in {"gemini", "google"}:
        model = (os.getenv("GEMINI_MODEL") or "gemini-3-flash-preview").strip()
        print(f"observe_and_compile LLM: gemini model={model}\n")
        return get_observed_gemini_llm(model=model, temperature=0)

    print(f"observe_and_compile LLM: unknown COMPILE_PROVIDER={provider!r}; using main\n")
    return default_llm


def get_worker_factories():
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
    return [BrowserUseWorkerFactory(config=browser_use_config)]


def main():
    """Run the Retriever agent test."""
    
    parser = argparse.ArgumentParser(
        description="Test the Retriever agent with any OpenAI-compatible API"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Who is Nathan Daeila?",
        help="Query to run",
    )
    args = parser.parse_args()

    try:
        llm = get_llm()
    except Exception as e:
        print(f"ERROR: {e}")
        return
    compile_llm = get_compile_llm(llm)

    # Create Judge Counsel
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("MODEL_URL")
    
    judge_counsel = JudgeCounsel(
        model=COUNSEL_MODEL,
        num_judges=NUM_JUDGES,
        api_key=api_key,
        base_url=base_url,
        evaluation_mode=EVALUATION_MODE,
    )
    counsel_model = COUNSEL_MODEL or JudgeCounsel.DEFAULT_MODEL
    print(f"Judge Counsel: {NUM_JUDGES} judges using {counsel_model}")

    # Initialize Retriever with Judge Counsel (NEED TO REFACTOR ALL SUB-NODES TO GET IMPORTED FROM HERE)
    retriever = Retriever(
        llm=llm,
        compile_llm=compile_llm,
        max_iterations=MAX_ITERATIONS,
        judge_counsel=judge_counsel,
        worker_factories=get_worker_factories(),
        worker_factory_seed=WORKER_FACTORY_SEED,
    )

    print(f"Query: {args.query}\n")
    print("=" * 60)
    print("Streaming execution (showing each step):\n")

    running_worker_results = []

    # Stream execution to see each step
    for step in retriever.stream(args.query):
        for node_name, node_state in step.items():
            print(f"→ Executed node: {node_name}")

            if not isinstance(node_state, dict):
                continue

            if node_name == "worker":
                results_patch = node_state.get("worker_results") or []
                if isinstance(results_patch, list) and results_patch:
                    running_worker_results.extend(results_patch)

            if node_name == "choose_workers":
                tasks = node_state.get("worker_tasks") or []
                if tasks:
                    print("  Planned tasks:")
                    for t in tasks:
                        wt = t.get("worker_type", "unknown")
                        desc = t.get("task_description", "")
                        print(f"  - [{wt}] {desc}")

            if node_name == "observe_and_compile":
                urls = sorted(unique_urls_from_results(running_worker_results))
                print(f"  Unique URLs so far: {len(urls)}")

            if node_name == "judgment":
                fb = node_state.get("judge_feedback")
                if fb:
                    print("  Judge feedback (summary):")
                    print("  " + str(fb).replace("\n", "\n  "))

            if node_name == "decide_next_action":
                action = node_state.get("next_action")
                final_status = node_state.get("final_status")
                print(f"  Next action: {action}")
                if final_status:
                    print(f"  Final status: {final_status}")

    print("\n" + "=" * 60)
    print("\nFinal Result:\n")

    # Run directly to get final result
    result = retriever.run(args.query)

    print(f"Compiled Output:\n{result['compiled_output']}\n")
    print(f"Iterations: {result['iteration_count']}")
    print(f"Judgment: {result['judgment_decision']}")
    print(f"Final status: {result.get('final_status')}")
    print(f"Next action: {result.get('next_action')}")
    print(f"Workers executed: {len(result['worker_results'])}")


if __name__ == "__main__":
    main()
