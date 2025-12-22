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
from odr.agents.workers import BrowserUseWorkerFactory
from odr.agents.retriever.contracts import unique_urls_from_results
from odr.integrations import get_observed_llm, is_observability_enabled

# Configuration - edit these values directly
MAX_ITERATIONS = 10
NUM_JUDGES = 10
COUNSEL_MODEL = None  # None uses JudgeCounsel.DEFAULT_MODEL
WORKER_FACTORY_SEED = 42

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


def main():
    """Run the Retriever agent test."""
    
    parser = argparse.ArgumentParser(
        description="Test the Retriever agent with any OpenAI-compatible API"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is OSINT and how is it used in cybersecurity?",
        help="Query to run",
    )
    args = parser.parse_args()

    try:
        llm = get_llm()
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Create Judge Counsel
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("MODEL_URL")
    
    judge_counsel = JudgeCounsel(
        model=COUNSEL_MODEL,
        num_judges=NUM_JUDGES,
        api_key=api_key,
        base_url=base_url,
    )
    counsel_model = COUNSEL_MODEL or JudgeCounsel.DEFAULT_MODEL
    print(f"Judge Counsel: {NUM_JUDGES} judges using {counsel_model}")

    # Initialize Retriever with Judge Counsel
    retriever = Retriever(
        llm=llm,
        max_iterations=MAX_ITERATIONS,
        judge_counsel=judge_counsel,
        worker_factories=[BrowserUseWorkerFactory()],
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
