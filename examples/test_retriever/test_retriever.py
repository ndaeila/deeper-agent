"""Test script for the Retriever agent.

Supports any OpenAI-compatible API endpoint via .env or CLI arguments.
Configure MODEL_NAME, MODEL_URL, and API_KEY in .env file.

Usage:
    poetry run python examples/test_retriever/test_retriever.py
    poetry run python examples/test_retriever/test_retriever.py --query "Your question here"
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from odr.agents import Retriever
from odr.integrations import get_observed_llm, is_observability_enabled

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)


def get_llm(
    model_name: str | None = None, 
    model_url: str | None = None
):
    """Get LLM configured from environment or arguments.

    Uses get_observed_llm which automatically adds Langfuse tracing
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

    Args:
        model_name: Model name (overrides MODEL_NAME from .env).
        model_url: Model endpoint URL (overrides MODEL_URL from .env).

    Returns:
        Configured ChatOpenAI instance with optional Langfuse observability.
    """
    # Get config from environment or use provided args
    name = model_name or os.getenv("MODEL_NAME", "none")
    url = model_url or os.getenv("MODEL_URL", "none")
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
        "--model-name",
        type=str,
        default=None,
        help="Model name (overrides MODEL_NAME from .env)",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default=None,
        help="Model endpoint URL (overrides MODEL_URL from .env)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is OSINT and how is it used in cybersecurity?",
        help="Query to run",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum retry iterations (default: 10)",
    )
    args = parser.parse_args()

    try:
        llm = get_llm(model_name=args.model_name, model_url=args.model_url)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Initialize Retriever
    retriever = Retriever(llm=llm, max_iterations=args.max_iterations)

    print(f"Query: {args.query}\n")
    print("=" * 60)
    print("Streaming execution (showing each step):\n")

    # Stream execution to see each step
    for step in retriever.stream(args.query):
        node_names = list(step.keys())
        if node_names:
            print(f"→ Executed node: {node_names[0]}")

    print("\n" + "=" * 60)
    print("\nFinal Result:\n")

    # Run directly to get final result
    result = retriever.run(args.query)

    print(f"Compiled Output:\n{result['compiled_output']}\n")
    print(f"Iterations: {result['iteration_count']}")
    print(f"Judgment: {result['judgment_decision']}")
    print(f"Workers executed: {len(result['worker_results'])}")


if __name__ == "__main__":
    main()
