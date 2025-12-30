"""Simple demo of the browser_use worker.

Run with:
    cd examples/test_browser_use
    cp .env-example .env
    # Edit .env with your API keys
    poetry run python test_browser_use.py

Requirements:
    - OPENAI_API_KEY set in .env (or environment)
    - browser_use installed: poetry add selenium browser-use langchain-openai
    - Drivers installed: uvx browser-use install

Optional (for search tool):
    - TAVILY_API_KEY or SERPAPI_API_KEY set in .env
    - poetry add tavily-python  (or google-search-results for SerpApi)
"""

from odr.agents.workers.browser_use_worker import (
    BrowserUseWorker,
    BrowserUseWorkerConfig,
)
from odr.agents.types import WorkerTask


def main():
    # Basic config - search tool disabled, visible browser
    config = BrowserUseWorkerConfig(
        model="gpt-5-nano",
        headless=False,  # Show the browser
        max_steps=10,
        use_search_tool=False,  # No Tavily/SerpApi needed
        log_level="ERROR",
    )

    worker = BrowserUseWorker(worker_id="demo", config=config)

    task = WorkerTask(
        worker_id="demo",
        task_description="Go to example.com and describe what you see.",
        context="Simple test task.",
    )

    print("Running browser_use worker...")
    result = worker.run(task=task, input_text="Test query", iteration=1)

    print(f"\nSuccess: {result['success']}")
    print(f"Findings:\n{result['findings'][:1000]}")


if __name__ == "__main__":
    main()

