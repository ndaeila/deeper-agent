"""Web search utilities using Tavily or SerpApi.

This module provides a unified interface for web search across different providers.
Tavily is preferred when available; SerpApi is used as a fallback.

Environment variables:
    TAVILY-PYTHON-RESEARCH-API-KEY or TAVILY_API_KEY: Tavily API key
    SERPAPI_API_KEY: SerpApi API key (fallback)
"""

from __future__ import annotations

import os


def search_provider_available() -> bool:
    """Check if at least one search provider is configured."""
    return bool(
        os.getenv("TAVILY-PYTHON-RESEARCH-API-KEY")
        or os.getenv("TAVILY_API_KEY")
        or os.getenv("SERPAPI_API_KEY")
    )


def search_for_links(queries: list[str], max_results: int = 5) -> str:
    """Search for links using Tavily (preferred) or SerpApi fallback.

    Args:
        queries: List of search queries to run.
        max_results: Maximum results per query (Tavily only).

    Returns:
        Stringified list of search results.

    Raises:
        ImportError: If the required search client is not installed.
        RuntimeError: If no search provider credentials are configured.
    """
    tavily_key = os.getenv("TAVILY-PYTHON-RESEARCH-API-KEY") or os.getenv("TAVILY_API_KEY")
    serpapi_key = os.getenv("SERPAPI_API_KEY")

    if tavily_key:
        from tavily import TavilyClient  # type: ignore[import-not-found]

        client = TavilyClient(tavily_key)
        responses: list[str] = []
        for query in queries:
            response = client.search(query=query, max_results=max_results)
            responses.append(str(response))
        return str(responses)

    if serpapi_key:
        from serpapi import GoogleSearch  # type: ignore[import-not-found]

        responses = []
        for query in queries:
            params = {
                "q": query,
                "hl": "en",
                "gl": "us",
                "google_domain": "google.com",
                "api_key": serpapi_key,
            }
            search = GoogleSearch(params)
            responses.append(str(search.get_dict()))
        return str(responses)

    raise RuntimeError(
        "Missing search provider credentials. Set TAVILY-PYTHON-RESEARCH-API-KEY "
        "(or TAVILY_API_KEY) or SERPAPI_API_KEY."
    )

