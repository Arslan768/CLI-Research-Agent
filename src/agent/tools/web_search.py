"""
web_search.py

Wraps the Tavily search API. Tavily is designed for agents —
it returns clean extracted text rather than raw HTML, which
saves us from writing a custom scraper for every site.

Get a free API key at: https://tavily.com
"""

import os
from tavily import TavilyClient


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily and return formatted results.

    Args:
        query: The search query string
        max_results: How many results to return (default 5)

    Returns:
        A formatted string with titles, URLs, and content snippets.
        Returns an error message string on failure (never raises).
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not set in environment."

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",  # pays ~2x credits but gets better snippets
            include_answer=True,       # Tavily's own AI summary of the results
        )

        parts = []

        # Tavily's quick-answer summary (useful as a first orientation)
        if response.get("answer"):
            parts.append(f"Quick summary: {response['answer']}\n")

        # Individual results
        for i, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "").strip()
            parts.append(f"[{i}] {title}\nURL: {url}\n{content}")

        if not parts:
            return f"No results found for query: {query}"

        return "\n\n".join(parts)

    except Exception as e:
        return f"Search error: {type(e).__name__}: {str(e)}"
