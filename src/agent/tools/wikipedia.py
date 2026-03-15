"""
wikipedia.py

Wraps the `wikipedia` Python library. Good for background
knowledge on well-established topics — faster and more
reliable than a general web search for encyclopedic content.
"""

import wikipedia


def wikipedia_search(topic: str, sentences: int = 10) -> str:
    """
    Search Wikipedia and return a summary of the most relevant article.

    Args:
        topic: The topic to look up
        sentences: How many sentences of the summary to return

    Returns:
        A string with the article title, URL, and summary.
        Returns an error/disambiguation message on failure (never raises).
    """
    try:
        # search() returns a list of matching article titles
        results = wikipedia.search(topic, results=3)
        if not results:
            return f"No Wikipedia articles found for: {topic}"

        # Try the top result first, fall back to others on disambiguation
        for title in results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                summary = wikipedia.summary(
                    title,
                    sentences=sentences,
                    auto_suggest=False,
                )
                return (
                    f"Wikipedia: {page.title}\n"
                    f"URL: {page.url}\n\n"
                    f"{summary}"
                )
            except wikipedia.DisambiguationError as e:
                # The title matched a disambiguation page — try the first suggestion
                try:
                    suggestion = e.options[0]
                    page = wikipedia.page(suggestion, auto_suggest=False)
                    summary = wikipedia.summary(suggestion, sentences=sentences, auto_suggest=False)
                    return (
                        f"Wikipedia: {page.title}\n"
                        f"URL: {page.url}\n\n"
                        f"{summary}"
                    )
                except Exception:
                    continue
            except wikipedia.PageError:
                continue

        return f"Could not retrieve a Wikipedia article for: {topic}"

    except Exception as e:
        return f"Wikipedia error: {type(e).__name__}: {str(e)}"
