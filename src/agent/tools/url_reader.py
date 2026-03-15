"""
url_reader.py

Fetches a URL and extracts clean readable text using
BeautifulSoup. Useful when the agent finds a promising
URL in search results and wants the full article.
"""

import httpx
from bs4 import BeautifulSoup


# Tags whose content is never useful
SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript"}

# Cap how much text we return — long pages would overflow the context window
MAX_CHARS = 8000


def read_url(url: str) -> str:
    """
    Fetch a URL and return its main text content.

    Args:
        url: The full URL to fetch (must include https://)

    Returns:
        Extracted text from the page, truncated to MAX_CHARS.
        Returns an error message string on failure (never raises).
    """
    try:
        headers = {
            # Pretend to be a regular browser — some sites block bot user-agents
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        with httpx.Client(follow_redirects=True, timeout=15.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

        # Only attempt to parse HTML/text content
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return f"Cannot parse content type: {content_type}"

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove tags we don't want
        for tag in soup(SKIP_TAGS):
            tag.decompose()

        # Try to find the main article body first
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(id="content")
            or soup.find(id="main")
            or soup.find(class_="content")
            or soup.body
        )

        if main is None:
            return "Could not extract content from page."

        # get_text() with separator gives us word-boundary-safe extraction
        text = main.get_text(separator="\n", strip=True)

        # Collapse runs of blank lines
        lines = [line for line in text.splitlines() if line.strip()]
        clean = "\n".join(lines)

        if not clean:
            return f"Page at {url} appears to have no readable text content."

        truncated = clean[:MAX_CHARS]
        suffix = f"\n\n[Truncated at {MAX_CHARS} chars]" if len(clean) > MAX_CHARS else ""

        return f"Content from {url}:\n\n{truncated}{suffix}"

    except httpx.HTTPStatusError as e:
        return f"HTTP {e.response.status_code} error fetching {url}"
    except httpx.TimeoutException:
        return f"Timeout fetching {url} (>15s)"
    except Exception as e:
        return f"Error fetching {url}: {type(e).__name__}: {str(e)}"
