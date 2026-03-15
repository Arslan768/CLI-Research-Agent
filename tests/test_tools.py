"""
tests/test_tools.py

Unit tests for individual tools. We mock external APIs so tests
run offline and never consume real API credits.

Run with:  pytest tests/test_tools.py -v
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─── web_search ────────────────────────────────────────────────────────────────

class TestWebSearch:
    def test_returns_formatted_results(self):
        from agent.tools.web_search import web_search

        mock_response = {
            "answer": "Test quick answer",
            "results": [
                {
                    "title": "Test Article",
                    "url": "https://example.com/test",
                    "content": "This is test content about the topic.",
                }
            ],
        }

        with patch("agent.tools.web_search.TavilyClient") as MockClient:
            MockClient.return_value.search.return_value = mock_response
            with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
                result = web_search("test query")

        assert "Test quick answer" in result
        assert "Test Article" in result
        assert "https://example.com/test" in result
        assert "test content" in result

    def test_returns_error_when_no_api_key(self):
        from agent.tools.web_search import web_search

        with patch.dict("os.environ", {}, clear=True):
            # Ensure TAVILY_API_KEY is absent
            import os
            os.environ.pop("TAVILY_API_KEY", None)
            result = web_search("test query")

        assert "Error" in result
        assert "TAVILY_API_KEY" in result

    def test_handles_empty_results(self):
        from agent.tools.web_search import web_search

        with patch("agent.tools.web_search.TavilyClient") as MockClient:
            MockClient.return_value.search.return_value = {"answer": None, "results": []}
            with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
                result = web_search("obscure query with no results")

        assert "No results" in result

    def test_handles_api_exception_gracefully(self):
        from agent.tools.web_search import web_search

        with patch("agent.tools.web_search.TavilyClient") as MockClient:
            MockClient.return_value.search.side_effect = Exception("Connection refused")
            with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
                result = web_search("test query")

        # Must return a string, not raise
        assert isinstance(result, str)
        assert "error" in result.lower()


# ─── wikipedia_search ──────────────────────────────────────────────────────────

class TestWikipediaSearch:
    def test_returns_article_content(self):
        from agent.tools.wikipedia import wikipedia_search

        mock_page = MagicMock()
        mock_page.title = "Python (programming language)"
        mock_page.url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

        with patch("agent.tools.wikipedia.wikipedia.search", return_value=["Python (programming language)"]):
            with patch("agent.tools.wikipedia.wikipedia.page", return_value=mock_page):
                with patch("agent.tools.wikipedia.wikipedia.summary", return_value="Python is a high-level programming language."):
                    result = wikipedia_search("Python programming")

        assert "Python" in result
        assert "wikipedia.org" in result
        assert "high-level" in result

    def test_handles_no_results(self):
        from agent.tools.wikipedia import wikipedia_search

        with patch("agent.tools.wikipedia.wikipedia.search", return_value=[]):
            result = wikipedia_search("xyzzy_nonexistent_topic_12345")

        assert "No Wikipedia" in result

    def test_handles_disambiguation(self):
        import wikipedia as wiki_module
        from agent.tools.wikipedia import wikipedia_search

        mock_page = MagicMock()
        mock_page.title = "Mercury (planet)"
        mock_page.url = "https://en.wikipedia.org/wiki/Mercury_(planet)"

        disambiguation_error = wiki_module.DisambiguationError("Mercury", ["Mercury (planet)", "Mercury (element)"])

        with patch("agent.tools.wikipedia.wikipedia.search", return_value=["Mercury"]):
            with patch("agent.tools.wikipedia.wikipedia.page", side_effect=[disambiguation_error, mock_page]):
                with patch("agent.tools.wikipedia.wikipedia.summary", return_value="Mercury is the smallest planet."):
                    result = wikipedia_search("Mercury")

        # Should recover and return something useful
        assert isinstance(result, str)

    def test_handles_exception_gracefully(self):
        from agent.tools.wikipedia import wikipedia_search

        with patch("agent.tools.wikipedia.wikipedia.search", side_effect=Exception("Network error")):
            result = wikipedia_search("test topic")

        assert isinstance(result, str)
        assert "error" in result.lower()


# ─── url_reader ────────────────────────────────────────────────────────────────

class TestUrlReader:
    def test_extracts_text_from_html(self):
        from agent.tools.url_reader import read_url

        fake_html = """
        <html><body>
          <article>
            <h1>Test Article</h1>
            <p>This is the main content of the article.</p>
            <p>It has multiple paragraphs.</p>
          </article>
          <script>alert('this should be removed')</script>
          <nav>Nav links should be removed</nav>
        </body></html>
        """

        mock_response = MagicMock()
        mock_response.text = fake_html
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.raise_for_status = MagicMock()

        with patch("agent.tools.url_reader.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock_response
            result = read_url("https://example.com/article")

        assert "Test Article" in result
        assert "main content" in result
        assert "alert" not in result  # script tag removed
        assert "Nav links" not in result  # nav tag removed

    def test_handles_http_error(self):
        import httpx
        from agent.tools.url_reader import read_url

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("agent.tools.url_reader.httpx.Client") as MockClient:
            error = httpx.HTTPStatusError("Not Found", request=MagicMock(), response=mock_response)
            MockClient.return_value.__enter__.return_value.get.side_effect = error
            result = read_url("https://example.com/missing-page")

        assert "404" in result
        assert isinstance(result, str)

    def test_handles_timeout(self):
        import httpx
        from agent.tools.url_reader import read_url

        with patch("agent.tools.url_reader.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.side_effect = (
                httpx.TimeoutException("Timeout")
            )
            result = read_url("https://slow-site.example.com")

        assert "Timeout" in result or "timeout" in result.lower()

    def test_truncates_long_content(self):
        from agent.tools.url_reader import read_url, MAX_CHARS

        long_content = "word " * 10000  # way more than MAX_CHARS
        fake_html = f"<html><body><article>{long_content}</article></body></html>"

        mock_response = MagicMock()
        mock_response.text = fake_html
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        with patch("agent.tools.url_reader.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock_response
            result = read_url("https://example.com/long-page")

        assert len(result) < len(long_content)
        assert "Truncated" in result


# ─── registry ─────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_get_tools_returns_list(self):
        from agent.tools.registry import get_tools
        tools = get_tools()
        assert isinstance(tools, list)
        assert len(tools) == 3  # web_search, wikipedia_search, read_url

    def test_each_tool_has_required_fields(self):
        from agent.tools.registry import get_tools
        for tool in get_tools():
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert "properties" in tool["input_schema"]

    def test_execute_known_tool(self):
        from agent.tools.registry import execute_tool

        with patch("agent.tools.registry.web_search", return_value="mocked result") as mock:
            result = execute_tool("web_search", {"query": "test"})

        assert result == "mocked result"
        mock.assert_called_once_with(query="test")

    def test_execute_unknown_tool_returns_string(self):
        from agent.tools.registry import execute_tool
        result = execute_tool("nonexistent_tool", {})
        assert isinstance(result, str)
        assert "Unknown tool" in result

    def test_execute_tool_with_wrong_args_returns_string(self):
        from agent.tools.registry import execute_tool
        # wikipedia_search requires 'topic', not 'query'
        result = execute_tool("wikipedia_search", {"wrong_arg": "test"})
        assert isinstance(result, str)
        # Should return an error string, not raise
