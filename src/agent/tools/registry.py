"""
registry.py

Central registry of all tools. Two responsibilities:
1. get_tools()    → returns tool schemas in the format Anthropic's API expects
2. execute_tool() → given a name + inputs, calls the right Python function

Adding a new tool means: write the function, add it to TOOL_MAP,
add its schema to TOOL_SCHEMAS. Nothing else needs to change.
"""

from .web_search import web_search
from .wikipedia import wikipedia_search
from .url_reader import read_url

# Maps tool name → callable. execute_tool() looks up here.
TOOL_MAP = {
    "web_search": web_search,
    "wikipedia_search": wikipedia_search,
    "read_url": read_url,
}

# JSON schemas in the format Anthropic's /messages API expects.
# Each entry becomes one element of the `tools` parameter.
TOOL_SCHEMAS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for current information, recent events, news, "
            "statistics, and general facts. Returns titles, URLs, and content "
            "snippets from multiple sources. Use this first for most queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific for better results.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "wikipedia_search",
        "description": (
            "Search Wikipedia for encyclopedic background on a topic. "
            "Best for well-established subjects: historical events, scientific "
            "concepts, notable people, places. Not suitable for very recent events."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic name to look up on Wikipedia.",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "read_url",
        "description": (
            "Fetch and extract the full text content of a specific URL. "
            "Use this when a search result looks relevant but the snippet "
            "is too short — pass in the URL to get the complete article text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch, including https://",
                },
            },
            "required": ["url"],
        },
    },
]


def get_tools() -> list[dict]:
    """Return all tool schemas for passing to the Anthropic API."""
    return TOOL_SCHEMAS


def execute_tool(name: str, inputs: dict) -> str:
    """
    Execute a tool by name with the given inputs.

    Always returns a string — errors are returned as strings so
    the agent can read them and decide how to recover, rather
    than crashing the loop with an exception.
    """
    if name not in TOOL_MAP:
        return (
            f"Unknown tool '{name}'. "
            f"Available tools: {', '.join(TOOL_MAP.keys())}"
        )
    try:
        return TOOL_MAP[name](**inputs)
    except TypeError as e:
        # Wrong arguments — likely a schema mismatch
        return f"Tool '{name}' called with wrong arguments: {e}"
    except Exception as e:
        return f"Tool '{name}' failed: {type(e).__name__}: {e}"
