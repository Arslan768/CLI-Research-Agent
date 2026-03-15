# CLI Research Agent

A terminal-based AI research agent built with Claude + Tavily. Ask any question and watch it reason through multiple web searches, synthesize sources, and return a cited answer — all visible in real time.

Built as Project 01 of the [Agentic AI Engineer roadmap](https://github.com/yourusername).

## Demo

```
$ research ask "What caused the 2008 financial crisis?"

╭─ Research Query ──────────────────────────────────────────╮
│ What caused the 2008 financial crisis?                     │
╰────────────────────────────────────────────────────────────╯

──────────────────────── step 1 ────────────────────────────
╭─ Thought ──────────────────────────────────────────────────╮
│ I'll search for the main causes of the 2008 financial      │
│ crisis to give a comprehensive answer.                     │
╰────────────────────────────────────────────────────────────╯
  > web_search(query='2008 financial crisis main causes')
╭─ Observation — web_search  [iteration 1] ─────────────────╮
│ [1] 2008 Crisis Overview — Investopedia                    │
│ URL: https://investopedia.com/...                          │
│ The crisis stemmed from the collapse of the housing...     │
╰────────────────────────────────────────────────────────────╯

──────────────────────── step 2 ────────────────────────────
╭─ Thought ──────────────────────────────────────────────────╮
│ I have good information. Let me also check Wikipedia for   │
│ the regulatory background.                                 │
╰────────────────────────────────────────────────────────────╯
  > wikipedia_search(topic='2007–2008 financial crisis')

──────────────────────── Answer ────────────────────────────

## The 2008 Financial Crisis: Causes

The 2008 crisis resulted from several interconnected factors...
```

## Install

```bash
git clone https://github.com/yourusername/cli-research-agent
cd cli-research-agent

# Install uv if you haven't: https://docs.astral.sh/uv/
uv sync

cp .env.example .env
# Edit .env and add your API keys
```

Get your API keys:
- **GEMINI API KEY**: https://aistudio.google.com/ 

Select GET API KEY from the left side panel. Proceed with the required steps

- **Tavily** (free tier available): https://tavily.com

## Usage

```bash
# Basic query
uv run research ask "What is the current state of fusion energy?"

# More iterations for complex topics
uv run research ask "Compare US and European healthcare systems" --iterations 15

# Quiet mode (final answer only, no reasoning steps)
uv run research ask "Who founded OpenAI?" --quiet

# Help
uv run research --help
```

## Architecture

```
src/agent/
├── agent.py          # ReAct loop: thought → action → observation
├── streaming.py      # Rich terminal output
└── tools/
    ├── registry.py   # Tool schemas + dispatcher
    ├── web_search.py # Tavily web search
    ├── wikipedia.py  # Wikipedia lookup
    └── url_reader.py # Full page text extraction
```

The loop in `agent.py` is ~40 lines. The core idea: build a `messages` list, call Claude with tool schemas, execute any tool calls, append results, repeat until `stop_reason == "end_turn"`.

## Run tests

```bash
uv run pytest tests/ -v
```

Tests use mocks — no API calls, no credits consumed.

## License

MIT
