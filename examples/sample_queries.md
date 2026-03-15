# Sample queries for portfolio demos

These queries showcase multi-step reasoning well — the agent
visibly uses 2-4 tool calls before synthesizing an answer.

## Good demo queries

```bash
# Multi-source research (shows search + wikipedia + url reader)
research ask "What caused the 2008 financial crisis and what reforms followed?"

# Recent events (forces web search, can't rely on training data)
research ask "What are the most recent breakthroughs in fusion energy?"

# Comparative research (agent naturally searches multiple angles)
research ask "How does the US healthcare system compare to Germany's?"

# Technical deep-dive (shows the agent reading full URLs)
research ask "How does RLHF work in training large language models?"

# Current state of a field
research ask "What is the current state of quantum computing in 2025?"
```

## What to capture in your DEMO.gif

1. The query appearing in the cyan panel
2. At least 2-3 "step N" separators showing the agent reasoning
3. The purple "Thought" panels showing Claude narrating its plan
4. The green "Observation" panels showing tool results coming back
5. The final answer rendered in Markdown

## Recording tip

Use `vhs` to record a reproducible terminal demo:
https://github.com/charmbracelet/vhs
