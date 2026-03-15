"""
tests/test_agent.py

Integration tests for the full ReAct agent loop. We mock the
Anthropic API client so these tests run offline and deterministically.

Run with:  pytest tests/test_agent.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def make_text_block(text: str) -> MagicMock:
    """Helper: create a fake TextBlock."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(name: str, inputs: dict, block_id: str = "tu_001") -> MagicMock:
    """Helper: create a fake ToolUseBlock."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = inputs
    block.id = block_id
    return block


def make_response(content: list, stop_reason: str) -> MagicMock:
    """Helper: create a fake Anthropic response."""
    response = MagicMock()
    response.content = content
    response.stop_reason = stop_reason
    return response


class TestAgentLoop:
    def test_single_turn_no_tool_call(self):
        """Agent gives a direct answer without calling any tools."""
        from agent.agent import run

        final_answer = "The capital of France is Paris."
        response = make_response(
            content=[make_text_block(final_answer)],
            stop_reason="end_turn",
        )

        with patch("agent.agent.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = response
            result = run("What is the capital of France?", max_iterations=5)

        assert "Paris" in result
        # Only one API call was made
        MockAnthropic.return_value.messages.create.assert_called_once()

    def test_one_tool_call_then_answer(self):
        """Agent calls one tool, gets the result, then gives a final answer."""
        from agent.agent import run

        # Turn 1: Claude decides to search
        turn1 = make_response(
            content=[
                make_text_block("I'll search for information."),
                make_tool_use_block("web_search", {"query": "2008 financial crisis causes"}, "tu_001"),
            ],
            stop_reason="tool_use",
        )

        # Turn 2: Claude synthesizes and answers
        turn2 = make_response(
            content=[make_text_block("The 2008 crisis was caused by subprime mortgages and deregulation.")],
            stop_reason="end_turn",
        )

        with patch("agent.agent.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.side_effect = [turn1, turn2]
            with patch("agent.agent.execute_tool", return_value="Search results about 2008 crisis...") as mock_tool:
                result = run("What caused the 2008 financial crisis?", max_iterations=5)

        assert "subprime" in result.lower() or "crisis" in result.lower()
        mock_tool.assert_called_once_with("web_search", {"query": "2008 financial crisis causes"})
        assert MockAnthropic.return_value.messages.create.call_count == 2

    def test_multiple_tool_calls_in_sequence(self):
        """Agent calls tools across multiple iterations before answering."""
        from agent.agent import run

        turn1 = make_response(
            content=[make_tool_use_block("web_search", {"query": "fusion energy 2024"}, "tu_001")],
            stop_reason="tool_use",
        )
        turn2 = make_response(
            content=[make_tool_use_block("wikipedia_search", {"topic": "nuclear fusion"}, "tu_002")],
            stop_reason="tool_use",
        )
        turn3 = make_response(
            content=[make_text_block("Fusion energy has made significant progress...")],
            stop_reason="end_turn",
        )

        with patch("agent.agent.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.side_effect = [turn1, turn2, turn3]
            with patch("agent.agent.execute_tool", return_value="Tool result"):
                result = run("Latest in fusion energy?", max_iterations=5)

        assert isinstance(result, str)
        assert MockAnthropic.return_value.messages.create.call_count == 3

    def test_max_iterations_guard(self):
        """Agent stops and returns a message when max_iterations is hit."""
        from agent.agent import run

        # Always return a tool call — never terminates naturally
        endless_tool_call = make_response(
            content=[make_tool_use_block("web_search", {"query": "something"}, "tu_001")],
            stop_reason="tool_use",
        )

        with patch("agent.agent.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = endless_tool_call
            with patch("agent.agent.execute_tool", return_value="result"):
                result = run("Never-ending query", max_iterations=3)

        assert isinstance(result, str)
        # Should have called the API exactly max_iterations times
        assert MockAnthropic.return_value.messages.create.call_count == 3

    def test_message_history_grows_correctly(self):
        """Verify the messages list is built correctly across turns."""
        from agent.agent import run

        turn1 = make_response(
            content=[make_tool_use_block("web_search", {"query": "test"}, "tu_001")],
            stop_reason="tool_use",
        )
        turn2 = make_response(
            content=[make_text_block("Final answer.")],
            stop_reason="end_turn",
        )

        captured_calls = []

        def capture_call(**kwargs):
            captured_calls.append(kwargs["messages"][:])  # snapshot
            return [turn1, turn2][len(captured_calls) - 1]

        with patch("agent.agent.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.side_effect = (
                lambda **kw: capture_call(**kw)
            )
            with patch("agent.agent.execute_tool", return_value="Tool output"):
                run("Test query", max_iterations=5)

        # First call: just the user query
        assert captured_calls[0][0]["role"] == "user"
        assert "Test query" in captured_calls[0][0]["content"]

        # Second call: user query + assistant (tool call) + user (tool result)
        assert len(captured_calls[1]) == 3
        assert captured_calls[1][1]["role"] == "assistant"
        assert captured_calls[1][2]["role"] == "user"
        # The tool result message contains the observation
        tool_result_content = captured_calls[1][2]["content"]
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "tu_001"

    def test_tool_error_does_not_crash_loop(self):
        """A tool returning an error string should not stop the agent."""
        from agent.agent import run

        turn1 = make_response(
            content=[make_tool_use_block("web_search", {"query": "test"}, "tu_001")],
            stop_reason="tool_use",
        )
        turn2 = make_response(
            content=[make_text_block("I encountered an error but here's what I know.")],
            stop_reason="end_turn",
        )

        with patch("agent.agent.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.side_effect = [turn1, turn2]
            # Tool returns an error string — should not raise
            with patch("agent.agent.execute_tool", return_value="Search error: Connection refused"):
                result = run("Test query", max_iterations=5)

        assert isinstance(result, str)
        # Agent completed despite the tool error
