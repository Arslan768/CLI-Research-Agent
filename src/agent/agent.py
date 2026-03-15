"""
agent.py - ReAct loop using Groq (free tier)
"""

from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import os
import json

from .tools.registry import get_tools, execute_tool
from .streaming import (
    print_thought,
    print_tool_call,
    print_tool_result,
    print_iteration_header,
    print_final_answer,
    print_warning,
)

load_dotenv()

_SYSTEM_PROMPT = (
    Path(__file__).parent / "prompts" / "system.txt"
).read_text()

MODEL = "llama-3.3-70b-versatile"


def run(query: str, max_iterations: int = 10) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not set in .env file."

    client = Groq(api_key=api_key)

    tools = []
    for t in get_tools():
        tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
        })

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    for iteration in range(1, max_iterations + 1):
        print_iteration_header(iteration)

        # First iteration: require a tool call so the agent always researches.
        # Subsequent iterations: auto, so it can either call tools or answer.
        tool_choice = "required" if iteration == 1 else "auto"

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=False,
            max_tokens=4096,
        )

        message = response.choices[0].message

        # Append assistant turn to history
        assistant_msg = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if message.content:
            print_thought(message.content)

        # ── TERMINATION CHECK ─────────────────────────────────────
        if not message.tool_calls:
            final_answer = message.content or "No answer generated."
            print_final_answer(final_answer)
            return final_answer

        # ── ACTION + OBSERVATION ──────────────────────────────────
        for tc in message.tool_calls:
            name = tc.function.name
            inputs = json.loads(tc.function.arguments)

            print_tool_call(name, inputs)
            result = execute_tool(name, inputs)
            print_tool_result(name, result, iteration)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    print_warning(f"Reached max_iterations ({max_iterations}).")
    return "Unable to complete research within the iteration limit."