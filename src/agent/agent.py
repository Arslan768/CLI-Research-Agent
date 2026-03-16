"""
agent.py

ReAct agent loop using Google Gemini (free tier).
Logic is identical to the Anthropic version — only the
API client and message format differ.
"""

from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

from .tools.registry import get_tools, execute_tool, TOOL_MAP
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

MODEL = "gemini-2.5-flash"


def _build_gemini_tools():
    """
    Convert our tool schemas into the format Gemini expects.
    """
    declarations = []
    for schema in get_tools():
        properties = {}
        for prop_name, prop_def in schema["input_schema"]["properties"].items():
            prop_type = prop_def.get("type", "string").upper()
            gemini_type = getattr(types.Type, prop_type, types.Type.STRING)
            properties[prop_name] = types.Schema(
                type=gemini_type,
                description=prop_def.get("description", ""),
            )

        declarations.append(
            types.FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=schema["input_schema"].get("required", []),
                ),
            )
        )

    return [types.Tool(function_declarations=declarations)]


def run(query: str, max_iterations: int = 10) -> str:
    """
    Run the ReAct agent on a query and return the final answer.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not set in .env file."

    client = genai.Client(api_key=api_key)

    chat = client.chats.create(
        model=MODEL,
        config={
            "system_instruction": _SYSTEM_PROMPT,
            "tools": _build_gemini_tools(),
        }
    )

    tool_response_parts = None

    for iteration in range(1, max_iterations + 1):
        print_iteration_header(iteration)

        # ── THOUGHT ──────────────────────────────────────────────
        if iteration == 1:
            response = chat.send_message(query)
        else:
            response = chat.send_message(tool_response_parts)

        candidate = response.candidates[0]
        content = candidate.content

        # ── EXTRACT TEXT + TOOL CALLS ─────────────────────────────
        thought_text = ""
        tool_calls = []

        for part in content.parts:
            if hasattr(part, "text") and part.text:
                thought_text += part.text
            if (
                hasattr(part, "function_call")
                and part.function_call is not None
                and hasattr(part.function_call, "name")
                and part.function_call.name
            ):
                tool_calls.append(part.function_call)

        if thought_text:
            print_thought(thought_text)

        # ── TERMINATION CHECK ─────────────────────────────────────
        if not tool_calls:
            if thought_text:
                print_final_answer(thought_text)
                return thought_text

            # Gemini returned neither text nor tool calls — nudge it to conclude
            if iteration == 1:
                print_final_answer("No answer generated.")
                return "No answer generated."

            nudge_response = chat.send_message(
                "Based on your research so far, please provide a complete summary."
            )
            nudge_candidate = nudge_response.candidates[0]
            final_answer = "".join(
                part.text
                for part in nudge_candidate.content.parts
                if hasattr(part, "text") and part.text
            ) or "No answer generated."
            print_final_answer(final_answer)
            return final_answer

        # ── ACTION + OBSERVATION ──────────────────────────────────
        tool_response_parts = []
        for fn_call in tool_calls:
            name = fn_call.name
            inputs = dict(fn_call.args)

            print_tool_call(name, inputs)
            result = execute_tool(name, inputs)
            print_tool_result(name, result, iteration)

            tool_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=name,
                        response={"result": result},
                    )
                )
            )

    print_warning(f"Reached max_iterations ({max_iterations}) without a final answer.")
    return "Unable to complete research within the iteration limit."