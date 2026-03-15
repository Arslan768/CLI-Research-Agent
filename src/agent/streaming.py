"""
streaming.py

Handles all Rich terminal output. Keeps display logic
out of the agent loop so agent.py stays focused on logic.
"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text

console = Console()


def print_query(query: str) -> None:
    """Print the user's query in a styled panel."""
    console.print()
    console.print(
        Panel(
            Text(query, style="bold white"),
            title="[bold cyan]Research Query[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    console.print()


def print_thought(text: str) -> None:
    """Print the agent's reasoning text."""
    if text.strip():
        console.print(
            Panel(
                text.strip(),
                title="[bold purple]Thought[/bold purple]",
                border_style="purple",
                padding=(0, 1),
        )
    )


def print_tool_call(tool_name: str, inputs: dict) -> None:
    """Print a tool invocation with its arguments."""
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in inputs.items())
    console.print(
        f"  [bold green]>[/bold green] [green]{tool_name}[/green]"
        f"([yellow]{args_str}[/yellow])"
    )


def print_tool_result(tool_name: str, result: str, iteration: int) -> None:
    """Print the result returned by a tool, truncated if very long."""
    max_preview = 400
    preview = result[:max_preview] + ("..." if len(result) > max_preview else "")
    console.print(
        Panel(
            preview,
            title=f"[bold green]Observation — {tool_name}[/bold green]",
            border_style="green",
            padding=(0, 1),
            subtitle=f"[dim]iteration {iteration}[/dim]",
        )
    )


def print_iteration_header(n: int) -> None:
    """Print a separator between reasoning iterations."""
    console.print(Rule(f"[dim]step {n}[/dim]", style="dim"))


def print_final_answer(answer: str) -> None:
    """Print the agent's final answer rendered as Markdown."""
    console.print()
    console.print(Rule("[bold cyan]Answer[/bold cyan]", style="cyan"))
    console.print()
    console.print(Markdown(answer))
    console.print()


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
