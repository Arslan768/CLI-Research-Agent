"""
cli/main.py

The CLI entrypoint. Typer turns Python function signatures
into proper CLI commands with --help, argument validation,
and nice error messages for free.

Usage:
    research ask "What caused the 2008 financial crisis?"
    research ask "Latest developments in fusion energy" --iterations 15
    research ask "Who is Yann LeCun?" --quiet
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add src/ to the Python path so `from agent import run` works
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent import run
from agent.streaming import print_query, print_error

app = typer.Typer(
    name="research",
    help="A ReAct-powered CLI research agent using Gemini + web search.",
    add_completion=False,
)

console = Console()


@app.command()
def ask(
    query: str = typer.Argument(
        ...,
        help="The research question to answer.",
    ),
    iterations: int = typer.Option(
        10,
        "--iterations", "-i",
        help="Maximum number of tool-calling rounds. Default: 10.",
        min=1,
        max=25,
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress reasoning steps; only print the final answer.",
    ),
) -> None:
    """Ask a research question. The agent will search the web and synthesize an answer."""
    if not query.strip():
        print_error("Query cannot be empty.")
        raise typer.Exit(code=1)

    if not quiet:
        print_query(query)

    try:
        answer = run(query=query, max_iterations=iterations)

        if quiet:
            # In quiet mode we only print the answer, no panels
            console.print(answer)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        print_error(f"Agent failed: {type(e).__name__}: {e}")
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the version and exit."""
    console.print("[bold]cli-research-agent[/bold] v0.1.0")


if __name__ == "__main__":
    app()
