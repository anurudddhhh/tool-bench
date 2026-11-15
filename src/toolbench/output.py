"""Terminal output and presentation formatting using Rich (F11e).

Provides reusable status indicators, tables, progress bars, and prompt panels
ensuring clean and consistent CLI output across all Toolbench commands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


def print_success(message: str, target_console: Console | None = None) -> None:
    """Print a success message with green checkmark."""
    c = target_console or console
    c.print(f"[bold green]✓[/] {message}")


def print_warning(message: str, target_console: Console | None = None) -> None:
    """Print a warning message with yellow alert icon."""
    c = target_console or console
    c.print(f"[bold yellow]⚠[/] {message}")


def print_error(message: str, target_console: Console | None = None) -> None:
    """Print an error message with red crossmark."""
    c = target_console or console
    c.print(f"[bold red]✗[/] {message}")


def print_info(message: str, target_console: Console | None = None) -> None:
    """Print an informational message with cyan indicator."""
    c = target_console or console
    c.print(f"[bold cyan]ℹ[/] {message}")


def render_tools_table(
    tools: list[dict[str, Any]],
    title: str = "Discovered Tools",
) -> Table:
    """Construct a formatted Rich Table summarizing tool names, descriptions, and schemas."""
    table = Table(title=title, show_lines=True)
    table.add_column("Server", style="cyan", no_wrap=True)
    table.add_column("Tool Name", style="bold white")
    table.add_column("Description", style="dim")
    table.add_column("Params", justify="right", style="magenta")

    for tool in tools:
        raw_name = str(tool.get("name", ""))
        if "__" in raw_name:
            server, name = raw_name.split("__", 1)
        else:
            server, name = "-", raw_name

        desc = str(tool.get("description", ""))
        if len(desc) > 80:
            desc = desc[:77] + "..."

        schema = tool.get("parameters") or tool.get("input_schema") or {}
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        param_count = str(len(properties))

        table.add_row(server, name, desc or "[italic dim]No description[/]", param_count)

    return table


def print_tools_table(
    tools: list[dict[str, Any]],
    title: str = "Discovered Tools",
    target_console: Console | None = None,
) -> None:
    """Print formatted tools table to console."""
    c = target_console or console
    table = render_tools_table(tools=tools, title=title)
    c.print(table)


def render_prompt_panel(
    prompt_id: int | str,
    prompt_text: str,
    expected_tool: str,
    expected_args: dict[str, Any],
    description: str = "",
) -> Panel:
    """Construct a stylized Panel presenting a prompt for review (F3)."""
    args_json = json.dumps(expected_args, indent=2)
    content = (
        f"[bold]Prompt text:[/]\n{prompt_text}\n\n"
        f"[bold cyan]Expected tool:[/] [white]{expected_tool}[/]\n"
        f"[bold dim]Tool description:[/] {description or 'N/A'}\n\n"
        f"[bold magenta]Expected arguments:[/]\n[dim]{args_json}[/]"
    )
    return Panel(
        content,
        title=f"[bold blue]Prompt #{prompt_id}[/]",
        border_style="blue",
        padding=(1, 2),
    )


def print_prompt_panel(
    prompt_id: int | str,
    prompt_text: str,
    expected_tool: str,
    expected_args: dict[str, Any],
    description: str = "",
    target_console: Console | None = None,
) -> None:
    """Print prompt panel to console."""
    c = target_console or console
    panel = render_prompt_panel(
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        expected_tool=expected_tool,
        expected_args=expected_args,
        description=description,
    )
    c.print(panel)


@dataclass
class RunMetrics:
    """Pre-computed benchmark metrics container for output presentation."""

    total_prompts: int = 0
    tool_correct: int = 0
    tool_accuracy_pct: float = 0.0
    strict_match_count: int = 0
    strict_match_pct: float = 0.0
    cache_hits: int = 0
    avg_latency_ms: float = 0.0


def compute_run_metrics(results: list[dict[str, Any]]) -> RunMetrics:
    """Compute high-level benchmark metrics from raw per-prompt result dictionaries."""
    total = len(results)
    if total == 0:
        return RunMetrics()

    tool_correct = sum(1 for r in results if r.get("tool_correct") is True)
    strict_match = sum(1 for r in results if r.get("args_strict_match") is True)
    cached_hits = sum(1 for r in results if r.get("cached") is True)
    latencies = [
        float(r.get("latency_ms", 0.0))
        for r in results
        if r.get("latency_ms") is not None
    ]
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    return RunMetrics(
        total_prompts=total,
        tool_correct=tool_correct,
        tool_accuracy_pct=(tool_correct / total * 100.0),
        strict_match_count=strict_match,
        strict_match_pct=(strict_match / total * 100.0),
        cache_hits=cached_hits,
        avg_latency_ms=avg_latency,
    )


def render_run_summary(
    metrics: RunMetrics | list[dict[str, Any]],
    title: str = "Benchmark Execution Summary",
) -> Table:
    """Generate a high-level metrics table for display."""
    if isinstance(metrics, list):
        data = compute_run_metrics(metrics)
    else:
        data = metrics

    table = Table(title=title, show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right", style="white")

    color = "green" if data.tool_accuracy_pct >= 80.0 else "yellow"

    table.add_row("Total Prompts Evaluated", str(data.total_prompts))
    table.add_row(
        "Tool Selection Accuracy",
        f"[{color}]{data.tool_accuracy_pct:.1f}% ({data.tool_correct}/{data.total_prompts})[/]",
    )
    table.add_row(
        "Strict Arguments Match Rate",
        f"{data.strict_match_pct:.1f}% ({data.strict_match_count}/{data.total_prompts})",
    )
    table.add_row("Cache Hits", f"{data.cache_hits}/{data.total_prompts}")
    table.add_row("Average Latency", f"{data.avg_latency_ms:.1f} ms")

    return table


def print_run_summary(
    metrics: RunMetrics | list[dict[str, Any]],
    title: str = "Benchmark Execution Summary",
    target_console: Console | None = None,
) -> None:
    """Print benchmark metrics summary table to console."""
    c = target_console or console
    table = render_run_summary(metrics=metrics, title=title)
    c.print(table)


def create_progress(target_console: Console | None = None) -> Progress:
    """Construct a standardized Rich Progress instance."""
    c = target_console or console
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/]"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=c,
    )
