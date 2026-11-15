"""Interactive Terminal UI (TUI) Dashboard for Toolbench evaluation results (D4).

Provides an interactive terminal dashboard built with Rich layouts, panels, and tables
to browse 2x2 factorial benchmark runs, statistical significance comparisons,
failure mode distributions, and prompt-level discordance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from toolbench.report import analyze_failures, compare_setups

logger = logging.getLogger(__name__)
console = Console()


class TerminalDashboard:
    """Manages data loading and Rich terminal dashboard views for benchmark results."""

    def __init__(
        self,
        results_dir: str | Path | None = None,
        results_map: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the dashboard with results directory or pre-loaded results.

        Args:
            results_dir: Path to directory containing evaluation JSON results (e.g. 'results/').
            results_map: Optional pre-loaded mapping of setup letters to result dicts.
        """
        self.results_dir = Path(results_dir) if results_dir else Path("results")
        self.results_map: dict[str, dict[str, Any]] = results_map or {}

        if not self.results_map and self.results_dir.exists():
            self._load_results_from_dir()

        self._comparison_report: dict[str, Any] | None = None
        if len(self.results_map) >= 2:
            try:
                self._comparison_report = compare_setups(self.results_map)
            except Exception as err:
                logger.warning("Failed to compute comparison report: %s", err)

    def _load_results_from_dir(self) -> None:
        """Discover and load the latest result file for each setup letter."""
        for letter in ("A", "B", "C", "D"):
            candidates = sorted(self.results_dir.glob(f"{letter}_*.json"))
            if candidates:
                latest = candidates[-1]
                try:
                    data = json.loads(latest.read_text(encoding="utf-8"))
                    self.results_map[letter] = data
                except Exception as err:
                    logger.warning("Could not read %s: %s", latest, err)

    def render_summary_panel(self) -> Panel:
        """Generate a summary table panel comparing all available setups."""
        table = Table(box=None, expand=True, show_header=True)
        table.add_column("Setup", style="bold cyan")
        table.add_column("Intervention Description", style="dim")
        table.add_column("Accuracy", justify="right")
        table.add_column("Strict Match", justify="right")
        table.add_column("Retry Recov.", justify="right")
        table.add_column("Avg Latency", justify="right")
        table.add_column("Prompts", justify="right")

        setup_desc = {
            "A": "Baseline (All tools + original descriptions)",
            "B": "Rewriting (All tools + rewritten descriptions)",
            "C": "Retrieval (Top-k tools + original descriptions)",
            "D": "Combined (Top-k tools + rewritten descriptions)",
        }

        for letter in ("A", "B", "C", "D"):
            data = self.results_map.get(letter)
            if not data:
                continue
            metrics = data.get("metrics", {})
            acc = metrics.get("tool_accuracy", 0.0) * 100.0
            strict = metrics.get("args_strict_match_rate", 0.0) * 100.0
            recov = metrics.get("recovery_rate", 0.0) * 100.0
            latency = metrics.get("avg_latency_ms", 0.0)
            total = metrics.get("total_prompts", len(data.get("results", [])))

            acc_style = "bold green" if acc >= 80.0 else "bold yellow"
            table.add_row(
                f"[bold cyan]Setup {letter}[/]",
                setup_desc.get(letter, "Custom"),
                f"[{acc_style}]{acc:.1f}%[/]",
                f"{strict:.1f}%",
                f"{recov:.1f}%",
                f"{latency:.0f} ms",
                str(total),
            )

        return Panel(
            table,
            title="[bold white]2x2 Factorial Benchmark Summary[/]",
            border_style="cyan",
        )

    def render_stats_panel(self) -> Panel:
        """Generate a panel displaying paired statistical test results."""
        if not self._comparison_report or "primary_comparisons" not in self._comparison_report:
            return Panel(
                Text("Need at least 2 evaluated setups to compute statistical significance."),
                title="Paired Statistical Comparisons",
                border_style="dim",
            )

        table = Table(box=None, expand=True, show_header=True)
        table.add_column("Comparison Pair", style="bold white")
        table.add_column("Accuracy Delta", justify="right")
        table.add_column("95% Bootstrap CI", justify="center")
        table.add_column("McNemar p-val", justify="right")
        table.add_column("Holm Adj p", justify="right")
        table.add_column("Significance", justify="center")

        all_comps = (
            self._comparison_report.get("primary_comparisons", [])
            + self._comparison_report.get("exploratory_comparisons", [])
        )

        for comp in all_comps:
            pair = comp.get("comparison", "")
            diff = comp.get("delta", 0.0) * 100.0
            ci_low = comp.get("ci_95_lower", 0.0) * 100.0
            ci_high = comp.get("ci_95_upper", 0.0) * 100.0
            raw_p = comp.get("raw_p_value", 1.0)
            adj_p = comp.get("adjusted_p_value", 1.0)
            sig = comp.get("significant", False)

            diff_str = f"{'+' if diff >= 0 else ''}{diff:.1f}%"
            ci_str = f"[{ci_low:+.1f}%, {ci_high:+.1f}%]"
            sig_str = "[bold green]Significant[/]" if sig else "[dim]Not Significant[/]"
            p_str = "< 0.001" if raw_p < 0.001 else f"{raw_p:.4f}"
            adj_str = "< 0.001" if adj_p < 0.001 else f"{adj_p:.4f}"

            table.add_row(
                pair,
                f"[bold]{diff_str}[/]",
                ci_str,
                p_str,
                adj_str,
                sig_str,
            )

        return Panel(
            table,
            title="[bold white]Paired Statistical Tests (McNemar + 2,000 Bootstrap CIs)[/]",
            border_style="cyan",
        )

    def render_failures_panel(self, setup: str = "A") -> Panel:
        """Generate a failure breakdown panel with category percentages and bar charts."""
        data = self.results_map.get(setup)
        if not data:
            return Panel(
                Text(f"No result data found for Setup {setup}."),
                title=f"Failure Analysis (Setup {setup})",
                border_style="dim",
            )

        analysis = analyze_failures(data)
        counts = analysis["category_counts"]
        pcts = analysis["category_percentages"]
        total = max(1, analysis["total_prompts"])

        table = Table(box=None, expand=True, show_header=True)
        table.add_column("Failure Category", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Percent", justify="right")
        table.add_column("Distribution", style="green")

        for cat, cnt in counts.items():
            pct = pcts.get(cat, 0.0)
            bar_len = int((cnt / total) * 30)
            bar = "█" * bar_len

            style = "green" if cat == "pass" else "yellow" if "recovery" in cat else "red"
            table.add_row(
                f"[{style}]{cat}[/]",
                str(cnt),
                f"{pct:.1f}%",
                f"[{style}]{bar}[/]",
            )

        return Panel(
            table,
            title=f"[bold white]Failure Mode Analysis — Setup {setup}[/]",
            border_style="cyan",
        )

    def render_discordance_panel(self, setup_1: str = "A", setup_2: str = "B") -> Panel:
        """Inspect prompt-level discordance (disagreements) between two setups."""
        d1 = self.results_map.get(setup_1, {}).get("results", [])
        d2 = self.results_map.get(setup_2, {}).get("results", [])

        if not d1 or not d2:
            return Panel(
                Text(f"Missing data for paired discordance check ({setup_1} vs {setup_2})."),
                title=f"Discordance Browser ({setup_1} vs {setup_2})",
                border_style="dim",
            )

        table = Table(box=None, expand=True, show_header=True)
        table.add_column("Prompt ID", style="cyan")
        table.add_column(f"Setup {setup_1}", justify="center")
        table.add_column(f"Setup {setup_2}", justify="center")
        table.add_column("Query Snippet", style="dim")

        map2 = {str(r.get("prompt_id")): r for r in d2}
        discordant_count = 0

        for r1 in d1:
            pid = str(r1.get("prompt_id"))
            r2 = map2.get(pid)
            if not r2:
                continue

            c1 = bool(r1.get("tool_correct", False))
            c2 = bool(r2.get("tool_correct", False))

            if c1 != c2:
                discordant_count += 1
                q = str(r1.get("prompt", ""))[:60]
                s1_text = "[green]Pass[/]" if c1 else "[red]Fail[/]"
                s2_text = "[green]Pass[/]" if c2 else "[red]Fail[/]"
                table.add_row(pid, s1_text, s2_text, q)

        diff_label = f"{discordant_count} differences between {setup_1} and {setup_2}"
        title = f"[bold white]Discordant Prompts ({diff_label})[/]"
        return Panel(table, title=title, border_style="cyan")

    def render_full_dashboard(self) -> Layout:
        """Compose the complete terminal dashboard into a single Rich Layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="summary", size=9),
            Layout(name="stats", size=8),
            Layout(name="failures", size=10),
        )

        header_text = Text.from_markup(
            "[bold white]TOOLBENCH[/] [dim]— MCP LLM Tool Selection Benchmarking Suite[/]",
            justify="center",
        )
        layout["header"].update(Panel(header_text, style="cyan"))
        layout["summary"].update(self.render_summary_panel())
        layout["stats"].update(self.render_stats_panel())
        layout["failures"].update(self.render_failures_panel("A"))

        return layout


def run_interactive_dashboard(
    dashboard: TerminalDashboard,
    input_provider: Callable[[str], str] = input,
    output_console: Console | None = None,
) -> None:
    """Run an interactive terminal navigation loop for viewing benchmark results."""
    con = output_console or console
    active_setup = "A"

    while True:
        con.print("\n[bold cyan]─── Toolbench Terminal Dashboard ───[/]")
        con.print(
            "[white]\\[1] Full Overview  \\[2] Statistical Tests\n"
            "\\[3] Failure Breakdown  \\[4] Discordance  \\[q] Exit[/]"
        )

        try:
            choice = input_provider("Select view: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice in ("q", "quit", "exit"):
            con.print("[dim]Exiting dashboard.[/]")
            break
        elif choice == "1":
            con.print(dashboard.render_full_dashboard())
        elif choice == "2":
            con.print(dashboard.render_stats_panel())
        elif choice == "3":
            con.print(dashboard.render_failures_panel(active_setup))
        elif choice == "4":
            con.print(dashboard.render_discordance_panel("A", "B"))
        else:
            con.print("[yellow]Invalid option. Choose 1, 2, 3, 4, or q.[/]")
