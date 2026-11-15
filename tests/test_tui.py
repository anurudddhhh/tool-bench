"""Unit tests for the Terminal UI Dashboard (D4)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel

from toolbench.tui import TerminalDashboard, run_interactive_dashboard


def _make_sample_result(setup: str, acc: float) -> dict:
    return {
        "setup": setup,
        "run_id": f"test-run-{setup}",
        "model": "gemini-3.5-flash-lite",
        "timestamp": "2026-11-15T12:00:00Z",
        "evaluation_temperature": 0.0,
        "metrics": {
            "tool_accuracy": acc,
            "args_valid_pre_retry_rate": 0.8,
            "args_valid_post_retry_rate": 0.9,
            "args_strict_match_rate": 0.75,
            "recovery_rate": 0.5,
            "args_match_score_mean": 0.85,
            "total_prompts": 2,
            "avg_latency_ms": 250.0,
            "cache_hit_rate": 0.0,
        },
        "results": [
            {
                "prompt_id": "1",
                "prompt": "Read the file foo.txt",
                "expected_tool": "fs_read",
                "chosen_tool": "fs_read" if acc >= 0.5 else "fs_write",
                "tool_correct": acc >= 0.5,
                "args_valid_pre_retry": True,
                "args_valid_post_retry": True,
                "args_strict_match": True,
                "exposed_tools": ["fs_read", "fs_write"],
                "retrieval_hit": True,
            },
            {
                "prompt_id": "2",
                "prompt": "List git status",
                "expected_tool": "git_status",
                "chosen_tool": "git_status" if acc >= 0.8 else "fs_write",
                "tool_correct": acc >= 0.8,
                "args_valid_pre_retry": True,
                "args_valid_post_retry": True,
                "args_strict_match": True,
                "exposed_tools": ["git_status", "fs_write"],
                "retrieval_hit": True,
            },
        ],
    }


def test_tui_summary_panel():
    """Verify summary panel formats metrics for loaded setups."""
    results = {
        "A": _make_sample_result("A", 0.5),
        "B": _make_sample_result("B", 1.0),
    }
    dashboard = TerminalDashboard(results_map=results)
    panel = dashboard.render_summary_panel()

    assert isinstance(panel, Panel)
    assert panel.title is not None
    assert "2x2 Factorial" in str(panel.title)


def test_tui_stats_panel():
    """Verify statistical significance panel shows McNemar and bootstrap comparisons."""
    results = {
        "A": _make_sample_result("A", 0.5),
        "B": _make_sample_result("B", 1.0),
    }
    dashboard = TerminalDashboard(results_map=results)
    panel = dashboard.render_stats_panel()

    assert isinstance(panel, Panel)
    assert "Statistical Tests" in str(panel.title)


def test_tui_failures_panel():
    """Verify failure panel categorizes errors and builds visual distribution."""
    results = {"A": _make_sample_result("A", 0.5)}
    dashboard = TerminalDashboard(results_map=results)
    panel = dashboard.render_failures_panel("A")

    assert isinstance(panel, Panel)
    assert "Failure Mode Analysis" in str(panel.title)


def test_tui_discordance_panel():
    """Verify discordance panel highlights disagreements between paired conditions."""
    results = {
        "A": _make_sample_result("A", 0.5),  # prompt 1 correct, prompt 2 incorrect
        "B": _make_sample_result("B", 1.0),  # prompt 1 & 2 correct
    }
    dashboard = TerminalDashboard(results_map=results)
    panel = dashboard.render_discordance_panel("A", "B")

    assert isinstance(panel, Panel)
    assert "Discordant Prompts" in str(panel.title)


def test_tui_full_dashboard_layout():
    """Verify composite dashboard layout contains all expected view sections."""
    results = {
        "A": _make_sample_result("A", 0.5),
        "B": _make_sample_result("B", 1.0),
    }
    dashboard = TerminalDashboard(results_map=results)
    layout = dashboard.render_full_dashboard()

    assert isinstance(layout, Layout)
    sub_names = {sub.name for sub in layout.children}
    assert "header" in sub_names
    assert "summary" in sub_names
    assert "stats" in sub_names
    assert "failures" in sub_names


def test_tui_load_from_dir(tmp_path: Path):
    """Verify dashboard loads setup files from directory correctly."""
    r_a = _make_sample_result("A", 0.5)
    r_b = _make_sample_result("B", 1.0)

    (tmp_path / "A_heldout_20261115.json").write_text(json.dumps(r_a), encoding="utf-8")
    (tmp_path / "B_heldout_20261115.json").write_text(json.dumps(r_b), encoding="utf-8")

    dashboard = TerminalDashboard(results_dir=tmp_path)
    assert "A" in dashboard.results_map
    assert "B" in dashboard.results_map


def test_run_interactive_dashboard_flow():
    """Verify run_interactive_dashboard handles navigation choices and cleanly exits."""
    results = {
        "A": _make_sample_result("A", 0.5),
        "B": _make_sample_result("B", 1.0),
    }
    dashboard = TerminalDashboard(results_map=results)

    # Sequence of user selections: 1 (overview), 2 (stats), 3 (failures), 4 (discordance), q (exit)
    inputs = ["1", "2", "3", "4", "q"]
    mock_input = MagicMock(side_effect=inputs)
    mock_console = MagicMock(spec=Console)

    run_interactive_dashboard(
        dashboard=dashboard,
        input_provider=mock_input,
        output_console=mock_console,
    )

    assert mock_input.call_count == 5
    assert mock_console.print.called
