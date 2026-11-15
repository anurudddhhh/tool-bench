"""Unit tests for terminal output and presentation formatting (F11e)."""

from typing import Any

from rich.console import Console

from toolbench.output import (
    create_progress,
    print_error,
    print_info,
    print_prompt_panel,
    print_run_summary,
    print_success,
    print_tools_table,
    print_warning,
    render_prompt_panel,
    render_run_summary,
    render_tools_table,
)


def test_status_printing() -> None:
    """Verify status banners print appropriate icons and text."""
    test_console = Console(record=True, color_system=None)

    print_success("Operation completed successfully", target_console=test_console)
    print_warning("Potential ceiling effect detected", target_console=test_console)
    print_error("Failed to connect to MCP server", target_console=test_console)
    print_info("Processing prompt 10 of 50", target_console=test_console)

    output = test_console.export_text()
    assert "✓ Operation completed successfully" in output
    assert "⚠ Potential ceiling effect detected" in output
    assert "✗ Failed to connect to MCP server" in output
    assert "ℹ Processing prompt 10 of 50" in output


def test_render_tools_table() -> None:
    """Verify tool table formatting and column decomposition."""
    tools: list[dict[str, Any]] = [
        {
            "name": "filesystem__read_file",
            "description": "Reads full content of a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
        {
            "name": "standalone_tool",
            "description": "A tool with no explicit server namespace",
            "parameters": {},
        },
    ]

    table = render_tools_table(tools, title="Test Tools")
    assert table.title == "Test Tools"
    assert len(table.columns) == 4

    test_console = Console(record=True, color_system=None)
    print_tools_table(tools, target_console=test_console)
    text = test_console.export_text()

    assert "filesystem" in text
    assert "read_file" in text
    assert "standalone_tool" in text
    assert "Reads full content" in text


def test_render_prompt_panel() -> None:
    """Verify prompt review panel rendering."""
    prompt_id = 42
    prompt_text = "What files exist in the current directory?"
    expected_tool = "filesystem__list_directory"
    expected_args = {"path": "."}
    description = "List entries in a folder"

    panel = render_prompt_panel(
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        expected_tool=expected_tool,
        expected_args=expected_args,
        description=description,
    )
    assert panel is not None

    test_console = Console(record=True, color_system=None)
    print_prompt_panel(
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        expected_tool=expected_tool,
        expected_args=expected_args,
        description=description,
        target_console=test_console,
    )
    output = test_console.export_text()

    assert "Prompt #42" in output
    assert "What files exist in the current directory?" in output
    assert "filesystem__list_directory" in output
    assert '"path": "."' in output
    assert "List entries in a folder" in output


def test_render_run_summary() -> None:
    """Verify evaluation run summary table metric computations."""
    results: list[dict[str, Any]] = [
        {"tool_correct": True, "args_strict_match": True, "cached": True, "latency_ms": 10.0},
        {"tool_correct": True, "args_strict_match": False, "cached": False, "latency_ms": 200.0},
        {"tool_correct": False, "args_strict_match": False, "cached": False, "latency_ms": 300.0},
        {"tool_correct": True, "args_strict_match": True, "cached": True, "latency_ms": 50.0},
    ]

    table = render_run_summary(results, title="Setup A Baseline")
    assert table.title == "Setup A Baseline"

    test_console = Console(record=True, color_system=None)
    print_run_summary(results, target_console=test_console)
    output = test_console.export_text()

    assert "Total Prompts Evaluated" in output
    assert "4" in output
    # 3/4 = 75.0%
    assert "75.0% (3/4)" in output
    # 2/4 = 50.0%
    assert "50.0% (2/4)" in output
    assert "Cache Hits" in output
    assert "2/4" in output
    # Avg latency = (10 + 200 + 300 + 50) / 4 = 140.0 ms
    assert "140.0 ms" in output


def test_render_run_summary_empty() -> None:
    """Verify summary table handles empty results safely without ZeroDivisionError."""
    table = render_run_summary([], title="Empty Results")
    assert table.title == "Empty Results"

    test_console = Console(record=True, color_system=None)
    print_run_summary([], target_console=test_console)
    output = test_console.export_text()

    assert "Total Prompts Evaluated" in output
    assert "0.0% (0/0)" in output


def test_create_progress() -> None:
    """Verify progress bar creation."""
    prog = create_progress()
    assert prog is not None
    assert len(prog.columns) >= 5
