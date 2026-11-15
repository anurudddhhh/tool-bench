"""Unit tests for the Toolbench unified Command-Line Interface (F11g/CLI)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from toolbench.cli import app
from toolbench.mcp_client import ToolInfo

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Scientific MCP tool selection benchmarking and evaluation CLI." in result.stdout
    assert "list-tools" in result.stdout
    assert "generate" in result.stdout
    assert "pilot" in result.stdout
    assert "review" in result.stdout
    assert "split" in result.stdout
    assert "select-k" in result.stdout
    assert "rewrite-descriptions" in result.stdout
    assert "run" in result.stdout
    assert "compare" in result.stdout
    assert "analyze-failures" in result.stdout
    assert "validate" in result.stdout


def test_cli_validate_command_success(tmp_path: Path):
    valid_result = {
        "setup": "A",
        "run_id": "test-uuid",
        "model": "mock-model",
        "timestamp": "2026-11-15T12:00:00Z",
        "evaluation_temperature": 0.0,
        "metrics": {
            "tool_accuracy": 0.8,
            "args_valid_pre_retry_rate": 0.8,
            "args_valid_post_retry_rate": 0.8,
            "args_strict_match_rate": 0.8,
            "recovery_rate": 0.0,
            "args_match_score_mean": 0.8,
            "total_prompts": 1,
            "avg_latency_ms": 1.0,
            "cache_hit_rate": 0.0,
        },
        "results": [
            {
                "prompt_id": "1",
                "prompt": "Test query",
                "expected_tool": "t1",
                "chosen_tool": "t1",
                "tool_correct": True,
                "args_valid_pre_retry": True,
                "args_valid_post_retry": True,
                "args_strict_match": True,
            }
        ],
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(valid_result), encoding="utf-8")

    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 0
    assert "valid schema contract" in result.stdout


def test_cli_validate_command_failure(tmp_path: Path):
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text(json.dumps({"unknown": "data"}), encoding="utf-8")

    result = runner.invoke(app, ["validate", str(invalid_file)])
    assert result.exit_code == 1
    assert "Validation failed" in result.stdout


def test_cli_analyze_failures_command(tmp_path: Path):
    res_data = {
        "setup": "C",
        "results": [
            {
                "prompt_id": "1",
                "expected_tool": "t1",
                "exposed_tools": ["t2"],
                "retrieval_hit": False,
            },
            {
                "prompt_id": "2",
                "chosen_tool": "t1",
                "tool_correct": True,
                "args_valid_post_retry": True,
                "args_strict_match": True,
            },
        ],
    }
    p = tmp_path / "res_c.json"
    p.write_text(json.dumps(res_data), encoding="utf-8")

    result = runner.invoke(app, ["analyze-failures", str(p)])
    assert result.exit_code == 0
    assert "Failure Mode Analysis for Setup C" in result.stdout
    assert "retrieval_miss" in result.stdout
    assert "pass" in result.stdout


def test_cli_run_command_invalid_setup():
    result = runner.invoke(app, ["run", "--setup", "Z"])
    assert result.exit_code == 1
    assert "Invalid setup 'Z'" in result.stdout


def test_cli_list_tools_command():
    sample_tools = [
        ToolInfo(
            name="fs_read",
            description="Reads files",
            input_schema={"type": "object", "properties": {}},
        )
    ]
    with patch("toolbench.cli._load_tools_from_servers", return_value=sample_tools):
        result = runner.invoke(app, ["list-tools"])
        assert result.exit_code == 0
        assert "fs_read" in result.stdout
        assert "Discovered Tools (1 total)" in result.stdout


def test_cli_rewrite_descriptions_command(tmp_path: Path):
    sample_tools = [
        ToolInfo(
            name="git_log",
            description="View commits",
            input_schema={"type": "object", "properties": {}},
        )
    ]
    out_file = tmp_path / "rewritten.json"
    with patch("toolbench.cli._load_tools_from_servers", return_value=sample_tools):
        result = runner.invoke(
            app,
            ["rewrite-descriptions", "--mock", "--output", str(out_file)],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        assert "Rewrote descriptions for 1 tools" in result.stdout
