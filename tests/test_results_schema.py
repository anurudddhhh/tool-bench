"""Unit tests for formal benchmark results and comparison schemas (F11f)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from toolbench.results_schema import (
    BenchmarkResultFile,
    ComparisonReportFile,
    PromptResultRecord,
    RunMetricsRecord,
    validate_benchmark_result_file,
    validate_comparison_report_file,
    validate_results_file,
)


def test_prompt_result_record_validation():
    rec = PromptResultRecord(
        prompt_id="prompt_001",
        prompt="List directory contents",
        expected_tool="filesystem_list_directory",
        chosen_tool="filesystem_list_directory",
        tool_correct=True,
        args_valid_pre_retry=True,
        args_valid_post_retry=True,
        args_strict_match=True,
        args_match_score=1.0,
    )
    assert rec.prompt_id == "prompt_001"
    assert rec.tool_correct is True

    # Test match score out of bounds
    with pytest.raises(ValidationError):
        PromptResultRecord(
            prompt_id="p",
            prompt="q",
            expected_tool="t",
            args_match_score=1.5,
        )


def test_run_metrics_record_validation():
    metrics = RunMetricsRecord(
        tool_accuracy=0.85,
        args_valid_pre_retry_rate=0.75,
        args_valid_post_retry_rate=0.90,
        args_strict_match_rate=0.80,
        recovery_rate=0.60,
        args_match_score_mean=0.88,
        total_prompts=50,
        avg_latency_ms=120.5,
        cache_hit_rate=0.20,
    )
    assert metrics.tool_accuracy == 0.85

    # Out of bounds rate
    with pytest.raises(ValidationError):
        RunMetricsRecord(
            tool_accuracy=1.2,
            args_valid_pre_retry_rate=0.75,
            args_valid_post_retry_rate=0.90,
            args_strict_match_rate=0.80,
            recovery_rate=0.60,
            args_match_score_mean=0.88,
            total_prompts=50,
            avg_latency_ms=10.0,
            cache_hit_rate=0.0,
        )


def test_validate_benchmark_result_file(tmp_path: Path):
    data = {
        "setup": "A",
        "run_id": "run-12345",
        "model": "gemini-3.5-flash-lite",
        "timestamp": "2026-11-15T12:00:00Z",
        "evaluation_temperature": 0.0,
        "heldout_hash": "deadbeef1234",
        "metrics": {
            "tool_accuracy": 0.80,
            "args_valid_pre_retry_rate": 0.70,
            "args_valid_post_retry_rate": 0.85,
            "args_strict_match_rate": 0.75,
            "recovery_rate": 0.50,
            "args_match_score_mean": 0.80,
            "total_prompts": 2,
            "avg_latency_ms": 150.0,
            "cache_hit_rate": 0.5,
        },
        "results": [
            {
                "prompt_id": "p1",
                "prompt": "Find a file",
                "expected_tool": "fs_find",
                "chosen_tool": "fs_find",
                "tool_correct": True,
                "args_valid_pre_retry": True,
                "args_valid_post_retry": True,
                "args_strict_match": True,
            },
            {
                "prompt_id": "p2",
                "prompt": "Git commit",
                "expected_tool": "git_commit",
                "chosen_tool": "git_status",
                "tool_correct": False,
                "args_valid_pre_retry": False,
                "args_valid_post_retry": False,
                "args_strict_match": False,
            },
        ],
    }

    # Test from dictionary
    validated = validate_benchmark_result_file(data)
    assert validated.setup == "A"
    assert len(validated.results) == 2

    # Test from file path
    f_path = tmp_path / "result_A.json"
    f_path.write_text(json.dumps(data), encoding="utf-8")

    loaded = validate_benchmark_result_file(f_path)
    assert loaded.run_id == "run-12345"

    # Test not found
    with pytest.raises(FileNotFoundError):
        validate_benchmark_result_file("nonexistent.json")


def test_validate_comparison_report_file(tmp_path: Path):
    data = {
        "timestamp": "20261115_120000",
        "heldout_hash": "deadbeef1234",
        "setups_evaluated": ["A", "B", "C", "D"],
        "metrics_summary": {"A": {"tool_accuracy": 0.70}},
        "primary_comparisons": [
            {
                "comparison": "A vs C",
                "setup_1": "A",
                "setup_2": "C",
                "n_prompts": 50,
                "accuracy_1": 0.70,
                "accuracy_2": 0.84,
                "delta": 0.14,
                "ci_95_lower": 0.04,
                "ci_95_upper": 0.24,
                "statistic": 8.5,
                "raw_p_value": 0.0035,
                "adjusted_p_value": 0.0105,
                "method": "exact binomial test",
                "is_primary": True,
            }
        ],
        "exploratory_comparisons": [],
        "failure_analyses": {
            "A": {
                "setup": "A",
                "total_prompts": 50,
                "category_counts": {"pass": 35, "confusion": 15},
                "category_percentages": {"pass": 70.0, "confusion": 30.0},
                "prompt_breakdown": [],
            }
        },
        "multiple_comparisons_note": "Holm-Bonferroni correction note.",
    }

    validated = validate_comparison_report_file(data)
    assert validated.heldout_hash == "deadbeef1234"
    assert len(validated.primary_comparisons) == 1

    f_path = tmp_path / "comparison.json"
    f_path.write_text(json.dumps(data), encoding="utf-8")

    loaded = validate_comparison_report_file(f_path)
    assert loaded.setups_evaluated == ["A", "B", "C", "D"]


def test_validate_results_file_auto_detection(tmp_path: Path):
    bench_data = {
        "setup": "B",
        "run_id": "run-b",
        "model": "gemini-3.5-flash-lite",
        "timestamp": "2026-11-15T12:00:00Z",
        "metrics": {
            "tool_accuracy": 0.75,
            "args_valid_pre_retry_rate": 0.70,
            "args_valid_post_retry_rate": 0.85,
            "args_strict_match_rate": 0.75,
            "recovery_rate": 0.50,
            "args_match_score_mean": 0.80,
            "total_prompts": 1,
            "avg_latency_ms": 100.0,
            "cache_hit_rate": 0.0,
        },
        "results": [
            {
                "prompt_id": "p1",
                "prompt": "Read file",
                "expected_tool": "fs_read",
                "chosen_tool": "fs_read",
                "tool_correct": True,
                "args_valid_pre_retry": True,
                "args_valid_post_retry": True,
                "args_strict_match": True,
            }
        ],
    }

    comp_data = {
        "timestamp": "20261115_120000",
        "setups_evaluated": ["A", "B"],
        "metrics_summary": {},
        "primary_comparisons": [],
        "exploratory_comparisons": [],
        "failure_analyses": {},
        "multiple_comparisons_note": "Note",
    }

    b_file = tmp_path / "bench.json"
    c_file = tmp_path / "comp.json"
    u_file = tmp_path / "unknown.json"
    arr_file = tmp_path / "array.json"

    b_file.write_text(json.dumps(bench_data), encoding="utf-8")
    c_file.write_text(json.dumps(comp_data), encoding="utf-8")
    u_file.write_text(json.dumps({"unrelated": 123}), encoding="utf-8")
    arr_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    kind_b, model_b = validate_results_file(b_file)
    assert kind_b == "benchmark"
    assert isinstance(model_b, BenchmarkResultFile)

    kind_c, model_c = validate_results_file(c_file)
    assert kind_c == "comparison"
    assert isinstance(model_c, ComparisonReportFile)

    with pytest.raises(ValueError, match="Unable to recognize schema"):
        validate_results_file(u_file)

    with pytest.raises(ValueError, match="expected JSON object"):
        validate_results_file(arr_file)

    with pytest.raises(FileNotFoundError):
        validate_results_file("missing_file.json")
