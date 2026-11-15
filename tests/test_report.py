"""Unit tests for comparison reporting, statistical tests, and failure analysis (F8 + F9)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolbench.report import (
    analyze_failures,
    append_heldout_audit_log,
    apply_holm_bonferroni,
    categorize_prompt_failure,
    compare_setups,
    compute_mcnemar_test,
    compute_paired_bootstrap_ci,
    render_comparison_tables,
)


def test_mcnemar_identical_outcomes():
    o1 = [True, False, True, False]
    o2 = [True, False, True, False]
    stat, p_val, method = compute_mcnemar_test(o1, o2)
    assert stat == 0.0
    assert p_val == 1.0
    assert "identical" in method


def test_mcnemar_small_sample_exact_binomial():
    # 5 discordant pairs (< 25)
    # o1 has 4 successes where o2 failed; o2 has 1 success where o1 failed
    o1 = [True] * 4 + [False] + [True] * 10
    o2 = [False] * 4 + [True] + [True] * 10
    stat, p_val, method = compute_mcnemar_test(o1, o2)
    assert stat == 5.0
    assert 0.0 < p_val < 1.0
    assert "exact binomial" in method


def test_mcnemar_large_sample_continuity_corrected():
    # 30 discordant pairs (>= 25)
    o1 = [True] * 25 + [False] * 5 + [True] * 10
    o2 = [False] * 25 + [True] * 5 + [True] * 10
    stat, p_val, method = compute_mcnemar_test(o1, o2)
    assert stat > 0.0
    assert 0.0 < p_val < 0.05
    assert "continuity-corrected" in method


def test_mcnemar_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="Prompt counts do not match"):
        compute_mcnemar_test([True], [True, False])


def test_paired_bootstrap_ci():
    o1 = [True] * 40 + [False] * 10
    o2 = [True] * 48 + [False] * 2
    delta, ci_lower, ci_upper = compute_paired_bootstrap_ci(o1, o2, num_resamples=500, seed=42)
    assert delta == pytest.approx(0.16)
    assert ci_lower <= delta <= ci_upper
    assert ci_lower > 0.0  # Significant difference


def test_paired_bootstrap_ci_empty():
    assert compute_paired_bootstrap_ci([], []) == (0.0, 0.0, 0.0)


def test_apply_holm_bonferroni():
    # m = 3 tests:
    # raw: [0.01, 0.04, 0.03]
    # sorted: 0.01 (rank 0, mult 3 -> 0.03)
    #         0.03 (rank 1, mult 2 -> 0.06)
    #         0.04 (rank 2, mult 1 -> 0.04 -> running max 0.06)
    raw = [0.01, 0.04, 0.03]
    adj = apply_holm_bonferroni(raw)
    assert adj == [pytest.approx(0.03), pytest.approx(0.06), pytest.approx(0.06)]

    # Cap at 1.0
    assert apply_holm_bonferroni([0.6, 0.8]) == [1.0, 1.0]
    assert apply_holm_bonferroni([]) == []


def test_categorize_prompt_failure_priority():
    # 1. Error
    assert categorize_prompt_failure({"error": "Timeout", "tool_correct": False}) == "error"

    # 2. Retrieval miss (retrieval setup)
    assert (
        categorize_prompt_failure(
            {
                "expected_tool": "fs_read",
                "exposed_tools": ["git_status", "time_now"],
                "retrieval_hit": False,
            },
            is_retrieval_setup=True,
        )
        == "retrieval_miss"
    )

    # Retrieval miss NOT triggered if non-retrieval setup
    assert (
        categorize_prompt_failure(
            {
                "expected_tool": "fs_read",
                "exposed_tools": ["git_status"],
                "chosen_tool": "git_status",
                "tool_correct": False,
            },
            is_retrieval_setup=False,
        )
        == "confusion"
    )

    # 3. No tool call
    assert (
        categorize_prompt_failure({"chosen_tool": None, "text_response": "I cannot answer this."})
        == "no_tool_call"
    )

    # 4. Malformed output
    assert (
        categorize_prompt_failure(
            {"chosen_tool": None, "text_response": None, "raw_tool_call_malformed": True}
        )
        == "malformed_output"
    )

    # 5. Confusion
    assert (
        categorize_prompt_failure({"chosen_tool": "wrong_tool", "tool_correct": False})
        == "confusion"
    )

    # 6. Invalid arguments post-retry
    assert (
        categorize_prompt_failure(
            {
                "chosen_tool": "correct_tool",
                "tool_correct": True,
                "args_valid_post_retry": False,
            }
        )
        == "invalid_arguments_unrecovered"
    )

    # 7. Wrong arguments (valid schema, but strict match failed)
    assert (
        categorize_prompt_failure(
            {
                "chosen_tool": "correct_tool",
                "tool_correct": True,
                "args_valid_post_retry": True,
                "args_strict_match": False,
            }
        )
        == "wrong_arguments"
    )

    # 8. Pass
    assert (
        categorize_prompt_failure(
            {
                "chosen_tool": "correct_tool",
                "tool_correct": True,
                "args_valid_post_retry": True,
                "args_strict_match": True,
            }
        )
        == "pass"
    )


def test_analyze_failures():
    result_data = {
        "setup": "C",
        "results": [
            {
                "prompt_id": "p1",
                "chosen_tool": "t1",
                "tool_correct": True,
                "args_valid_post_retry": True,
                "args_strict_match": True,
            },
            {
                "prompt_id": "p2",
                "expected_tool": "t1",
                "exposed_tools": ["t2"],
                "retrieval_hit": False,
            },
            {
                "prompt_id": "p3",
                "chosen_tool": "t2",
                "tool_correct": False,
            },
        ],
    }
    analysis = analyze_failures(result_data)
    assert analysis["total_prompts"] == 3
    assert analysis["category_counts"]["pass"] == 1
    assert analysis["category_counts"]["retrieval_miss"] == 1
    assert analysis["category_counts"]["confusion"] == 1
    assert analysis["category_percentages"]["pass"] == pytest.approx(33.333, abs=0.01)


def test_compare_setups_and_render(tmp_path: Path):
    h_hash = "abc123456789deadbeef"
    # Build 4 mock setups
    results_map: dict[str, dict] = {}
    for letter, acc in [("A", 0.6), ("B", 0.7), ("C", 0.8), ("D", 0.9)]:
        items = []
        for i in range(50):
            is_correct = i < int(50 * acc)
            items.append(
                {
                    "prompt_id": f"prompt_{i:03d}",
                    "expected_tool": "tool_x",
                    "chosen_tool": "tool_x" if is_correct else "tool_y",
                    "tool_correct": is_correct,
                    "args_valid_post_retry": is_correct,
                    "args_strict_match": is_correct,
                }
            )
        results_map[letter] = {
            "setup": letter,
            "heldout_hash": h_hash,
            "metrics": {"tool_accuracy": acc},
            "results": items,
        }

    report = compare_setups(results_map, output_dir=tmp_path)

    assert report["heldout_hash"] == h_hash
    assert len(report["primary_comparisons"]) == 3
    assert len(report["exploratory_comparisons"]) == 3

    # Primary comparisons should have Holm-adjusted p-values
    for comp in report["primary_comparisons"]:
        assert "adjusted_p_value" in comp
        assert comp["adjusted_p_value"] >= comp["raw_p_value"]

    # Render without errors
    render_comparison_tables(report)


def test_compare_setups_hash_mismatch_raises():
    results_map = {
        "A": {"setup": "A", "heldout_hash": "hash_1", "results": []},
        "B": {"setup": "B", "heldout_hash": "hash_2", "results": []},
    }
    with pytest.raises(ValueError, match="Held-out dataset hash mismatch"):
        compare_setups(results_map)


def test_append_heldout_audit_log(tmp_path: Path):
    log_file = tmp_path / "heldout_log.jsonl"
    record = {
        "run_id": "run-001",
        "setup": "A",
        "model": "gemini-3.5-flash-lite",
        "heldout_hash": "hash_abc",
        "prompt_count": 50,
        "tool_accuracy": 0.84,
        "args_strict_match_rate": 0.78,
        "args_valid_post_retry_rate": 0.92,
    }
    append_heldout_audit_log(record, log_file=log_file)

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["run_id"] == "run-001"
    assert loaded["tool_accuracy"] == 0.84
