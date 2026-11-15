"""Comparison reporting, statistical significance tests, and failure analysis (F8 + F9).

Provides paired statistical tests (McNemar exact binomial / continuity-corrected,
paired bootstrap confidence intervals, and Holm-Bonferroni correction), structured
failure mode categorization, and audit logging to results/heldout_log.jsonl.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy import stats

from toolbench.output import print_info, print_success, print_warning

logger = logging.getLogger(__name__)
console = Console()

MULTIPLE_COMPARISONS_EXPLANATION = (
    "When you test multiple hypotheses, the chance of finding a 'significant' result "
    "by luck increases. If you test 3 things at p<0.05, the actual false-positive rate "
    "is about 14%, not 5%. Holm-Bonferroni adjusts for this."
)


def compute_mcnemar_test(
    outcomes_1: list[bool],
    outcomes_2: list[bool],
) -> tuple[float, float, str]:
    """Perform McNemar's test for paired binary outcomes.

    Args:
        outcomes_1: Boolean pass/correct indicators for setup 1 across prompts.
        outcomes_2: Boolean pass/correct indicators for setup 2 across identical prompts.

    Returns:
        tuple of (test_statistic, p_value, test_method_used).
    """
    if len(outcomes_1) != len(outcomes_2):
        raise ValueError(
            f"Prompt counts do not match for paired test: {len(outcomes_1)} vs {len(outcomes_2)}"
        )

    # b: 1 correct, 2 incorrect; c: 1 incorrect, 2 correct
    b = sum(1 for o1, o2 in zip(outcomes_1, outcomes_2, strict=True) if o1 and not o2)
    c = sum(1 for o1, o2 in zip(outcomes_1, outcomes_2, strict=True) if not o1 and o2)
    discordant = b + c

    if discordant == 0:
        return 0.0, 1.0, "identical (no discordant pairs)"

    if discordant < 25:
        # Exact two-sided binomial test under H0: p = 0.5
        result = stats.binomtest(min(b, c), n=discordant, p=0.5, alternative="two-sided")
        return float(discordant), float(result.pvalue), "exact binomial test (discordant < 25)"

    # Edwards' continuity correction for McNemar: (|b - c| - 1)^2 / (b + c)
    statistic = (abs(b - c) - 1.0) ** 2 / discordant
    p_value = float(stats.chi2.sf(statistic, df=1))
    return float(statistic), p_value, "continuity-corrected chi-square (discordant >= 25)"


def compute_paired_bootstrap_ci(
    outcomes_1: list[bool],
    outcomes_2: list[bool],
    num_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap 95% confidence interval on paired accuracy difference (Acc2 - Acc1).

    Args:
        outcomes_1: Boolean outcomes for setup 1.
        outcomes_2: Boolean outcomes for setup 2.
        num_resamples: Number of bootstrap iterations.
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for deterministic reproducibility.

    Returns:
        tuple of (mean_diff, ci_lower, ci_upper).
    """
    arr1 = np.asarray(outcomes_1, dtype=np.float64)
    arr2 = np.asarray(outcomes_2, dtype=np.float64)
    n = len(arr1)
    if n == 0:
        return 0.0, 0.0, 0.0

    observed_diff = float(np.mean(arr2) - np.mean(arr1))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(num_resamples, n))

    resampled_diffs = np.mean(arr2[indices] - arr1[indices], axis=1)

    ci_lower = float(np.percentile(resampled_diffs, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(resampled_diffs, 100 * (1 - alpha / 2)))

    return observed_diff, ci_lower, ci_upper


def apply_holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply step-down Holm-Bonferroni correction to a list of p-values.

    Args:
        p_values: List of raw p-values.

    Returns:
        List of adjusted p-values matching input order.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values ascending while keeping track of original indices
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted: list[tuple[int, float]] = []

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        adj_p = min(1.0, p_val * multiplier)
        running_max = max(running_max, adj_p)
        adjusted.append((orig_idx, running_max))

    # Restore original ordering
    adjusted.sort(key=lambda x: x[0])
    return [p for _, p in adjusted]


def categorize_prompt_failure(
    item: dict[str, Any],
    is_retrieval_setup: bool = False,
) -> str:
    """Categorize the outcome or failure mode of a single evaluated prompt (F9).

    Failure categories (priority order, first match wins):
    1. error: LLM exception, timeout, rate limit, etc.
    2. retrieval_miss (C/D only): Expected tool was not in the exposed top-k list.
    3. no_tool_call: LLM returned text but no tool call was emitted.
    4. malformed_output: LLM output could not be parsed as a tool call.
    5. confusion: Expected tool was exposed, but LLM picked an incorrect tool.
    6. invalid_arguments_unrecovered: Tool was correct, but arguments failed schema validation.
    7. wrong_arguments: Tool & schema valid, but acceptable_args strict match failed.
    8. pass: tool_correct AND args_valid_post_retry AND args_strict_match.

    Args:
        item: Single prompt evaluation result dict.
        is_retrieval_setup: True if setup C or D.

    Returns:
        Category string identifier.
    """
    error = item.get("error")
    if error:
        return "error"

    # Retrieval miss check (C and D setups)
    if is_retrieval_setup:
        # Check if retrieval explicitly missed the expected tool
        retrieval_hit = item.get("retrieval_hit")
        exposed_tools = item.get("exposed_tools", [])
        expected_tool = item.get("expected_tool")
        if retrieval_hit is False or (
            exposed_tools and expected_tool and expected_tool not in exposed_tools
        ):
            return "retrieval_miss"

    chosen_tool = item.get("chosen_tool")
    text_response = item.get("text_response")

    if not chosen_tool:
        # Check if it was malformed or pure conversational refusal
        if item.get("raw_tool_call_malformed", False):
            return "malformed_output"
        if text_response:
            return "no_tool_call"
        return "malformed_output"

    tool_correct = bool(item.get("tool_correct", False))
    if not tool_correct:
        return "confusion"

    args_valid = bool(item.get("args_valid_post_retry", False))
    if not args_valid:
        return "invalid_arguments_unrecovered"

    args_strict_match = bool(item.get("args_strict_match", False))
    if not args_strict_match:
        return "wrong_arguments"

    return "pass"


def analyze_failures(
    result_data: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate failure categories and counts across an entire evaluation run (F9).

    Args:
        result_data: Benchmark results dictionary (Setup A/B/C/D).

    Returns:
        Dictionary containing category breakdown counts, percentages, and details.
    """
    setup = result_data.get("setup", "A")
    is_retrieval = setup.upper() in {"C", "D"}
    items = result_data.get("results", [])

    categories = [
        "pass",
        "retrieval_miss",
        "no_tool_call",
        "malformed_output",
        "confusion",
        "invalid_arguments_unrecovered",
        "wrong_arguments",
        "error",
    ]

    counts: dict[str, int] = {c: 0 for c in categories}
    categorized_items: list[dict[str, Any]] = []

    for item in items:
        cat = categorize_prompt_failure(item, is_retrieval_setup=is_retrieval)
        counts[cat] = counts.get(cat, 0) + 1
        categorized_items.append(
            {
                "prompt_id": item.get("prompt_id"),
                "expected_tool": item.get("expected_tool"),
                "chosen_tool": item.get("chosen_tool"),
                "category": cat,
                "error": item.get("error"),
            }
        )

    total = len(items)
    percentages = {
        cat: (cnt / total * 100.0) if total > 0 else 0.0 for cat, cnt in counts.items()
    }

    return {
        "setup": setup,
        "total_prompts": total,
        "category_counts": counts,
        "category_percentages": percentages,
        "prompt_breakdown": categorized_items,
    }


def compare_setups(
    results_map: dict[str, dict[str, Any] | Path | str],
    output_dir: Path | str = "results",
) -> dict[str, Any]:
    """Compare evaluation outcomes across Setups A, B, C, D (F8).

    Validates held-out hash equivalence, computes McNemar's tests with Holm-Bonferroni
    correction for primary comparisons, calculates bootstrap 95% CIs, and emits
    the final comparison JSON and Rich table.

    Args:
        results_map: Dictionary mapping setup letter ("A", "B", "C", "D") to result data/path.
        output_dir: Destination directory for comparison JSON.

    Returns:
        Full comparison report dictionary.
    """
    # Load all result structures
    loaded_results: dict[str, dict[str, Any]] = {}
    for key, val in results_map.items():
        if isinstance(val, (str, Path)):
            p = Path(val)
            if not p.exists():
                print_warning(f"Result file for Setup {key} not found at '{p}'. Skipping.")
                continue
            loaded_results[key.upper()] = json.loads(p.read_text(encoding="utf-8"))
        else:
            loaded_results[key.upper()] = val

    if not loaded_results:
        raise ValueError("No valid results provided for comparison.")

    # Validate held-out hash consistency
    hashes = {
        k: res.get("heldout_hash") for k, res in loaded_results.items() if res.get("heldout_hash")
    }
    if len(set(hashes.values())) > 1:
        raise ValueError(
            f"Held-out dataset hash mismatch across setups: {hashes}. Cannot perform paired tests."
        )

    # Extract paired outcomes
    # Build prompt_id -> tool_correct mapping per setup
    prompt_outcomes: dict[str, dict[str, bool]] = {}
    metrics_summary: dict[str, dict[str, Any]] = {}

    for setup_key, res in loaded_results.items():
        outcomes: dict[str, bool] = {}
        for item in res.get("results", []):
            p_id = item.get("prompt_id")
            if p_id:
                outcomes[p_id] = bool(item.get("tool_correct", False))
        prompt_outcomes[setup_key] = outcomes
        metrics_summary[setup_key] = res.get("metrics", {})

    # Define comparison pairs
    primary_pairs = [("A", "C"), ("A", "B"), ("A", "D")]
    exploratory_pairs = [("B", "C"), ("B", "D"), ("C", "D")]

    # Compute raw p-values for primary pairs
    primary_results: list[dict[str, Any]] = []
    primary_raw_p: list[float] = []

    for s1, s2 in primary_pairs:
        if s1 not in prompt_outcomes or s2 not in prompt_outcomes:
            continue
        common_ids = sorted(
            set(prompt_outcomes[s1].keys()).intersection(prompt_outcomes[s2].keys())
        )
        if not common_ids:
            continue

        o1 = [prompt_outcomes[s1][pid] for pid in common_ids]
        o2 = [prompt_outcomes[s2][pid] for pid in common_ids]

        stat, p_val, method = compute_mcnemar_test(o1, o2)
        diff, ci_l, ci_u = compute_paired_bootstrap_ci(o1, o2)

        primary_raw_p.append(p_val)
        primary_results.append(
            {
                "comparison": f"{s1} vs {s2}",
                "setup_1": s1,
                "setup_2": s2,
                "n_prompts": len(common_ids),
                "accuracy_1": float(np.mean(o1)),
                "accuracy_2": float(np.mean(o2)),
                "delta": diff,
                "ci_95_lower": ci_l,
                "ci_95_upper": ci_u,
                "statistic": stat,
                "raw_p_value": p_val,
                "method": method,
                "is_primary": True,
            }
        )

    # Apply Holm-Bonferroni correction to primary comparisons
    adj_p_values = apply_holm_bonferroni(primary_raw_p)
    for entry, adj_p in zip(primary_results, adj_p_values, strict=True):
        entry["adjusted_p_value"] = adj_p

    # Compute exploratory pairs
    exploratory_results: list[dict[str, Any]] = []
    for s1, s2 in exploratory_pairs:
        if s1 not in prompt_outcomes or s2 not in prompt_outcomes:
            continue
        common_ids = sorted(
            set(prompt_outcomes[s1].keys()).intersection(prompt_outcomes[s2].keys())
        )
        if not common_ids:
            continue

        o1 = [prompt_outcomes[s1][pid] for pid in common_ids]
        o2 = [prompt_outcomes[s2][pid] for pid in common_ids]

        stat, p_val, method = compute_mcnemar_test(o1, o2)
        diff, ci_l, ci_u = compute_paired_bootstrap_ci(o1, o2)

        exploratory_results.append(
            {
                "comparison": f"{s1} vs {s2}",
                "setup_1": s1,
                "setup_2": s2,
                "n_prompts": len(common_ids),
                "accuracy_1": float(np.mean(o1)),
                "accuracy_2": float(np.mean(o2)),
                "delta": diff,
                "ci_95_lower": ci_l,
                "ci_95_upper": ci_u,
                "statistic": stat,
                "raw_p_value": p_val,
                "method": method,
                "is_primary": False,
                "note": "exploratory — not corrected for multiple comparisons",
            }
        )

    # Failure analysis per setup
    failure_analyses: dict[str, Any] = {}
    for setup_key, res in loaded_results.items():
        failure_analyses[setup_key] = analyze_failures(res)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "heldout_hash": next(iter(hashes.values())) if hashes else None,
        "setups_evaluated": list(loaded_results.keys()),
        "metrics_summary": metrics_summary,
        "primary_comparisons": primary_results,
        "exploratory_comparisons": exploratory_results,
        "failure_analyses": failure_analyses,
        "multiple_comparisons_note": MULTIPLE_COMPARISONS_EXPLANATION,
    }

    # Save to disk
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"comparison_{timestamp}.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_success(f"Saved comparison report to '{report_file}'.")

    return report


def render_comparison_tables(report: dict[str, Any]) -> None:
    """Print Rich formatted tables of benchmark comparison results to terminal."""
    console.print("\n[bold cyan]Factorial Setup Comparison (Primary Outcome: Tool Accuracy)[/]")
    console.print(f"[dim]{MULTIPLE_COMPARISONS_EXPLANATION}[/]\n")

    # Primary table
    table = Table(title="Primary Comparisons (Holm-Bonferroni Corrected)", show_lines=True)
    table.add_column("Comparison", style="cyan", no_wrap=True)
    table.add_column("Acc 1", justify="right")
    table.add_column("Acc 2", justify="right")
    table.add_column("Delta (Acc2 - Acc1)", justify="right")
    table.add_column("95% Bootstrap CI", justify="center")
    table.add_column("Raw p", justify="right")
    table.add_column("Holm Adj p", justify="right", style="bold green")

    for p in report.get("primary_comparisons", []):
        table.add_row(
            p["comparison"],
            f"{p['accuracy_1']:.2%}",
            f"{p['accuracy_2']:.2%}",
            f"{p['delta']:+.2%}",
            f"[{p['ci_95_lower']:+.2%}, {p['ci_95_upper']:+.2%}]",
            f"{p['raw_p_value']:.4f}",
            f"{p['adjusted_p_value']:.4f}",
        )
    console.print(table)

    # Exploratory table
    exp_list = report.get("exploratory_comparisons", [])
    if exp_list:
        exp_table = Table(
            title="Exploratory Comparisons (Unadjusted, Hypothesis-Generating)", show_lines=True
        )
        exp_table.add_column("Comparison", style="cyan", no_wrap=True)
        exp_table.add_column("Acc 1", justify="right")
        exp_table.add_column("Acc 2", justify="right")
        exp_table.add_column("Delta (Acc2 - Acc1)", justify="right")
        exp_table.add_column("95% Bootstrap CI", justify="center")
        exp_table.add_column("Raw p", justify="right")

        for e in exp_list:
            exp_table.add_row(
                e["comparison"],
                f"{e['accuracy_1']:.2%}",
                f"{e['accuracy_2']:.2%}",
                f"{e['delta']:+.2%}",
                f"[{e['ci_95_lower']:+.2%}, {e['ci_95_upper']:+.2%}]",
                f"{e['raw_p_value']:.4f}",
            )
        console.print(exp_table)


def append_heldout_audit_log(
    record: dict[str, Any],
    log_file: Path | str = "results/heldout_log.jsonl",
) -> None:
    """Append a single run record to the held-out audit log (F8 append-only audit trail).

    Args:
        record: Audit summary record dictionary.
        log_file: Path to audit log JSONL file.
    """
    f_path = Path(log_file)
    f_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "run_id": record.get("run_id"),
        "timestamp": record.get("timestamp")
        or datetime.now(timezone.utc).isoformat(),
        "setup": record.get("setup"),
        "model": record.get("model"),
        "heldout_hash": record.get("heldout_hash"),
        "prompt_count": record.get("prompt_count", 0),
        "tool_accuracy": record.get("tool_accuracy", 0.0),
        "args_strict_match_rate": record.get("args_strict_match_rate", 0.0),
        "args_valid_post_retry_rate": record.get("args_valid_post_retry_rate", 0.0),
    }

    with open(f_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print_info(f"Appended run audit entry to '{f_path}'.")
