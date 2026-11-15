"""Formal Pydantic schemas and validation utilities for benchmark results (F11f).

Defines explicit, typed contracts for:
- Per-prompt execution and validation records.
- Run-level aggregate evaluation metrics.
- Benchmark run result files (Setups A, B, C, D).
- Statistical pairwise comparison reports.
- Structured failure analysis breakdowns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PromptResultRecord(BaseModel):
    """Execution and evaluation details for a single prompt."""

    model_config = ConfigDict(extra="ignore")

    prompt_id: str | int = Field(description="Unique prompt identifier")
    prompt: str = Field(description="User prompt text")
    expected_tool: str = Field(description="Ground truth target tool")
    chosen_tool: str | None = Field(default=None, description="Tool selected by the LLM")
    expected_args: dict[str, Any] = Field(default_factory=dict, description="Expected arguments")
    chosen_args: dict[str, Any] = Field(
        default_factory=dict, description="Arguments emitted by LLM"
    )
    tool_correct: bool = Field(
        default=False, description="Whether chosen_tool matches expected_tool"
    )
    args_valid_pre_retry: bool = Field(
        default=False, description="Initial schema validation status"
    )
    args_valid_post_retry: bool = Field(default=False, description="Final schema validation status")
    args_strict_match: bool = Field(
        default=False, description="Exact/semantic match to acceptable args"
    )
    args_match_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Partial arg match ratio"
    )
    retry_attempted: bool = Field(
        default=False, description="Whether schema correction retry was sent"
    )
    retry_succeeded: bool = Field(
        default=False, description="Whether retry turned invalid args to valid"
    )
    chosen_args_has_extra_keys: bool = Field(
        default=False, description="True if chosen args contains unmodeled keys"
    )
    retrieval_hit: bool | None = Field(
        default=None, description="For C/D, whether expected tool was in top-k"
    )
    exposed_tools: list[str] = Field(
        default_factory=list, description="List of tools presented in context"
    )
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = 0
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = Field(default=None, description="Error string if LLM invocation failed")
    cached: bool = Field(default=False, description="Whether response came from local disk cache")


class RunMetricsRecord(BaseModel):
    """Aggregated quantitative performance metrics for a benchmark run."""

    model_config = ConfigDict(extra="ignore")

    tool_accuracy: float = Field(ge=0.0, le=1.0, description="Primary outcome metric")
    args_valid_pre_retry_rate: float = Field(ge=0.0, le=1.0)
    args_valid_post_retry_rate: float = Field(ge=0.0, le=1.0)
    args_strict_match_rate: float = Field(ge=0.0, le=1.0)
    recovery_rate: float = Field(ge=0.0, le=1.0)
    args_match_score_mean: float = Field(ge=0.0, le=1.0)
    retrieval_recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    total_prompts: int = Field(ge=0)
    total_tokens: int = Field(default=0, ge=0)
    avg_latency_ms: float = Field(default=0.0, ge=0.0)
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    extra_keys_count: int = Field(default=0, ge=0)


class BenchmarkResultFile(BaseModel):
    """Top-level schema for a complete benchmark run results JSON file."""

    model_config = ConfigDict(extra="ignore")

    setup: str = Field(description="Factorial setup letter (A, B, C, D) or 'pilot'")
    run_id: str = Field(description="Unique run execution UUID or identifier")
    model: str = Field(description="Evaluated LLM model name")
    timestamp: str = Field(description="ISO timestamp of run execution")
    evaluation_temperature: float = Field(
        default=0.0, description="Strictly 0.0 for evaluation runs"
    )
    heldout_hash: str | None = Field(
        default=None, description="SHA-256 integrity hash of held-out set"
    )
    metrics: RunMetricsRecord = Field(description="Aggregated run metrics")
    results: list[PromptResultRecord] = Field(description="Per-prompt detailed results")


class PairComparisonRecord(BaseModel):
    """Statistical outcome comparing two experimental setups on paired prompts."""

    model_config = ConfigDict(extra="ignore")

    comparison: str = Field(description="Label, e.g., 'A vs C'")
    setup_1: str = Field(description="Baseline setup identifier")
    setup_2: str = Field(description="Treatment setup identifier")
    n_prompts: int = Field(ge=1, description="Number of paired prompts evaluated")
    accuracy_1: float = Field(ge=0.0, le=1.0)
    accuracy_2: float = Field(ge=0.0, le=1.0)
    delta: float = Field(description="Accuracy difference (Acc2 - Acc1)")
    ci_95_lower: float = Field(description="95% bootstrap CI lower bound")
    ci_95_upper: float = Field(description="95% bootstrap CI upper bound")
    statistic: float = Field(ge=0.0, description="McNemar test statistic")
    raw_p_value: float = Field(ge=0.0, le=1.0, description="Unadjusted two-sided p-value")
    adjusted_p_value: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Holm-Bonferroni adjusted p-value"
    )
    method: str = Field(description="Statistical test method used (exact binomial or chi-square)")
    is_primary: bool = Field(description="True if one of the 3 primary hypotheses")
    note: str | None = Field(
        default=None, description="Contextual qualification (e.g. exploratory)"
    )


class FailureBreakdownRecord(BaseModel):
    """Categorized failure distribution for an experimental setup."""

    model_config = ConfigDict(extra="ignore")

    setup: str = Field(description="Setup identifier")
    total_prompts: int = Field(ge=0)
    category_counts: dict[str, int] = Field(description="Raw counts per failure category")
    category_percentages: dict[str, float] = Field(
        description="Percentage distribution per failure category"
    )
    prompt_breakdown: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-prompt failure classifications"
    )


class ComparisonReportFile(BaseModel):
    """Top-level schema for a complete factorial comparison JSON report."""

    model_config = ConfigDict(extra="ignore")

    timestamp: str = Field(description="Timestamp of comparison execution")
    heldout_hash: str | None = Field(default=None, description="Verified held-out dataset SHA-256")
    setups_evaluated: list[str] = Field(description="List of setups included in comparison")
    metrics_summary: dict[str, Any] = Field(description="Summary metrics dictionary across setups")
    primary_comparisons: list[PairComparisonRecord] = Field(
        description="Holm-corrected primary comparisons"
    )
    exploratory_comparisons: list[PairComparisonRecord] = Field(
        default_factory=list, description="Unadjusted exploratory comparisons"
    )
    failure_analyses: dict[str, FailureBreakdownRecord] = Field(
        description="Failure distribution per setup"
    )
    multiple_comparisons_note: str = Field(
        description="Methodological note regarding family-wise error rate control"
    )


def validate_benchmark_result_file(
    data_or_path: dict[str, Any] | Path | str,
) -> BenchmarkResultFile:
    """Validate a benchmark result dictionary or file against BenchmarkResultFile schema.

    Args:
        data_or_path: JSON-loaded dictionary or file path to inspect.

    Returns:
        Validated BenchmarkResultFile Pydantic model.

    Raises:
        ValueError / ValidationError if schema invariants are violated.
    """
    if isinstance(data_or_path, (str, Path)):
        p = Path(data_or_path)
        if not p.exists():
            raise FileNotFoundError(f"Result file not found: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        raw = data_or_path

    return BenchmarkResultFile.model_validate(raw)


def validate_comparison_report_file(
    data_or_path: dict[str, Any] | Path | str,
) -> ComparisonReportFile:
    """Validate a comparison report dictionary or file against ComparisonReportFile schema.

    Args:
        data_or_path: JSON-loaded dictionary or file path to inspect.

    Returns:
        Validated ComparisonReportFile Pydantic model.

    Raises:
        ValueError / ValidationError if schema invariants are violated.
    """
    if isinstance(data_or_path, (str, Path)):
        p = Path(data_or_path)
        if not p.exists():
            raise FileNotFoundError(f"Comparison report file not found: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        raw = data_or_path

    return ComparisonReportFile.model_validate(raw)


def validate_results_file(
    file_path: Path | str,
) -> tuple[Literal["benchmark", "comparison"], BaseModel]:
    """Auto-detect result file kind and validate against its respective schema.

    Args:
        file_path: Path to JSON file.

    Returns:
        tuple of (file_kind, validated_pydantic_model).

    Raises:
        ValueError: If file does not match either known schema.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Results file does not exist: {p}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid JSON format in {p}: expected JSON object.")

    # Determine kind based on key fingerprint
    if "primary_comparisons" in raw or "setups_evaluated" in raw:
        return "comparison", validate_comparison_report_file(raw)

    if "setup" in raw and "results" in raw:
        return "benchmark", validate_benchmark_result_file(raw)

    raise ValueError(
        f"Unable to recognize schema of '{p}'. "
        "Expected benchmark result or comparison report structure."
    )
