"""Benchmark evaluation runner, validation loop, and retry handling (F4+F7).

Executes benchmark prompts through LLM providers across setups A, B, C, and D,
performs JSON Schema argument validation, invokes a single-attempt retry hook on
validation failure with honest tool exposure, and persists comprehensive run records
to `results/{setup}_{split}_{timestamp}.json`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from toolbench.config import ToolbenchConfig
from toolbench.mcp_client import ToolInfo
from toolbench.output import compute_run_metrics, console, render_run_summary
from toolbench.providers.base import ProviderBase
from toolbench.validate import (
    compute_args_match_score,
    compute_args_strict_match,
    validate_schema,
)

logger = logging.getLogger(__name__)


class PromptResultRecord(BaseModel):
    """Detailed benchmark evaluation result record for a single prompt."""

    prompt_id: int | str
    prompt_text: str = ""
    prompt: str = ""
    expected_tool: str
    expected_args: dict[str, Any]
    acceptable_args: list[dict[str, Any]]
    chosen_tool: str | None = None
    chosen_args: dict[str, Any] = Field(default_factory=dict)
    tool_correct: bool = False
    args_valid_pre_retry: bool = False
    args_valid_post_retry: bool = False
    args_strict_match: bool = False
    args_match_score: float = 0.0
    chosen_args_has_extra_keys: bool = False
    retry_attempted: bool = False
    retry_fixed: bool = False
    retry_succeeded: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    cached: bool = False


class BenchmarkRunResult(BaseModel):
    """Summary and serialized per-prompt records for an evaluation run."""

    setup: str
    split: str
    timestamp: str
    model: str
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    heldout_hash: str | None = None
    evaluation_temperature: float = 0.0
    total_prompts: int = 0
    tool_accuracy: float = 0.0
    strict_match_rate: float = 0.0
    pre_retry_validity_rate: float = 0.0
    post_retry_validity_rate: float = 0.0
    records: list[PromptResultRecord] = Field(default_factory=list)
    results: list[PromptResultRecord] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    unverified: bool = False
    output_file: str | None = None


def verify_heldout_hash(
    heldout_path: Path | str = "data/heldout_prompts.json",
    hash_path: Path | str = "data/heldout_hash.sha256",
) -> None:
    """Verify integrity of frozen held-out prompts against saved SHA-256 digest."""
    h_file = Path(heldout_path)
    hash_file = Path(hash_path)

    if not h_file.exists():
        raise FileNotFoundError(f"Held-out dataset not found at '{h_file}'.")
    if not hash_file.exists():
        raise FileNotFoundError(f"Frozen held-out hash file not found at '{hash_file}'.")

    content_bytes = h_file.read_bytes()
    computed_hash = hashlib.sha256(content_bytes).hexdigest()
    expected_hash = hash_file.read_text(encoding="utf-8").strip()

    if computed_hash != expected_hash:
        raise ValueError(
            f"Held-out dataset hash mismatch! Expected {expected_hash}, got {computed_hash}. "
            "Dataset tampering or accidental modification detected."
        )


def load_prompts_for_split(
    split: str,
    data_dir: Path | str = "data",
    allow_unverified: bool = False,
) -> list[dict[str, Any]]:
    """Load prompt dataset records for the designated split.

    Args:
        split: 'pilot', 'dev', or 'heldout'.
        data_dir: Root directory storing prompt files.
        allow_unverified: Flag permitting execution on raw prompts.

    Returns:
        List of prompt dictionaries.
    """
    d_dir = Path(data_dir)
    if split == "pilot":
        file_path = d_dir / "raw_prompts.json"
    elif split == "dev":
        file_path = d_dir / "dev_prompts.json"
        if not file_path.exists() and allow_unverified:
            file_path = d_dir / "raw_prompts.json"
    elif split == "heldout":
        file_path = d_dir / "heldout_prompts.json"
        verify_heldout_hash(file_path, d_dir / "heldout_hash.sha256")
    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'pilot', 'dev', or 'heldout'.")

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file for split '{split}' not found at '{file_path}'.")

    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError(f"Dataset at '{file_path}' must be a JSON array.")
    return raw_data


def retry_hook(
    provider: ProviderBase,
    prompt_text: str,
    exposed_tools: list[dict[str, Any]],
    chosen_tool: str,
    validation_error: str,
) -> Any:
    """Execute a single schema validation retry attempt with honest tool exposure.

    Args:
        provider: ProviderBase instance.
        prompt_text: Original user query prompt.
        exposed_tools: Exact tool list exposed to the prompt (e.g. top-k in retrieval).
        chosen_tool: Tool selected in the first attempt.
        validation_error: Descriptive error message explaining schema violation.

    Returns:
        ProviderResponse from retry attempt.
    """
    retry_messages = [
        {"role": "user", "content": prompt_text},
        {
            "role": "assistant",
            "content": f"Selected tool '{chosen_tool}' with invalid arguments.",
        },
        {
            "role": "user",
            "content": (
                f"Your arguments for tool '{chosen_tool}' failed schema validation: "
                f"{validation_error}\nPlease provide corrected arguments for this tool call."
            ),
        },
    ]

    return provider.generate(
        messages=retry_messages,
        tools=exposed_tools,
        temperature=0.0,  # Evaluation temperature strictly 0.0
    )


def run_benchmark(
    tools: list[ToolInfo],
    provider: ProviderBase,
    config: ToolbenchConfig | None = None,
    split: str = "pilot",
    setup: str = "A",
    data_dir: Path | str = "data",
    results_dir: Path | str = "results",
    allow_unverified: bool = False,
    acknowledge_ceiling: bool = False,
    tool_selector: Any | None = None,
    no_cache: bool = False,
    clear_cache: bool = False,
) -> BenchmarkRunResult:
    """Execute evaluation across prompts in the designated split and setup.

    Args:
        tools: Discovered tools.
        provider: Initialized ProviderBase instance.
        config: ToolbenchConfig instance.
        split: 'pilot', 'dev', or 'heldout'.
        setup: 'A', 'B', 'C', or 'D'.
        data_dir: Directory containing prompt JSON files.
        results_dir: Directory to save run results.
        allow_unverified: Allow dev split to run on raw prompts.
        acknowledge_ceiling: Bypass ceiling effect gate if tool accuracy >= 95%.
        tool_selector: Optional retrieval callback (query, tools) -> list[ToolInfo] for C/D.

    Returns:
        BenchmarkRunResult instance with summary and per-prompt records.
    """
    cfg = config or ToolbenchConfig()

    if clear_cache:
        from toolbench.cache import DiskResponseCache

        cache_p = Path(getattr(cfg.paths, "cache_dir", ".cache/toolbench")) / "responses"
        DiskResponseCache(cache_dir=cache_p).clear()

    orig_cache = getattr(provider, "cache", None)
    if no_cache and hasattr(provider, "cache"):
        provider.cache = None

    prompts = load_prompts_for_split(
        split=split,
        data_dir=data_dir,
        allow_unverified=allow_unverified,
    )

    # Sort tools alphabetically by name for primary benchmark runs
    # (invariance against positional bias)
    sorted_tools = sorted(tools, key=lambda t: t.name)
    tool_map = {t.name: t for t in sorted_tools}

    records: list[PromptResultRecord] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(
        "Starting benchmark run: setup=%s, split=%s, prompts=%d, model=%s",
        setup,
        split,
        len(prompts),
        provider.model_name,
    )

    for item in prompts:
        prompt_id = int(item.get("prompt_id", 0))
        prompt_text = str(item.get("prompt_text", ""))
        expected_tool = str(item.get("expected_tool", ""))
        expected_args = dict(item.get("expected_args") or {})
        acceptable_args: list[dict[str, Any]] = item.get("acceptable_args") or []
        if not acceptable_args and expected_args:
            acceptable_args = [expected_args]

        # Determine exposed tools based on setup
        if setup in ("C", "D") and tool_selector is not None:
            exposed_tool_objs = tool_selector(prompt_text, sorted_tools)
        else:
            exposed_tool_objs = sorted_tools

        exposed_tool_schemas = [t.to_dict() for t in exposed_tool_objs]
        exposed_tool_names = {t.name for t in exposed_tool_objs}

        # 1. Primary evaluation call
        resp = provider.generate(
            messages=prompt_text,
            tools=exposed_tool_schemas,
            temperature=cfg.evaluation.temperature,  # Strictly 0.0
        )

        chosen_tool = resp.chosen_tool
        chosen_args = resp.chosen_args or {}
        error = resp.error
        prompt_tokens = resp.prompt_tokens
        completion_tokens = resp.completion_tokens
        latency_ms = resp.latency_ms
        cached = resp.cached

        # Evaluate tool selection
        tool_in_exposed = chosen_tool in exposed_tool_names if chosen_tool else False
        tool_correct = bool(chosen_tool and chosen_tool == expected_tool and tool_in_exposed)

        # 2. Validate arguments against schema
        target_schema = {}
        if chosen_tool and chosen_tool in tool_map:
            target_schema = tool_map[chosen_tool].input_schema

        valid_pre, val_err = validate_schema(target_schema, chosen_args)

        # 3. Retry hook (at most 1 retry attempt)
        retry_attempted = False
        retry_fixed = False
        valid_post = valid_pre

        if (
            not valid_pre
            and chosen_tool
            and cfg.evaluation.max_retries >= 1
            and val_err is not None
        ):
            retry_attempted = True
            retry_resp = retry_hook(
                provider=provider,
                prompt_text=prompt_text,
                exposed_tools=exposed_tool_schemas,
                chosen_tool=chosen_tool,
                validation_error=val_err,
            )
            # Accumulate metrics
            prompt_tokens += retry_resp.prompt_tokens
            completion_tokens += retry_resp.completion_tokens
            latency_ms += retry_resp.latency_ms

            if retry_resp.chosen_tool:
                chosen_tool = retry_resp.chosen_tool
                chosen_args = retry_resp.chosen_args or {}
                # Update tool correctness if retry changed the tool selection
                tool_in_exposed = chosen_tool in exposed_tool_names
                tool_correct = bool(chosen_tool == expected_tool and tool_in_exposed)

                retry_schema = tool_map[chosen_tool].input_schema if chosen_tool in tool_map else {}
                valid_post, _ = validate_schema(retry_schema, chosen_args)
                retry_fixed = valid_post

        # 4. Compute argument metrics
        strict_match, has_extra_keys = compute_args_strict_match(
            acceptable_args=acceptable_args,
            chosen_args=chosen_args,
        )
        match_score = compute_args_match_score(
            acceptable_args=acceptable_args,
            chosen_args=chosen_args,
            tool_correct=tool_correct,
        )

        record = PromptResultRecord(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            prompt=prompt_text,
            expected_tool=expected_tool,
            expected_args=expected_args,
            acceptable_args=acceptable_args,
            chosen_tool=chosen_tool,
            chosen_args=chosen_args,
            tool_correct=tool_correct,
            args_valid_pre_retry=valid_pre,
            args_valid_post_retry=valid_post,
            args_strict_match=strict_match,
            args_match_score=match_score,
            chosen_args_has_extra_keys=has_extra_keys,
            retry_attempted=retry_attempted,
            retry_fixed=retry_fixed,
            retry_succeeded=retry_fixed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            error=error,
            cached=cached,
        )
        records.append(record)

    total_count = len(records)
    tool_accuracy = sum(1 for r in records if r.tool_correct) / total_count if total_count else 0.0
    strict_match_rate = (
        sum(1 for r in records if r.args_strict_match) / total_count if total_count else 0.0
    )
    pre_validity_rate = (
        sum(1 for r in records if r.args_valid_pre_retry) / total_count if total_count else 0.0
    )
    post_validity_rate = (
        sum(1 for r in records if r.args_valid_post_retry) / total_count if total_count else 0.0
    )

    heldout_hash_val = None
    if split == "heldout":
        h_file = Path(data_dir) / "heldout_hash.sha256"
        if h_file.exists():
            heldout_hash_val = h_file.read_text(encoding="utf-8").strip()

    recovery_count = sum(1 for r in records if r.retry_fixed)
    invalid_pre_count = sum(1 for r in records if not r.args_valid_pre_retry)
    recovery_rate = (recovery_count / invalid_pre_count) if invalid_pre_count > 0 else 0.0
    match_score_mean = (
        sum(r.args_match_score for r in records) / total_count if total_count else 0.0
    )
    cache_hits = sum(1 for r in records if r.cached)
    cache_hit_rate = (cache_hits / total_count) if total_count else 0.0
    avg_latency = sum(r.latency_ms for r in records) / total_count if total_count else 0.0
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in records)

    metrics_payload = {
        "tool_accuracy": tool_accuracy,
        "args_valid_pre_retry_rate": pre_validity_rate,
        "args_valid_post_retry_rate": post_validity_rate,
        "args_strict_match_rate": strict_match_rate,
        "recovery_rate": recovery_rate,
        "args_match_score_mean": match_score_mean,
        "total_prompts": total_count,
        "total_tokens": total_tokens,
        "avg_latency_ms": avg_latency,
        "cache_hit_rate": cache_hit_rate,
    }

    records_dict = [r.model_dump() for r in records]
    run_metrics = compute_run_metrics(records_dict)

    res_dir = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    out_file = res_dir / f"{setup}_{split}_{timestamp}.json"
    run_id = f"{timestamp}_{setup}_{uuid.uuid4().hex[:8]}"
    run_folder = res_dir / run_id
    run_folder.mkdir(parents=True, exist_ok=True)

    run_result = BenchmarkRunResult(
        setup=setup,
        split=split,
        timestamp=timestamp,
        model=provider.model_name,
        run_id=run_id,
        heldout_hash=heldout_hash_val,
        evaluation_temperature=cfg.evaluation.temperature,
        total_prompts=total_count,
        tool_accuracy=tool_accuracy,
        strict_match_rate=strict_match_rate,
        pre_retry_validity_rate=pre_validity_rate,
        post_retry_validity_rate=post_validity_rate,
        records=records,
        results=records,
        metrics=metrics_payload,
        unverified=allow_unverified,
        output_file=str(out_file),
    )

    json_content = run_result.model_dump_json(indent=2)
    out_file.write_text(json_content, encoding="utf-8")
    (run_folder / "results.json").write_text(json_content, encoding="utf-8")
    (run_folder / "summary.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    # Save sanitized config snapshot
    try:
        import yaml

        safe_cfg = cfg.model_dump()
        if "model" in safe_cfg and "api_key" in safe_cfg["model"]:
            safe_cfg["model"]["api_key"] = None
        (run_folder / "config_snapshot.yaml").write_text(
            yaml.dump(safe_cfg, sort_keys=False), encoding="utf-8"
        )
    except Exception as err:
        logger.warning("Could not write config snapshot: %s", err)

    if no_cache and orig_cache is not None and hasattr(provider, "cache"):
        provider.cache = orig_cache

    logger.info("Saved benchmark evaluation results to '%s' and '%s'.", out_file, run_folder)
    summary_table = render_run_summary(
        run_metrics,
        title=f"Benchmark Execution Summary ({setup} - {split})",
    )
    console.print(summary_table)

    # Dev-set ceiling effect warning check
    if split == "dev" and setup == "A" and tool_accuracy >= 0.95 and not acknowledge_ceiling:
        logger.warning(
            "CEILING EFFECT DETECTED: Baseline tool accuracy on dev split is %.1f%% "
            "(>= 95%%). Downstream retrieval may show minimal observable delta. "
            "Use --acknowledge-ceiling to proceed.",
            tool_accuracy * 100.0,
        )

    return run_result
