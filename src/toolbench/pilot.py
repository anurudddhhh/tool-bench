"""Pre-review feasibility pilot gate (Phase 4.5 / Pilot).

Evaluates baseline tool accuracy on ~40 raw unverified prompts before manual review.
If accuracy >= 95%, the gate fails to prevent wasting hours reviewing a tool set
that is too simple for meaningful retrieval benchmarking (ceiling effect).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from toolbench.config import GenerationConfig, ToolbenchConfig
from toolbench.generate import PromptRecord, generate_prompts
from toolbench.mcp_client import ToolInfo
from toolbench.output import print_error, print_info, print_success, print_warning
from toolbench.providers.base import ProviderBase

logger = logging.getLogger(__name__)


class PilotResultRecord(BaseModel):
    """Evaluation result for an individual pilot prompt."""

    prompt_id: int
    prompt_text: str
    expected_tool: str
    chosen_tool: str | None = None
    tool_correct: bool = False
    error: str | None = None


class PilotSummary(BaseModel):
    """Complete summary of the pilot feasibility gate run."""

    passed: bool
    total_prompts: int
    correct_count: int
    tool_accuracy: float
    tool_accuracy_pct: float
    timestamp: str
    result_path: str
    records: list[PilotResultRecord] = Field(default_factory=list)


def run_pilot_gate(
    tools: list[ToolInfo],
    provider: ProviderBase,
    config: ToolbenchConfig | None = None,
    count: int = 40,
    prompts: list[PromptRecord] | list[dict[str, Any]] | None = None,
    results_dir: Path | str = "results",
    raw_prompts_path: Path | str = "data/raw_prompts.json",
) -> tuple[bool, PilotSummary]:
    """Execute the pre-review pilot feasibility gate.

    Args:
        tools: Full list of discovered tools.
        provider: ProviderBase instance (GeminiProvider or MockProvider).
        config: Optional ToolbenchConfig instance.
        count: Target raw prompt count to evaluate (default 40, minimum 20).
        prompts: Optional pre-loaded prompt records to evaluate.
        results_dir: Directory where pilot results are logged (default 'results').
        raw_prompts_path: Path to raw prompts JSON if generated.

    Returns:
        Tuple of (passed: bool, summary: PilotSummary).
        passed is True if tool_accuracy < 0.95, False if tool_accuracy >= 0.95.
    """
    cfg = config or ToolbenchConfig()

    # Warn if available tool set is small
    if len(tools) < 40:
        print_warning(
            f"Available tools ({len(tools)}) is below the recommended 40-tool threshold. "
            "A ceiling effect is likely regardless of pilot outcome."
        )

    # 1. Acquire raw prompts
    pilot_prompts: list[dict[str, Any]] = []

    if prompts is not None:
        for p in prompts:
            if isinstance(p, PromptRecord):
                pilot_prompts.append(p.model_dump())
            elif isinstance(p, dict):
                pilot_prompts.append(p)
    else:
        raw_path = Path(raw_prompts_path)
        if raw_path.exists():
            try:
                loaded = json.loads(raw_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list) and len(loaded) >= 20:
                    pilot_prompts = loaded[:count]
            except Exception as err:
                logger.warning("Could not load existing raw prompts at %s: %s", raw_path, err)

        if not pilot_prompts:
            print_info(f"Generating ~{count} synthetic prompts for pilot feasibility gate...")
            gen_cfg = GenerationConfig(count=max(40, count))
            generated_records = generate_prompts(
                tools=tools,
                provider=provider,
                config=gen_cfg,
                output_path=raw_prompts_path,
            )
            pilot_prompts = [r.model_dump() for r in generated_records[:count]]

    if len(pilot_prompts) < 20:
        raise RuntimeError(
            f"Insufficient prompts for pilot evaluation: need at least 20, "
            f"got {len(pilot_prompts)}."
        )

    # Sort tools alphabetically by name for primary benchmark runs
    # (invariance against positional bias)
    sorted_tools = sorted(tools, key=lambda t: t.name)
    exposed_schemas = [t.to_dict() for t in sorted_tools]
    exposed_names = {t.name for t in sorted_tools}

    print_info(
        f"Evaluating all-tools baseline across {len(pilot_prompts)} raw prompts "
        f"with {len(sorted_tools)} tools..."
    )

    records: list[PilotResultRecord] = []
    correct_count = 0

    for item in pilot_prompts:
        prompt_id = int(item.get("prompt_id", 0))
        prompt_text = str(item.get("prompt_text", ""))
        expected_tool = str(item.get("expected_tool", ""))

        try:
            resp = provider.generate(
                messages=prompt_text,
                tools=exposed_schemas,
                temperature=cfg.evaluation.temperature,  # Evaluation temperature strictly 0.0
            )
            chosen_tool = resp.chosen_tool
            err = resp.error
        except Exception as exc:
            logger.warning("Provider error on pilot prompt %d: %s", prompt_id, exc)
            chosen_tool = None
            err = str(exc)

        tool_in_exposed = chosen_tool in exposed_names if chosen_tool else False
        tool_correct = bool(chosen_tool and chosen_tool == expected_tool and tool_in_exposed)

        if tool_correct:
            correct_count += 1

        records.append(
            PilotResultRecord(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                expected_tool=expected_tool,
                chosen_tool=chosen_tool,
                tool_correct=tool_correct,
                error=err,
            )
        )

    total_prompts = len(records)
    tool_accuracy = (correct_count / total_prompts) if total_prompts > 0 else 0.0
    tool_accuracy_pct = tool_accuracy * 100.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Decision Logic
    # If accuracy >= 95%, gate fails
    if tool_accuracy >= 0.95:
        passed = False
        print_error(
            f"PILOT FAILED: tool accuracy is {tool_accuracy_pct:.1f}% (>= 95%). "
            "The tool set is too easy for meaningful retrieval measurement."
        )
        print_warning(
            "Suggestions to introduce challenge:\n"
            "  1. Add more MCP servers (e.g. GitHub, Docker, Slack, Postgres).\n"
            "  2. Include overlapping or fine-grained tool suites with similar descriptions."
        )
    else:
        passed = True
        print_success(
            f"Pilot passed: tool accuracy is {tool_accuracy_pct:.1f}% (< 95%). "
            "Proceed to hand review (F3)."
        )

    # 3. Persist pilot results for audit
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"pilot_{timestamp}.json"

    summary = PilotSummary(
        passed=passed,
        total_prompts=total_prompts,
        correct_count=correct_count,
        tool_accuracy=tool_accuracy,
        tool_accuracy_pct=tool_accuracy_pct,
        timestamp=timestamp,
        result_path=str(out_file),
        records=records,
    )

    out_file.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved pilot gate run results to '%s'.", out_file)
    return passed, summary
