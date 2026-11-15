"""Unit tests for pre-review feasibility pilot gate (Phase 4.5 / Pilot)."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from toolbench.mcp_client import ToolInfo
from toolbench.pilot import run_pilot_gate
from toolbench.providers.base import ProviderBase, ProviderResponse


class ScriptedPilotProvider(ProviderBase):
    """Provider returning scripted tool choices for pilot testing."""

    def __init__(self, tool_choices: list[str | None]) -> None:
        super().__init__(model_name="mock-pilot-model")
        self.tool_choices = list(tool_choices)
        self.idx = 0

    def generate(
        self,
        messages: list[dict[str, Any]] | str,
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> ProviderResponse:
        chosen = self.tool_choices[self.idx % len(self.tool_choices)]
        self.idx += 1
        return ProviderResponse(chosen_tool=chosen, chosen_args={})


def _make_tools(count: int = 50) -> list[ToolInfo]:
    return [
        ToolInfo(
            name=f"server__tool_{i}",
            description=f"Description for tool {i}",
            input_schema={"type": "object", "properties": {}},
        )
        for i in range(count)
    ]


def test_pilot_gate_pass(tmp_path: Path) -> None:
    """Verify pilot gate passes when tool accuracy is below 95%."""
    tools = _make_tools(50)
    results_dir = tmp_path / "results"

    # Create 20 prompts targeting tool_0
    prompts = [
        {
            "prompt_id": i,
            "prompt_text": f"Query {i}",
            "expected_tool": "server__tool_0",
        }
        for i in range(20)
    ]

    # Scripted choices: 15 correct (tool_0), 5 incorrect (tool_1) -> 75% accuracy
    choices = ["server__tool_0"] * 15 + ["server__tool_1"] * 5
    provider = ScriptedPilotProvider(choices)

    passed, summary = run_pilot_gate(
        tools=tools,
        provider=provider,
        prompts=prompts,
        results_dir=results_dir,
    )

    assert passed is True
    assert summary.total_prompts == 20
    assert summary.correct_count == 15
    assert summary.tool_accuracy == 0.75
    assert summary.tool_accuracy_pct == 75.0

    # Verify JSON file written to results_dir
    json_files = list(results_dir.glob("pilot_*.json"))
    assert len(json_files) == 1
    disk_data = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert disk_data["passed"] is True
    assert disk_data["tool_accuracy"] == 0.75


def test_pilot_gate_fail_ceiling(tmp_path: Path) -> None:
    """Verify pilot gate fails when tool accuracy is >= 95% (ceiling effect)."""
    tools = _make_tools(50)
    results_dir = tmp_path / "results"

    # 20 prompts all targeting tool_0
    prompts = [
        {
            "prompt_id": i,
            "prompt_text": f"Query {i}",
            "expected_tool": "server__tool_0",
        }
        for i in range(20)
    ]

    # 100% correct choices
    choices = ["server__tool_0"] * 20
    provider = ScriptedPilotProvider(choices)

    passed, summary = run_pilot_gate(
        tools=tools,
        provider=provider,
        prompts=prompts,
        results_dir=results_dir,
    )

    assert passed is False
    assert summary.total_prompts == 20
    assert summary.correct_count == 20
    assert summary.tool_accuracy == 1.0


def test_pilot_gate_insufficient_prompts_raises() -> None:
    """Verify RuntimeError is raised when fewer than 20 raw prompts are provided."""
    tools = _make_tools(10)
    provider = MagicMock(spec=ProviderBase)

    # Only 10 prompts
    few_prompts = [
        {"prompt_id": i, "prompt_text": f"Q{i}", "expected_tool": "t0"}
        for i in range(10)
    ]

    with pytest.raises(RuntimeError) as exc_info:
        run_pilot_gate(
            tools=tools,
            provider=provider,
            prompts=few_prompts,
        )
    assert "Insufficient prompts for pilot evaluation: need at least 20" in str(exc_info.value)


def test_pilot_gate_low_tool_count_warning() -> None:
    """Verify warning is emitted when total tool count is under 40."""
    tools = _make_tools(10)  # Only 10 tools (< 40)
    provider = ScriptedPilotProvider(["server__tool_0"] * 20)
    prompts = [
        {"prompt_id": i, "prompt_text": f"Q{i}", "expected_tool": "server__tool_0"}
        for i in range(20)
    ]

    with patch("toolbench.pilot.print_warning") as mock_warn:
        run_pilot_gate(
            tools=tools,
            provider=provider,
            prompts=prompts,
        )
        assert mock_warn.call_count >= 1
        args, _ = mock_warn.call_args_list[0]
        assert "below the recommended 40-tool threshold" in args[0]
