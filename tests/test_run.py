"""Unit tests for benchmark runner, validation loop, and retry handling (F4+F7)."""

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from toolbench.config import ToolbenchConfig
from toolbench.mcp_client import ToolInfo
from toolbench.providers.base import ProviderBase, ProviderResponse
from toolbench.run import (
    load_prompts_for_split,
    retry_hook,
    run_benchmark,
    verify_heldout_hash,
)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_verify_heldout_hash(tmp_path: Path) -> None:
    """Verify held-out dataset SHA-256 integrity verification."""
    data_dir = tmp_path / "data"
    heldout_file = data_dir / "heldout_prompts.json"
    hash_file = data_dir / "heldout_hash.sha256"

    content = '[{"prompt_id": 1, "prompt_text": "sample"}]'
    heldout_file.parent.mkdir(parents=True, exist_ok=True)
    heldout_file.write_text(content, encoding="utf-8")

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    hash_file.write_text(digest, encoding="utf-8")

    # 1. Valid hash matches
    verify_heldout_hash(heldout_file, hash_file)

    # 2. Tampered content fails
    heldout_file.write_text('[{"prompt_id": 1, "prompt_text": "tampered"}]', encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        verify_heldout_hash(heldout_file, hash_file)
    assert "Held-out dataset hash mismatch" in str(exc_info.value)


def test_load_prompts_for_split(tmp_path: Path) -> None:
    """Verify prompt loading across different splits and fallback handling."""
    data_dir = tmp_path / "data"
    raw_prompts = [{"prompt_id": 10, "prompt_text": "raw prompt"}]
    dev_prompts = [{"prompt_id": 20, "prompt_text": "dev prompt"}]

    _write_json(data_dir / "raw_prompts.json", raw_prompts)
    _write_json(data_dir / "dev_prompts.json", dev_prompts)

    # Pilot split loads raw_prompts.json
    p_pilot = load_prompts_for_split("pilot", data_dir=data_dir)
    assert len(p_pilot) == 1
    assert p_pilot[0]["prompt_id"] == 10

    # Dev split loads dev_prompts.json
    p_dev = load_prompts_for_split("dev", data_dir=data_dir)
    assert len(p_dev) == 1
    assert p_dev[0]["prompt_id"] == 20

    # Unknown split raises ValueError
    with pytest.raises(ValueError) as exc_info:
        load_prompts_for_split("invalid_split", data_dir=data_dir)
    assert "Unknown split" in str(exc_info.value)


def test_retry_hook_invocation() -> None:
    """Verify retry hook constructs messages and calls provider at temperature 0.0."""
    mock_provider = MagicMock(spec=ProviderBase)
    mock_provider.generate.return_value = ProviderResponse(
        chosen_tool="fs__read_file",
        chosen_args={"path": "corrected.txt"},
    )

    resp = retry_hook(
        provider=mock_provider,
        prompt_text="Read my file",
        exposed_tools=[{"name": "fs__read_file"}],
        chosen_tool="fs__read_file",
        validation_error="Missing required property 'path'",
    )

    assert resp.chosen_tool == "fs__read_file"
    assert resp.chosen_args == {"path": "corrected.txt"}
    assert mock_provider.generate.call_count == 1

    _, kwargs = mock_provider.generate.call_args
    assert kwargs["temperature"] == 0.0
    messages = kwargs["messages"]
    assert len(messages) == 3
    assert "Missing required property 'path'" in messages[2]["content"]


class StateCheckingMockProvider(ProviderBase):
    """Mock provider allowing custom response sequencing per call."""

    def __init__(self, responses: list[ProviderResponse]) -> None:
        super().__init__(model_name="mock-model")
        self.responses = list(responses)
        self.call_count = 0

    def generate(
        self,
        messages: list[dict[str, Any]] | str,
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> ProviderResponse:
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp


def test_run_benchmark_execution_and_retry(tmp_path: Path) -> None:
    """Verify run_benchmark loop, retry fixing invalid args, and record generation."""
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"

    tool_fs = ToolInfo(
        name="fs__read_file",
        description="Read file",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    tools = [tool_fs]

    prompts = [
        # Prompt 1: Correct tool, valid args first try
        {
            "prompt_id": 1,
            "prompt_text": "Read report.txt",
            "expected_tool": "fs__read_file",
            "expected_args": {"path": "report.txt"},
            "acceptable_args": [{"path": "report.txt"}],
        },
        # Prompt 2: Correct tool, invalid args (missing path), fixed on retry
        {
            "prompt_id": 2,
            "prompt_text": "Read data.csv",
            "expected_tool": "fs__read_file",
            "expected_args": {"path": "data.csv"},
            "acceptable_args": [{"path": "data.csv"}],
        },
        # Prompt 3: Wrong tool
        {
            "prompt_id": 3,
            "prompt_text": "Commit changes",
            "expected_tool": "git__commit",
            "expected_args": {"msg": "initial"},
            "acceptable_args": [{"msg": "initial"}],
        },
    ]
    _write_json(data_dir / "raw_prompts.json", prompts)

    # Sequence of 4 provider responses:
    # 1. Prompt 1 initial (valid)
    # 2. Prompt 2 initial (invalid: missing path)
    # 3. Prompt 2 retry (valid: has path)
    # 4. Prompt 3 initial (wrong tool)
    responses = [
        ProviderResponse(chosen_tool="fs__read_file", chosen_args={"path": "report.txt"}),
        ProviderResponse(chosen_tool="fs__read_file", chosen_args={"wrong_field": 1}),
        ProviderResponse(chosen_tool="fs__read_file", chosen_args={"path": "data.csv"}),
        ProviderResponse(chosen_tool="fs__read_file", chosen_args={"path": "commit.txt"}),
    ]
    provider = StateCheckingMockProvider(responses)

    cfg = ToolbenchConfig()
    cfg.evaluation.max_retries = 1

    run_res = run_benchmark(
        tools=tools,
        provider=provider,
        config=cfg,
        split="pilot",
        setup="A",
        data_dir=data_dir,
        results_dir=results_dir,
    )

    assert run_res.total_prompts == 3
    # Prompts 1 and 2 are correct tool; Prompt 3 is incorrect
    assert pytest.approx(run_res.tool_accuracy, 1e-4) == 2.0 / 3.0

    # Prompt 1 checks
    rec1 = run_res.records[0]
    assert rec1.tool_correct is True
    assert rec1.args_valid_pre_retry is True
    assert rec1.retry_attempted is False
    assert rec1.args_strict_match is True

    # Prompt 2 checks (Fixed by retry)
    rec2 = run_res.records[1]
    assert rec2.tool_correct is True
    assert rec2.args_valid_pre_retry is False
    assert rec2.retry_attempted is True
    assert rec2.retry_fixed is True
    assert rec2.args_valid_post_retry is True
    assert rec2.args_strict_match is True

    # Prompt 3 checks (Incorrect tool)
    rec3 = run_res.records[2]
    assert rec3.tool_correct is False
    assert rec3.args_match_score == 0.0

    # Verify JSON file on disk
    json_files = list(results_dir.glob("A_pilot_*.json"))
    assert len(json_files) == 1
    disk_data = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert disk_data["setup"] == "A"
    assert disk_data["split"] == "pilot"
    assert disk_data["total_prompts"] == 3


def test_run_benchmark_setup_c_tool_selector(tmp_path: Path) -> None:
    """Verify Setup C restricts exposed tools via the tool_selector callback."""
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"

    t1 = ToolInfo(name="tool_1", description="T1")
    t2 = ToolInfo(name="tool_2", description="T2")
    t3 = ToolInfo(name="tool_3", description="T3")
    all_tools = [t1, t2, t3]

    prompts = [
        {
            "prompt_id": 1,
            "prompt_text": "Do task 1",
            "expected_tool": "tool_1",
            "expected_args": {},
        }
    ]
    _write_json(data_dir / "raw_prompts.json", prompts)

    captured_exposed_tools: list[list[dict[str, Any]]] = []

    class CapturingProvider(ProviderBase):
        def __init__(self) -> None:
            super().__init__(model_name="capturing-model")

        def generate(
            self,
            messages: list[dict[str, Any]] | str,
            tools: list[dict[str, Any]],
            temperature: float = 0.0,
        ) -> ProviderResponse:
            captured_exposed_tools.append(tools)
            return ProviderResponse(chosen_tool="tool_1", chosen_args={})

    # Selector returns top-1 tool (only t1)
    def mock_selector(query: str, tool_list: list[ToolInfo]) -> list[ToolInfo]:
        return [t1]

    run_benchmark(
        tools=all_tools,
        provider=CapturingProvider(),
        split="pilot",
        setup="C",
        data_dir=data_dir,
        results_dir=results_dir,
        tool_selector=mock_selector,
    )

    assert len(captured_exposed_tools) == 1
    exposed_names = [t["name"] for t in captured_exposed_tools[0]]
    # Only tool_1 was exposed!
    assert exposed_names == ["tool_1"]


def test_run_benchmark_run_folder_and_snapshot(tmp_path: Path) -> None:
    """Verify run folder is created with results.json, summary.json, and config_snapshot.yaml."""
    t1 = ToolInfo(
        name="tool_1",
        description="Tool 1",
        input_schema={"type": "object", "properties": {}},
    )
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    prompts = [
        {
            "prompt_id": 1,
            "prompt_text": "Do tool 1",
            "expected_tool": "tool_1",
            "expected_args": {},
        }
    ]
    _write_json(data_dir / "raw_prompts.json", prompts)

    mock_provider = MagicMock(spec=ProviderBase)
    mock_provider.model_name = "test-model"
    mock_provider.generate.return_value = ProviderResponse(chosen_tool="tool_1", chosen_args={})

    cfg = ToolbenchConfig()
    result = run_benchmark(
        tools=[t1],
        provider=mock_provider,
        config=cfg,
        split="pilot",
        setup="A",
        data_dir=data_dir,
        results_dir=results_dir,
        no_cache=True,
    )

    run_folder = results_dir / result.run_id
    assert run_folder.exists()
    assert (run_folder / "results.json").exists()
    assert (run_folder / "summary.json").exists()
    assert (run_folder / "config_snapshot.yaml").exists()
