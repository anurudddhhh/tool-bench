"""Unit tests for Toolbench configuration and invariant validation (F11c)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from toolbench.config import (
    EvaluationConfig,
    MCPServerConfig,
    RateLimitConfig,
    SplitConfig,
    ToolbenchConfig,
    load_config,
)


def test_default_config() -> None:
    """Verify that ToolbenchConfig initializes with valid default parameters."""
    cfg = ToolbenchConfig()
    assert cfg.model.provider == "gemini"
    assert cfg.model.name == "gemini-3.5-flash-lite"
    assert cfg.generation.count == 160
    assert cfg.generation.temperature == 0.7
    assert cfg.evaluation.temperature == 0.0
    assert cfg.evaluation.max_retries == 1
    assert cfg.split.heldout_size == 50
    assert cfg.split.min_dev_size == 40
    assert cfg.rate_limit.rpm == 15
    assert cfg.rate_limit.rpd == 500
    assert cfg.rate_limit.tpm == 250000
    assert cfg.rate_limit.max_retries == 5


def test_load_config_default_file() -> None:
    """Verify that load_config() finds and parses the workspace config.yaml."""
    cfg = load_config()
    assert len(cfg.mcp) >= 5
    server_names = [s.name for s in cfg.mcp]
    assert "filesystem" in server_names
    assert "git" in server_names
    assert cfg.evaluation.temperature == 0.0


def test_load_config_explicit_path(tmp_path: Path) -> None:
    """Verify that load_config() parses an explicit custom YAML path."""
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(
        """
model:
  name: "gemini-1.5-pro"
generation:
  count: 80
split:
  heldout_size: 60
""",
        encoding="utf-8",
    )
    cfg = load_config(custom_yaml)
    assert cfg.model.name == "gemini-1.5-pro"
    assert cfg.generation.count == 80
    assert cfg.split.heldout_size == 60


def test_eval_temperature_invariant() -> None:
    """Verify that evaluation temperature must strictly equal 0.0."""
    # Valid
    cfg = EvaluationConfig(temperature=0.0)
    assert cfg.temperature == 0.0

    # Non-zero temperature must be rejected
    with pytest.raises(ValidationError) as exc_info:
        EvaluationConfig(temperature=0.7)
    assert "Evaluation temperature must be 0.0" in str(exc_info.value)


def test_max_retries_invariant() -> None:
    """Verify that retry attempts are capped at 1."""
    # 0 and 1 are valid
    assert EvaluationConfig(max_retries=0).max_retries == 0
    assert EvaluationConfig(max_retries=1).max_retries == 1

    # 2 retries violates the one-retry cap rule
    with pytest.raises(ValidationError):
        EvaluationConfig(max_retries=2)

    with pytest.raises(ValidationError):
        EvaluationConfig(max_retries=-1)


def test_heldout_size_minimum() -> None:
    """Verify that heldout_size enforces the minimum of 50 prompts."""
    assert SplitConfig(heldout_size=50).heldout_size == 50
    assert SplitConfig(heldout_size=100).heldout_size == 100

    # Under 50 is invalid
    with pytest.raises(ValidationError):
        SplitConfig(heldout_size=49)


def test_mcp_server_config() -> None:
    """Verify MCP server config structure and defaults."""
    srv = MCPServerConfig(name="test_srv", command="uvx", args=["mcp-server-fetch"])
    assert srv.name == "test_srv"
    assert srv.command == "uvx"
    assert srv.args == ["mcp-server-fetch"]
    assert srv.env == {}


def test_nonexistent_config_file_raises() -> None:
    """Verify FileNotFoundError when explicit config file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist_at_all.yaml")


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    """Verify ValueError when YAML syntax is malformed."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("model: [unclosed list", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_config(bad_yaml)
    assert "Failed to parse YAML" in str(exc_info.value)


def test_non_dict_yaml_raises(tmp_path: Path) -> None:
    """Verify ValueError when YAML content is not a mapping."""
    bad_yaml = tmp_path / "list.yaml"
    bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_config(bad_yaml)
    assert "must define a YAML mapping/dict" in str(exc_info.value)


def test_rate_limit_max_retries_range() -> None:
    """Verify that rate limit backoff retries enforce 1..10 range."""
    assert RateLimitConfig(max_retries=1).max_retries == 1
    assert RateLimitConfig(max_retries=5).max_retries == 5
    assert RateLimitConfig(max_retries=10).max_retries == 10

    # Below 1 is invalid
    with pytest.raises(ValidationError):
        RateLimitConfig(max_retries=0)

    # Above 10 is invalid
    with pytest.raises(ValidationError):
        RateLimitConfig(max_retries=11)

