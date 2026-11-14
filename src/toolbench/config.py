"""Configuration management and schema validation for Toolbench.

Loads YAML configuration files into strongly-typed Pydantic models, enforcing
architectural invariants (e.g. evaluation temperature == 0.0, heldout_size >= 50).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class MCPServerConfig(BaseModel):
    """Configuration for an individual MCP server subprocess."""

    name: str = Field(
        ...,
        description="Unique human-readable server identifier (e.g. 'filesystem')",
    )
    command: str = Field(
        ...,
        description="Command binary to execute ('cmd', 'uvx', 'docker')",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Positional arguments for launch",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for subprocess",
    )


class ModelConfig(BaseModel):
    """LLM provider and model identifier configuration."""

    provider: str = Field(
        default="gemini",
        description="Provider identifier ('gemini' or 'mock')",
    )
    name: str = Field(
        default="gemini-2.5-flash",
        description="Exact model identifier used across all A/B/C/D evaluation setups",
    )
    api_key_env: str = Field(
        default="GEMINI_API_KEY",
        description="Name of the environment variable storing the API key",
    )


class GenerationConfig(BaseModel):
    """Settings for synthetic prompt and argument generation (F2)."""

    count: int = Field(default=160, ge=40, description="Target number of raw prompts to generate")
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for prompt generation diversity",
    )
    prompts_per_tool: int = Field(default=3, ge=1, description="Prompts to generate per tool batch")
    max_words: int = Field(default=200, ge=20, description="Word limit for rewritten descriptions")


class EvaluationConfig(BaseModel):
    """Settings for benchmark evaluation runs (F4+F7)."""

    temperature: float = Field(
        default=0.0,
        description="LLM temperature for evaluation (enforced strictly to 0.0 for reproducibility)",
    )
    max_retries: int = Field(
        default=1,
        ge=0,
        le=1,
        description="Maximum schema validation retry attempts per prompt (capped at 1)",
    )

    @field_validator("temperature")
    @classmethod
    def validate_eval_temperature(cls, v: float) -> float:
        """Enforce deterministic evaluation temperature invariant."""
        if v != 0.0:
            raise ValueError(
                f"Evaluation temperature must be 0.0 for deterministic measurement, got {v}"
            )
        return v


class RetrievalConfig(BaseModel):
    """Settings for local embedding retrieval and top-k selection (F5, D1)."""

    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Local sentence-transformers embedding model (CPU-only)",
    )
    candidate_k_values: list[int] = Field(
        default_factory=lambda: [5, 10, 15, 20, 25, 30],
        description="Values of k swept during dev-set calibration",
    )
    target_recall: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Target recall@k threshold for selecting k",
    )


class SplitConfig(BaseModel):
    """Dataset splitting and freezing parameters (F3)."""

    heldout_size: int = Field(
        default=50,
        ge=50,
        description="Fixed held-out set size (minimum 50 required for statistical validity)",
    )
    min_dev_size: int = Field(
        default=40,
        ge=40,
        description="Minimum dev set size required to proceed with calibration",
    )
    seed: int = Field(default=42, description="Random seed for deterministic splitting")
    similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="SequenceMatcher similarity threshold for near-duplicate clustering",
    )
    max_cluster_size: int = Field(
        default=5,
        ge=1,
        description="Hard cap on prompts moved together in a near-duplicate cluster",
    )


class RateLimitConfig(BaseModel):
    """Rate limiting and exponential backoff configuration (F11b)."""

    rpm: int = Field(default=15, ge=1, description="Maximum requests per minute")
    rpd: int = Field(default=1500, ge=1, description="Maximum requests per day")
    tpm: int = Field(default=1000000, ge=1, description="Maximum tokens per minute")
    backoff_factor: float = Field(default=2.0, ge=1.0, description="Exponential backoff multiplier")
    max_backoff: float = Field(default=60.0, ge=1.0, description="Maximum backoff wait in seconds")


class ToolbenchConfig(BaseModel):
    """Root configuration model combining all Toolbench sub-configurations."""

    mcp: list[MCPServerConfig] = Field(default_factory=list)
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


def load_config(path: str | Path | None = None) -> ToolbenchConfig:
    """Load and validate Toolbench configuration from a YAML file.

    Args:
        path: Optional path to YAML config. If None, checks 'config.yaml' in CWD.
              If 'config.yaml' does not exist, returns defaults.

    Returns:
        Validated ToolbenchConfig instance.

    Raises:
        FileNotFoundError: If an explicit path is provided but does not exist.
        ValueError: If YAML syntax is invalid or schema validation fails.
    """
    config_path: Path | None = Path(path) if path is not None else None

    if config_path is None:
        default_candidate = Path("config.yaml")
        if default_candidate.exists():
            config_path = default_candidate

    if config_path is None:
        return ToolbenchConfig()

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
        parsed: Any = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as err:
        raise ValueError(f"Failed to parse YAML from {config_path}: {err}") from err

    if not isinstance(parsed, dict):
        raise ValueError(f"Configuration file {config_path} must define a YAML mapping/dict")

    return ToolbenchConfig.model_validate(parsed)
