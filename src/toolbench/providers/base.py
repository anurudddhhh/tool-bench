"""Abstract base class and data structures for LLM providers (F11d).

Defines the contract for LLM invocation across Gemini and deterministic Mock
providers, ensuring identical response structures and token accounting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResponse:
    """Standardized response from an LLM invocation."""

    chosen_tool: str | None = None
    chosen_args: dict[str, Any] = field(default_factory=dict)
    text_response: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    cached: bool = False
    raw_response: Any = None

    @property
    def has_tool_call(self) -> bool:
        """Return True if the model returned an explicit tool invocation."""
        return self.chosen_tool is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert response to a serializable dictionary for result recording."""
        return {
            "chosen_tool": self.chosen_tool,
            "chosen_args": self.chosen_args,
            "text_response": self.text_response,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "cached": self.cached,
        }


class ProviderBase(ABC):
    """Abstract interface implemented by all LLM providers (Gemini, Mock)."""

    def __init__(self, model_name: str) -> None:
        """Initialize provider with target model identifier.

        Args:
            model_name: Exact model identifier (e.g. 'gemini-2.5-flash', 'mock-model').
        """
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]] | str,
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Send prompt messages and tool definitions to the LLM.

        Args:
            messages: List of message dictionaries (`[{"role": "user", "content": ...}]`)
                      or a single user prompt string.
            tools: List of exposed tool definitions (JSON schema specifications).
            temperature: Sampling temperature (0.0 for evaluation, 0.7 for generation).

        Returns:
            Standardized ProviderResponse object.
        """
        pass

    @staticmethod
    def normalize_messages(messages: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
        """Normalize input to standard list of message dictionaries."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return messages
