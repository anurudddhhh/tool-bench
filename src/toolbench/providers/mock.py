"""Deterministic mock provider for offline testing and CI workflows (F11d).

Allows unit and integration test suites to simulate LLM responses, tool calls,
schema errors, and token metrics without network access or API credentials.
"""

from __future__ import annotations

import time
from typing import Any

from toolbench.providers.base import ProviderBase, ProviderResponse


class MockProvider(ProviderBase):
    """Deterministic LLM provider mock for unit testing and offline benchmarking."""

    def __init__(
        self,
        model_name: str = "mock-model",
        default_tool: str | None = None,
        default_args: dict[str, Any] | None = None,
        tool_mapping: dict[str, tuple[str, dict[str, Any]]] | None = None,
        simulate_error: str | None = None,
        simulate_text_only: str | None = None,
    ) -> None:
        """Initialize mock provider with programmable responses.

        Args:
            model_name: Mock model identifier string.
            default_tool: Fallback tool name returned if no mapping matches.
            default_args: Fallback arguments returned with default_tool.
            tool_mapping: Substring-to-(tool_name, args) mapping for dynamic responses.
            simulate_error: If set, raises or returns this error message.
            simulate_text_only: If set, returns pure text with no tool call.
        """
        super().__init__(model_name=model_name)
        self.default_tool = default_tool
        self.default_args = default_args or {}
        self.tool_mapping = tool_mapping or {}
        self.simulate_error = simulate_error
        self.simulate_text_only = simulate_text_only
        self.call_history: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, Any]] | str,
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Generate a deterministic response based on mock configuration."""
        start_time = time.perf_counter()
        normalized_messages = self.normalize_messages(messages)

        # Extract last prompt content for keyword matching
        last_prompt = ""
        if normalized_messages:
            last_prompt = str(normalized_messages[-1].get("content", ""))

        record = {
            "messages": normalized_messages,
            "tools": tools,
            "temperature": temperature,
            "prompt": last_prompt,
        }
        self.call_history.append(record)

        # 1. Error simulation
        if self.simulate_error:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ProviderResponse(
                chosen_tool=None,
                chosen_args={},
                error=self.simulate_error,
                latency_ms=max(latency_ms, 1.0),
            )

        # 2. Text-only simulation (no tool invocation)
        if self.simulate_text_only:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ProviderResponse(
                chosen_tool=None,
                chosen_args={},
                text_response=self.simulate_text_only,
                prompt_tokens=len(last_prompt.split()) + 10,
                completion_tokens=len(self.simulate_text_only.split()),
                latency_ms=max(latency_ms, 1.0),
            )

        # 3. Dynamic keyword-based tool mapping
        chosen_tool: str | None = None
        chosen_args: dict[str, Any] = {}

        for keyword, (mapped_tool, mapped_args) in self.tool_mapping.items():
            if keyword.lower() in last_prompt.lower():
                chosen_tool = mapped_tool
                chosen_args = mapped_args
                break

        # 4. Fallback to default tool or first exposed tool
        if chosen_tool is None:
            if self.default_tool is not None:
                chosen_tool = self.default_tool
                chosen_args = dict(self.default_args)
            elif tools:
                first_tool_name = str(tools[0].get("name", "unknown_tool"))
                chosen_tool = first_tool_name
                chosen_args = {}

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        prompt_tokens = len(last_prompt.split()) + len(tools) * 15
        completion_tokens = 20 if chosen_tool else 5

        return ProviderResponse(
            chosen_tool=chosen_tool,
            chosen_args=chosen_args,
            text_response=f"Invoking {chosen_tool}" if chosen_tool else "No tool chosen",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=max(latency_ms, 1.0),
        )
