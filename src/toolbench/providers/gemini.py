"""Google Gemini provider implementation using the official google-genai SDK (F11d).

Integrates rate limiting (F11b), response disk caching (F11a), and bidirectional
function-name sanitization directly into Gemini tool-calling execution.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from google import genai
from google.genai import types

from toolbench.cache import DiskCache
from toolbench.providers.base import ProviderBase, ProviderResponse
from toolbench.rate_limit import RateLimiter

logger = logging.getLogger(__name__)


class GeminiProvider(ProviderBase):
    """Production provider using Google Gemini models via google-genai."""

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash-lite",
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        rate_limiter: RateLimiter | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        """Initialize the Gemini provider.

        Args:
            model_name: Exact Gemini model identifier string.
            api_key: Optional explicit API key string.
            api_key_env: Environment variable to read API key from if not provided.
            rate_limiter: Optional RateLimiter instance for RPM/RPD throttling & backoff.
            cache: Optional DiskCache instance for response memoization.
        """
        super().__init__(model_name=model_name)
        self.api_key_env = api_key_env
        self._api_key = api_key or os.environ.get(api_key_env)
        self.rate_limiter = rate_limiter or RateLimiter()
        self.cache = cache
        self._client: genai.Client | None = None
        # Bidirectional mapping for function-name sanitization
        self._safe_to_original: dict[str, str] = {}
        self._original_to_safe: dict[str, str] = {}

    def _get_client(self) -> genai.Client:
        """Lazily initialize and return the official genai.Client with v1beta options."""
        if self._client is None:
            key = self._api_key or os.environ.get(self.api_key_env)
            if not key:
                raise ValueError(
                    f"Gemini API key not found. Please set the {self.api_key_env} "
                    "environment variable or pass api_key directly."
                )
            self._client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(api_version="v1beta"),
            )
        return self._client

    @staticmethod
    def sanitize_tool_name(raw_name: str, used_names: dict[str, str]) -> str:
        """Sanitize a tool name to comply with Gemini API function-name constraints.

        Rule: ^[a-zA-Z_][a-zA-Z0-9_]{0,63}$
        1. Replace characters outside [a-zA-Z0-9_] with '_'.
        2. If starting with a digit, prefix with 'tool_'.
        3. Truncate to maximum 64 characters.
        4. If a collision occurs with a different tool name, append '_2', '_3', etc.
        5. Record mapping in used_names: safe_name -> raw_name.

        Args:
            raw_name: Original tool name (e.g. 'mcp-server-git__commit').
            used_names: Dictionary tracking current safe_name -> raw_name mappings.

        Returns:
            API-safe function name string.
        """
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", raw_name)
        if cleaned and cleaned[0].isdigit():
            cleaned = f"tool_{cleaned}"
        if not cleaned:
            cleaned = "unnamed_tool"
        base = cleaned[:64]

        candidate = base
        suffix_counter = 2
        while candidate in used_names and used_names[candidate] != raw_name:
            suffix = f"_{suffix_counter}"
            candidate = f"{base[: 64 - len(suffix)]}{suffix}"
            suffix_counter += 1

        used_names[candidate] = raw_name
        return candidate

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[types.Tool]:
        """Convert standard tool schemas into Gemini FunctionDeclaration format."""
        if not tools:
            return []

        # Reset per-batch mappings
        self._safe_to_original.clear()
        self._original_to_safe.clear()

        declarations: list[types.FunctionDeclaration] = []
        for tool in tools:
            original_name = str(tool.get("name", ""))
            safe_name = self.sanitize_tool_name(original_name, self._safe_to_original)
            self._original_to_safe[original_name] = safe_name

            description = str(tool.get("description", ""))
            schema = tool.get("parameters") or tool.get("input_schema") or {}

            # Gemini parameters must be an object type schema
            if not isinstance(schema, dict) or not schema:
                schema = {"type": "object", "properties": {}}
            elif "type" not in schema:
                schema = {"type": "object", "properties": schema.get("properties", {})}

            declarations.append(
                types.FunctionDeclaration(
                    name=safe_name,
                    description=description,
                    parameters=schema,
                )
            )

        return [types.Tool(function_declarations=declarations)]

    def generate(
        self,
        messages: list[dict[str, Any]] | str,
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Send prompt and tools to Gemini, utilizing cache and rate limiting."""
        normalized_messages = self.normalize_messages(messages)
        prompt_text = "\n".join(
            str(m.get("content", "")) for m in normalized_messages
        )

        # 1. Check disk cache if available
        cache_key: str | None = None
        if self.cache:
            cache_key = self.cache.compute_key(
                prompt=normalized_messages,
                tools=tools,
                model=self.model_name,
                temperature=temperature,
            )
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit for key %s", cache_key)
                return ProviderResponse(
                    chosen_tool=cached_result.get("chosen_tool"),
                    chosen_args=cached_result.get("chosen_args", {}),
                    text_response=cached_result.get("text_response"),
                    prompt_tokens=cached_result.get("prompt_tokens", 0),
                    completion_tokens=cached_result.get("completion_tokens", 0),
                    latency_ms=cached_result.get("latency_ms", 0.0),
                    error=cached_result.get("error"),
                    cached=True,
                )

        client = self._get_client()
        gemini_tools = self._convert_tools(tools)

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools

        config = types.GenerateContentConfig(**config_kwargs)

        # 2. Invoke Gemini under rate-limited backoff execution
        start_time = time.perf_counter()

        def _call_api() -> types.GenerateContentResponse:
            return client.models.generate_content(
                model=self.model_name,
                contents=prompt_text,
                config=config,
            )

        try:
            api_response: types.GenerateContentResponse = (
                self.rate_limiter.execute_with_retry(_call_api)
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error("Gemini API invocation failed: %s", exc)
            return ProviderResponse(
                chosen_tool=None,
                chosen_args={},
                error=str(exc),
                latency_ms=latency_ms,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Extract function calls, reverse-map safe names, text, and tokens
        chosen_tool: str | None = None
        chosen_args: dict[str, Any] = {}

        if api_response.function_calls:
            first_call = api_response.function_calls[0]
            api_name = first_call.name
            # Reverse-map to original tool name
            chosen_tool = self._safe_to_original.get(api_name, api_name)
            chosen_args = dict(first_call.args or {})

        text_response: str | None = None
        try:
            text_response = api_response.text
        except Exception:
            # Function calls without text parts may raise on response.text
            pass

        prompt_tokens = 0
        completion_tokens = 0
        if api_response.usage_metadata:
            prompt_tokens = api_response.usage_metadata.prompt_token_count or 0
            completion_tokens = api_response.usage_metadata.candidates_token_count or 0

        response = ProviderResponse(
            chosen_tool=chosen_tool,
            chosen_args=chosen_args,
            text_response=text_response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cached=False,
            raw_response=api_response,
        )

        # 4. Save to disk cache if available
        if self.cache and cache_key:
            self.cache.set(cache_key, response.to_dict())

        return response
