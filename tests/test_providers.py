"""Unit tests for LLM provider interface and implementations (F11d)."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from toolbench.cache import DiskCache
from toolbench.config import ToolbenchConfig
from toolbench.providers import (
    GeminiProvider,
    MockProvider,
    ProviderResponse,
    get_provider,
)


def test_provider_response_properties() -> None:
    """Verify ProviderResponse properties and serialization."""
    resp = ProviderResponse(
        chosen_tool="read_file",
        chosen_args={"path": "test.txt"},
        prompt_tokens=100,
        completion_tokens=20,
    )
    assert resp.has_tool_call is True
    as_dict = resp.to_dict()
    assert as_dict["chosen_tool"] == "read_file"
    assert as_dict["chosen_args"] == {"path": "test.txt"}
    assert as_dict["prompt_tokens"] == 100
    assert as_dict["cached"] is False

    no_tool = ProviderResponse(text_response="Just conversational text")
    assert no_tool.has_tool_call is False


def test_mock_provider_keyword_matching() -> None:
    """Verify MockProvider accurately routes prompts via keyword mapping."""
    mapping = {
        "read": ("filesystem__read_file", {"path": "/tmp/doc.txt"}),
        "commit": ("git__git_commit", {"message": "initial commit"}),
    }
    provider = MockProvider(tool_mapping=mapping)

    resp1 = provider.generate("Please read the file for me", tools=[])
    assert resp1.chosen_tool == "filesystem__read_file"
    assert resp1.chosen_args == {"path": "/tmp/doc.txt"}

    resp2 = provider.generate("Commit changes now", tools=[])
    assert resp2.chosen_tool == "git__git_commit"
    assert resp2.chosen_args == {"message": "initial commit"}


def test_mock_provider_fallback() -> None:
    """Verify MockProvider fallback behavior when no keywords match."""
    provider = MockProvider(
        default_tool="default_tool",
        default_args={"default_arg": 1},
    )

    tools = [{"name": "tool_x"}, {"name": "tool_y"}]
    resp = provider.generate("Unrelated prompt query", tools=tools)
    assert resp.chosen_tool == "default_tool"
    assert resp.chosen_args == {"default_arg": 1}

    # When no default is set, picks first available tool
    provider_no_default = MockProvider()
    resp_first = provider_no_default.generate("Unrelated prompt", tools=tools)
    assert resp_first.chosen_tool == "tool_x"


def test_mock_provider_simulations() -> None:
    """Verify MockProvider simulates errors and text-only outputs."""
    # Simulate text-only
    text_provider = MockProvider(simulate_text_only="I cannot use tools.")
    resp_text = text_provider.generate("test prompt", tools=[{"name": "tool_a"}])
    assert resp_text.chosen_tool is None
    assert resp_text.text_response == "I cannot use tools."

    # Simulate error
    error_provider = MockProvider(simulate_error="API Rate limit 429")
    resp_err = error_provider.generate("test prompt", tools=[])
    assert resp_err.chosen_tool is None
    assert resp_err.error == "API Rate limit 429"


def test_mock_provider_call_history() -> None:
    """Verify MockProvider records invocation history."""
    provider = MockProvider()
    provider.generate("Prompt 1", tools=[{"name": "t1"}], temperature=0.0)
    provider.generate("Prompt 2", tools=[{"name": "t2"}], temperature=0.7)

    assert len(provider.call_history) == 2
    assert provider.call_history[0]["temperature"] == 0.0
    assert provider.call_history[1]["temperature"] == 0.7


def test_gemini_provider_convert_tools() -> None:
    """Verify conversion of JSON schemas and MCP input schemas to Gemini format."""
    provider = GeminiProvider(api_key="mock_key")
    raw_tools = [
        {
            "name": "filesystem__read_file",
            "description": "Read file contents",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
        {
            "name": "git__status",
            "description": "Check git status",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]

    gemini_tools = provider._convert_tools(raw_tools)
    assert len(gemini_tools) == 1
    decls = gemini_tools[0].function_declarations
    assert len(decls) == 2
    assert decls[0].name == "filesystem__read_file"
    assert decls[1].name == "git__status"


def test_gemini_provider_missing_key_raises() -> None:
    """Verify ValueError when API key is completely absent."""
    with patch.dict("os.environ", {}, clear=True):
        provider = GeminiProvider(api_key=None, api_key_env="NONEXISTENT_KEY_VAR")
        with pytest.raises(ValueError) as exc_info:
            provider.generate("test", tools=[])
        assert "Gemini API key not found" in str(exc_info.value)


def test_gemini_provider_caching(tmp_path: Path) -> None:
    """Verify that GeminiProvider reads from and populates the disk cache."""
    cache = DiskCache(cache_dir=tmp_path / "cache")
    provider = GeminiProvider(api_key="mock_key", cache=cache)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_call = MagicMock()
    mock_call.name = "read_file"
    mock_call.args = {"path": "doc.txt"}
    mock_response.function_calls = [mock_call]
    mock_response.text = None
    mock_response.usage_metadata.prompt_token_count = 50
    mock_response.usage_metadata.candidates_token_count = 10

    mock_client.models.generate_content.return_value = mock_response
    provider._client = mock_client

    tools: list[dict[str, Any]] = [{"name": "read_file", "parameters": {}}]

    # 1st call: Cache miss, hits API client
    resp1 = provider.generate("Read doc.txt", tools=tools, temperature=0.0)
    assert resp1.chosen_tool == "read_file"
    assert resp1.cached is False
    assert mock_client.models.generate_content.call_count == 1
    assert cache.size() == 1

    # 2nd call: Cache hit, does NOT call API client
    resp2 = provider.generate("Read doc.txt", tools=tools, temperature=0.0)
    assert resp2.chosen_tool == "read_file"
    assert resp2.chosen_args == {"path": "doc.txt"}
    assert resp2.cached is True
    # Call count remains 1!
    assert mock_client.models.generate_content.call_count == 1


def test_get_provider_factory() -> None:
    """Verify get_provider() factory instantiates correct provider types."""
    mock_cfg = ToolbenchConfig()
    mock_cfg.model.provider = "mock"
    mock_cfg.model.name = "mock-gpt"

    provider = get_provider(mock_cfg)
    assert isinstance(provider, MockProvider)
    assert provider.model_name == "mock-gpt"

    gemini_cfg = ToolbenchConfig()
    gemini_cfg.model.provider = "gemini"
    gemini_provider = get_provider(gemini_cfg)
    assert isinstance(gemini_provider, GeminiProvider)

    bad_cfg = ToolbenchConfig()
    bad_cfg.model.provider = "unsupported_provider"
    with pytest.raises(ValueError) as exc_info:
        get_provider(bad_cfg)
    assert "Unsupported provider" in str(exc_info.value)


def test_gemini_sanitize_tool_name_rules() -> None:
    """Verify Gemini function name sanitization rules."""
    used: dict[str, str] = {}

    # Basic invalid characters
    assert (
        GeminiProvider.sanitize_tool_name("mcp-server-git__commit", used)
        == "mcp_server_git__commit"
    )

    # Leading digit
    assert GeminiProvider.sanitize_tool_name("123_tool", used) == "tool_123_tool"

    # Truncation at 64 characters
    long_name = "a" * 80
    sanitized_long = GeminiProvider.sanitize_tool_name(long_name, used)
    assert len(sanitized_long) == 64
    assert sanitized_long == "a" * 64

    # Empty name fallback
    assert GeminiProvider.sanitize_tool_name("", used) == "unnamed_tool"


def test_gemini_sanitize_tool_name_collisions() -> None:
    """Verify collision resolution disambiguates differing tools with numeric suffixes."""
    used: dict[str, str] = {}

    # Two different tools that normalize to the same string
    name1 = GeminiProvider.sanitize_tool_name("tool_a", used)
    name2 = GeminiProvider.sanitize_tool_name("tool-a", used)
    name3 = GeminiProvider.sanitize_tool_name("tool.a", used)

    assert name1 == "tool_a"
    assert name2 == "tool_a_2"
    assert name3 == "tool_a_3"

    # Re-sanitizing the exact same original name returns its already-assigned candidate
    assert GeminiProvider.sanitize_tool_name("tool-a", used) == "tool_a_2"


def test_gemini_provider_reverse_mapping() -> None:
    """Verify GeminiProvider reverse-maps sanitized function names back to original MCP names."""
    provider = GeminiProvider(api_key="mock_key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_call = MagicMock()
    # Simulated API returns the sanitized name
    mock_call.name = "mcp_server_git__commit"
    mock_call.args = {"message": "fix"}
    mock_response.function_calls = [mock_call]
    mock_response.text = None
    mock_response.usage_metadata.prompt_token_count = 30
    mock_response.usage_metadata.candidates_token_count = 15

    mock_client.models.generate_content.return_value = mock_response
    provider._client = mock_client

    tools: list[dict[str, Any]] = [
        {"name": "mcp-server-git__commit", "parameters": {}},
    ]

    resp = provider.generate("Commit changes", tools=tools)
    # Must be reverse-mapped to the original raw name with hyphen
    assert resp.chosen_tool == "mcp-server-git__commit"
    assert resp.chosen_args == {"message": "fix"}

