"""Unit tests for multi-server aggregation, namespacing, and sanitization (F1.5)."""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from toolbench.config import MCPServerConfig
from toolbench.mcp_client import MultiServerAggregator, ToolInfo


def _make_mock_tool(name: str, desc: str = "", server: str = "") -> ToolInfo:
    return ToolInfo(
        name=name,
        description=desc,
        input_schema={"type": "object", "properties": {}},
        server_name=server,
        original_name=name,
    )


@pytest.mark.anyio
async def test_multi_server_aggregation_namespacing() -> None:
    """Verify multi-server tool namespacing as server_name__tool_name."""
    configs = [
        MCPServerConfig(name="filesystem", command="uvx", args=["mcp-fs"]),
        MCPServerConfig(name="git", command="uvx", args=["mcp-git"]),
    ]
    aggregator = MultiServerAggregator(configs)

    fs_tools = [
        _make_mock_tool("read_file", "Reads file", "filesystem"),
        _make_mock_tool("write_file", "Writes file", "filesystem"),
    ]
    git_tools = [
        _make_mock_tool("status", "Git status", "git"),
        _make_mock_tool("commit", "Git commit", "git"),
    ]

    with patch(
        "toolbench.mcp_client.MCPClient.list_tools_async",
        new=AsyncMock(side_effect=[fs_tools, git_tools]),
    ):
        tools = await aggregator.aggregate_async()

    assert len(tools) == 4
    names = [t.name for t in tools]
    assert names == [
        "filesystem__read_file",
        "filesystem__write_file",
        "git__status",
        "git__commit",
    ]

    # Verify original names and server names preserved
    assert tools[0].original_name == "read_file"
    assert tools[0].server_name == "filesystem"
    assert tools[2].original_name == "status"
    assert tools[2].server_name == "git"


@pytest.mark.anyio
async def test_multi_server_deduplication(caplog: pytest.LogCaptureFixture) -> None:
    """Verify duplicate namespaced tools are skipped with a warning."""
    configs = [MCPServerConfig(name="srv", command="uvx")]
    aggregator = MultiServerAggregator(configs)

    duplicate_tools = [
        _make_mock_tool("tool_x", "First", "srv"),
        _make_mock_tool("tool_x", "Duplicate", "srv"),
    ]

    mock_list_dup = AsyncMock(return_value=duplicate_tools)
    with (
        patch("toolbench.mcp_client.MCPClient.list_tools_async", new=mock_list_dup),
        caplog.at_level(logging.WARNING),
    ):
        tools = await aggregator.aggregate_async()

    assert len(tools) == 1
    assert tools[0].description == "First"
    assert "Duplicate tool 'srv__tool_x' discovered" in caplog.text


@pytest.mark.anyio
async def test_multi_server_partial_failure_tolerance(caplog: pytest.LogCaptureFixture) -> None:
    """Verify aggregator continues when a subset of servers fail."""
    configs = [
        MCPServerConfig(name="good_srv", command="uvx"),
        MCPServerConfig(name="broken_srv", command="uvx"),
    ]
    aggregator = MultiServerAggregator(configs)

    good_tools = [_make_mock_tool("ping", "Ping", "good_srv")]

    with (
        patch(
            "toolbench.mcp_client.MCPClient.list_tools_async",
            new=AsyncMock(side_effect=[good_tools, RuntimeError("Subprocess crashed")]),
        ),
        caplog.at_level(logging.WARNING),
    ):
        tools = await aggregator.aggregate_async()

    assert len(tools) == 1
    assert tools[0].name == "good_srv__ping"
    assert aggregator.failed_servers == ["broken_srv"]
    assert "MCP server 'broken_srv' failed during aggregation" in caplog.text


@pytest.mark.anyio
async def test_multi_server_all_failures_raises() -> None:
    """Verify RuntimeError is raised when all configured servers fail."""
    configs = [
        MCPServerConfig(name="srv1", command="uvx"),
        MCPServerConfig(name="srv2", command="uvx"),
    ]
    aggregator = MultiServerAggregator(configs)

    with patch(
        "toolbench.mcp_client.MCPClient.list_tools_async",
        side_effect=RuntimeError("Connection refused"),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            await aggregator.aggregate_async()
        assert "All 2 configured MCP servers failed to connect" in str(exc_info.value)


@pytest.mark.anyio
async def test_multi_server_sanitization_and_reverse_mapping() -> None:
    """Verify function name sanitization and reverse mapping."""
    configs = [MCPServerConfig(name="server-git", command="uvx")]
    aggregator = MultiServerAggregator(configs)

    raw_tools = [
        _make_mock_tool("diff-cached", "Diff tool", "server-git"),
        _make_mock_tool("diff_cached", "Diff tool with underscore", "server-git"),
    ]

    with patch(
        "toolbench.mcp_client.MCPClient.list_tools_async",
        new=AsyncMock(return_value=raw_tools),
    ):
        await aggregator.aggregate_async()

    # Original namespaced names:
    # 1. 'server-git__diff-cached'
    # 2. 'server-git__diff_cached'
    # Both normalize to 'server_git__diff_cached' base
    safe1 = aggregator._namespaced_to_safe["server-git__diff-cached"]
    safe2 = aggregator._namespaced_to_safe["server-git__diff_cached"]

    assert safe1 == "server_git__diff_cached"
    assert safe2 == "server_git__diff_cached_2"

    # Reverse mapping round-trip
    assert aggregator.reverse_map_tool_name(safe1) == "server-git__diff-cached"
    assert aggregator.reverse_map_tool_name(safe2) == "server-git__diff_cached"

    # get_tool lookup
    found = aggregator.get_tool("server-git__diff-cached")
    assert found is not None
    assert found.description == "Diff tool"


@pytest.mark.anyio
async def test_multi_server_tool_count_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Verify warning is logged when aggregated tool count is under 40."""
    configs = [MCPServerConfig(name="small_srv", command="uvx")]
    aggregator = MultiServerAggregator(configs)

    few_tools = [_make_mock_tool(f"t_{i}") for i in range(5)]

    mock_list_few = AsyncMock(return_value=few_tools)
    with (
        patch("toolbench.mcp_client.MCPClient.list_tools_async", new=mock_list_few),
        caplog.at_level(logging.WARNING),
    ):
        await aggregator.aggregate_async()

    assert "Aggregated tool count (5) is below the recommended 40-tool threshold" in caplog.text
