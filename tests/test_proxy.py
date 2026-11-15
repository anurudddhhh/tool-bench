"""Unit tests for the Retrieval Proxy MCP Server (D3)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from toolbench.mcp_client import ToolInfo
from toolbench.proxy import RetrievalProxyServer, create_proxy_server


def _make_tool(name: str, desc: str = "") -> ToolInfo:
    return ToolInfo(
        name=name,
        description=desc,
        input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
    )


@pytest.fixture
def sample_tools() -> list[ToolInfo]:
    return [
        _make_tool("fs_read", "Read contents of a file on local filesystem"),
        _make_tool("fs_write", "Write contents to a file on local filesystem"),
        _make_tool("git_commit", "Record changes to the repository"),
    ]


@pytest.fixture
def mock_retriever(sample_tools: list[ToolInfo]) -> MagicMock:
    ret = MagicMock()
    ret.retrieve_top_k.return_value = [sample_tools[0], sample_tools[1]]
    return ret


@pytest.mark.anyio
async def test_proxy_server_tool_registration(
    sample_tools: list[ToolInfo],
    mock_retriever: MagicMock,
) -> None:
    """Verify that proxy registers search_tools, call_tool, and list_all_tools."""
    proxy = RetrievalProxyServer(tools=sample_tools, retriever=mock_retriever)
    server = proxy.get_server()

    tools = await server.list_tools()
    tool_names = {t.name for t in tools}

    assert "search_tools" in tool_names
    assert "call_tool" in tool_names
    assert "list_all_tools" in tool_names


@pytest.mark.anyio
async def test_proxy_search_tools(
    sample_tools: list[ToolInfo],
    mock_retriever: MagicMock,
) -> None:
    """Verify search_tools retrieves top-k tools and returns serialized JSON schema."""
    proxy = RetrievalProxyServer(tools=sample_tools, retriever=mock_retriever)
    server = proxy.get_server()

    response = await server.call_tool("search_tools", {"query": "read files", "k": 2})
    assert not response.is_error
    assert response.content

    data = json.loads(response.content[0].text)
    assert len(data) == 2
    assert data[0]["name"] == "fs_read"
    assert "input_schema" in data[0]
    mock_retriever.retrieve_top_k.assert_called_once_with(query="read files", k=2)


@pytest.mark.anyio
async def test_proxy_call_tool_passthrough(
    sample_tools: list[ToolInfo],
    mock_retriever: MagicMock,
) -> None:
    """Verify call_tool dispatches to tool_caller with server, tool, and args."""
    mock_caller = MagicMock(return_value={"bytes_written": 42})

    proxy = RetrievalProxyServer(
        tools=sample_tools,
        retriever=mock_retriever,
        tool_caller=mock_caller,
    )
    server = proxy.get_server()

    response = await server.call_tool(
        "call_tool",
        {
            "server": "fs",
            "tool": "fs_write",
            "arguments": {"path": "test.txt", "content": "hello"},
        },
    )
    assert not response.is_error
    result = json.loads(response.content[0].text)

    assert result["status"] == "success"
    assert result["result"] == {"bytes_written": 42}
    mock_caller.assert_called_once_with(
        "fs", "fs_write", {"path": "test.txt", "content": "hello"}
    )


@pytest.mark.anyio
async def test_proxy_call_tool_async_caller(
    sample_tools: list[ToolInfo],
    mock_retriever: MagicMock,
) -> None:
    """Verify call_tool properly awaits asynchronous tool caller functions."""
    async_caller = AsyncMock(return_value="commit 8096236")

    proxy = RetrievalProxyServer(
        tools=sample_tools,
        retriever=mock_retriever,
        tool_caller=async_caller,
    )
    server = proxy.get_server()

    response = await server.call_tool(
        "call_tool",
        {
            "server": "git",
            "tool": "git_commit",
            "arguments": {"message": "feat: initial commit"},
        },
    )
    assert not response.is_error
    result = json.loads(response.content[0].text)
    assert result["status"] == "success"
    assert result["result"] == "commit 8096236"


@pytest.mark.anyio
async def test_proxy_call_tool_simulation_fallback(
    sample_tools: list[ToolInfo],
    mock_retriever: MagicMock,
) -> None:
    """Verify call_tool produces a simulated routed response when tool_caller is None."""
    proxy = RetrievalProxyServer(
        tools=sample_tools,
        retriever=mock_retriever,
        tool_caller=None,
    )
    server = proxy.get_server()

    response = await server.call_tool(
        "call_tool",
        {
            "server": "test_server",
            "tool": "fs_read",
            "arguments": {"file": "sample.py"},
        },
    )
    assert not response.is_error
    result = json.loads(response.content[0].text)
    assert result["status"] == "routed"
    assert result["server"] == "test_server"
    assert result["tool"] == "fs_read"


@pytest.mark.anyio
async def test_proxy_call_tool_error_handling(
    sample_tools: list[ToolInfo],
    mock_retriever: MagicMock,
) -> None:
    """Verify call_tool captures and safely serializes backend execution errors."""
    mock_caller = MagicMock(side_effect=RuntimeError("Connection refused by MCP subprocess"))

    proxy = RetrievalProxyServer(
        tools=sample_tools,
        retriever=mock_retriever,
        tool_caller=mock_caller,
    )
    server = proxy.get_server()

    response = await server.call_tool(
        "call_tool",
        {"server": "fs", "tool": "fs_read", "arguments": {}},
    )
    assert not response.is_error
    result = json.loads(response.content[0].text)
    assert result["status"] == "error"
    assert "Connection refused" in result["error"]


@pytest.mark.anyio
async def test_proxy_list_all_tools(
    sample_tools: list[ToolInfo],
    mock_retriever: MagicMock,
) -> None:
    """Verify list_all_tools meta-tool lists all available backend tools."""
    proxy = RetrievalProxyServer(tools=sample_tools, retriever=mock_retriever)
    server = proxy.get_server()

    response = await server.call_tool("list_all_tools", {})
    assert not response.is_error
    result = json.loads(response.content[0].text)
    assert result["total_tools"] == 3
    names = [t["name"] for t in result["tools"]]
    assert "fs_read" in names
    assert "git_commit" in names


def test_create_proxy_server_factory(
    sample_tools: list[ToolInfo],
    mock_retriever: MagicMock,
) -> None:
    """Verify create_proxy_server builds a valid MCPServer instance."""
    server = create_proxy_server(tools=sample_tools, retriever=mock_retriever)
    assert server.name == "toolbench-retrieval-proxy"
