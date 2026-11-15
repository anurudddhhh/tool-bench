"""Unit tests for single MCP server client and tool metadata extraction (F1)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from toolbench.config import MCPServerConfig
from toolbench.mcp_client import (
    MCPClient,
    ToolInfo,
    normalize_server_command,
)


def test_tool_info_properties() -> None:
    """Verify ToolInfo attributes, defaults, and dictionary serialization."""
    tool = ToolInfo(
        name="read_file",
        description="Reads contents of a file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        server_name="filesystem",
        original_name="read_file",
    )
    assert tool.name == "read_file"
    assert tool.server_name == "filesystem"
    assert tool.original_name == "read_file"

    data = tool.to_dict()
    assert data["name"] == "read_file"
    assert data["description"] == "Reads contents of a file"
    assert data["parameters"] == {"type": "object", "properties": {"path": {"type": "string"}}}
    assert data["input_schema"] == data["parameters"]
    assert data["server_name"] == "filesystem"
    assert data["original_name"] == "read_file"

    # Default schema fallback
    minimal_tool = ToolInfo(name="simple_tool")
    assert minimal_tool.description == ""
    assert minimal_tool.input_schema == {"type": "object", "properties": {}}


def test_normalize_server_command_windows() -> None:
    """Verify command wrapping for npm/npx on Windows."""
    with patch("sys.platform", "win32"):
        cmd, args = normalize_server_command("npx", ["-y", "@modelcontextprotocol/server-git"])
        assert cmd == "cmd"
        assert args == ["/c", "npx", "-y", "@modelcontextprotocol/server-git"]

        cmd_bat, args_bat = normalize_server_command("server.bat", ["--port", "8080"])
        assert cmd_bat == "cmd"
        assert args_bat == ["/c", "server.bat", "--port", "8080"]

        # Native binary (uvx or docker) is not wrapped
        cmd_uvx, args_uvx = normalize_server_command("uvx", ["mcp-server-git"])
        assert cmd_uvx == "uvx"
        assert args_uvx == ["mcp-server-git"]


def test_normalize_server_command_posix() -> None:
    """Verify command remains untouched on non-Windows platforms."""
    with patch("sys.platform", "linux"):
        cmd, args = normalize_server_command("npx", ["-y", "server"])
        assert cmd == "npx"
        assert args == ["-y", "server"]


@pytest.mark.anyio
async def test_mcp_client_list_tools_async_success() -> None:
    """Verify async tool extraction and metadata extraction."""
    cfg = MCPServerConfig(
        name="test_fs",
        command="uvx",
        args=["mcp-server-filesystem", "/tmp"],
    )
    client = MCPClient(cfg)

    # Mock MCP response tools
    mock_t1 = MagicMock()
    mock_t1.name = "read_file"
    mock_t1.description = "Read file from disk"
    mock_t1.inputSchema = {"type": "object", "properties": {"path": {"type": "string"}}}

    # Mock tool with None description and inputSchema
    mock_t2 = MagicMock()
    mock_t2.name = "ping"
    mock_t2.description = None
    mock_t2.inputSchema = None

    mock_response = MagicMock()
    mock_response.tools = [mock_t1, mock_t2]

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools.return_value = mock_response

    class MockStdioContext:
        async def __aenter__(self) -> tuple[MagicMock, MagicMock]:
            return MagicMock(), MagicMock()

        async def __aexit__(self, *args: object) -> None:
            pass

    class MockSessionContext:
        async def __aenter__(self) -> AsyncMock:
            return mock_session

        async def __aexit__(self, *args: object) -> None:
            pass

    with (
        patch("toolbench.mcp_client.stdio_client", return_value=MockStdioContext()),
        patch("toolbench.mcp_client.ClientSession", return_value=MockSessionContext()),
    ):
        tools = await client.list_tools_async()

    assert len(tools) == 2
    assert tools[0].name == "read_file"
    assert tools[0].server_name == "test_fs"
    assert tools[0].original_name == "read_file"
    assert tools[0].description == "Read file from disk"

    # Verify fallbacks on tool 2
    assert tools[1].name == "ping"
    assert tools[1].description == ""
    assert tools[1].input_schema == {"type": "object", "properties": {}}

    # Verify session caching (calling again returns cached tools)
    cached_tools = await client.list_tools_async()
    assert len(cached_tools) == 2
    assert mock_session.list_tools.call_count == 1


def test_mcp_client_launch_failure_raises() -> None:
    """Verify RuntimeError is raised when the server subprocess fails to connect."""
    cfg = MCPServerConfig(
        name="failing_server",
        command="invalid_binary_name_that_does_not_exist",
    )
    client = MCPClient(cfg)

    mock_err = FileNotFoundError("Executable not found")
    with patch("toolbench.mcp_client.stdio_client", side_effect=mock_err):
        with pytest.raises(RuntimeError) as exc_info:
            client.list_tools()
        assert "MCP server 'failing_server' failed to launch or initialize" in str(exc_info.value)
        assert "Executable not found" in str(exc_info.value)
