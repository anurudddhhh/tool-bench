"""Model Context Protocol (MCP) client and multi-server aggregator (F1, F1.5).

Connects to MCP servers via stdio transport using the official `mcp` SDK, extracts
tool schemas, namespaces tools across multiple servers (`server__tool`), handles
Windows subprocess command adaptations, and manages bidirectional function-name
sanitization for downstream LLM providers.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from toolbench.config import MCPServerConfig

logger = logging.getLogger(__name__)


class ToolInfo(BaseModel):
    """Structured metadata representing an MCP tool."""

    name: str = Field(
        ...,
        description="Tool name (namespaced as server__tool in multi-server configurations)",
    )
    description: str = Field(
        default="",
        description="Human-readable explanation of tool behavior and usage",
    )
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema defining the accepted input arguments",
    )
    server_name: str = Field(
        default="",
        description="Originating MCP server name",
    )
    original_name: str = Field(
        default="",
        description="Original un-namespaced tool name defined on the server",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a standard dictionary compatible with LLM provider formats."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "input_schema": self.input_schema,
            "server_name": self.server_name,
            "original_name": self.original_name,
        }


def normalize_server_command(
    command: str,
    args: list[str],
) -> tuple[str, list[str]]:
    """Normalize command and arguments for platform-specific execution.

    On Windows, npm/npx commands need 'cmd /c' wrapping to execute properly as
    subprocess binaries unless already wrapped.
    """
    if sys.platform == "win32":
        cmd_lower = command.lower()
        is_script = cmd_lower.endswith(".cmd") or cmd_lower.endswith(".bat")
        if cmd_lower in ("npx", "npm") or is_script:
            return "cmd", ["/c", command, *args]
    return command, list(args)


class MCPClient:
    """Client for connecting to a single MCP server over stdio transport (F1)."""

    def __init__(self, config: MCPServerConfig) -> None:
        """Initialize the MCP client with server configuration.

        Args:
            config: MCPServerConfig detailing command, args, and environment.
        """
        self.config = config
        self._cached_tools: list[ToolInfo] | None = None

    def _build_params(self) -> StdioServerParameters:
        """Construct StdioServerParameters with platform-normalized commands."""
        cmd, args = normalize_server_command(self.config.command, self.config.args)
        # Merge parent process environment with configured env vars
        env = {**os.environ, **self.config.env}
        return StdioServerParameters(command=cmd, args=args, env=env)

    async def list_tools_async(self) -> list[ToolInfo]:
        """Asynchronously connect to the server, query tools, and return normalized metadata."""
        if self._cached_tools is not None:
            return list(self._cached_tools)

        params = self._build_params()
        logger.debug(
            "Connecting to MCP server '%s' (%s %s)...",
            self.config.name,
            params.command,
            " ".join(params.args),
        )

        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    response = await session.list_tools()

                    tools: list[ToolInfo] = []
                    for t in response.tools:
                        schema = t.inputSchema or {"type": "object", "properties": {}}
                        tools.append(
                            ToolInfo(
                                name=t.name,
                                description=t.description or "",
                                input_schema=schema,
                                server_name=self.config.name,
                                original_name=t.name,
                            )
                        )

                    self._cached_tools = tools
                    logger.info(
                        "Discovered %d tools from MCP server '%s'.",
                        len(tools),
                        self.config.name,
                    )
                    return list(tools)
        except Exception as err:
            logger.error(
                "Failed to connect to MCP server '%s': %s",
                self.config.name,
                err,
            )
            raise RuntimeError(
                f"MCP server '{self.config.name}' failed to launch or initialize: {err}"
            ) from err

    def list_tools(self) -> list[ToolInfo]:
        """Synchronously list tools from the MCP server."""
        return anyio.run(self.list_tools_async)


class MultiServerAggregator:
    """Aggregates and namespaces tools from multiple MCP servers (F1.5).

    Namespaces tool names as 'server_name__tool_name', deduplicates collisions,
    tolerates individual server launch failures, and maintains bidirectional
    function-name sanitization mappings for Gemini.
    """

    def __init__(self, server_configs: list[MCPServerConfig]) -> None:
        """Initialize aggregator with list of server configurations.

        Args:
            server_configs: List of MCPServerConfig instances.
        """
        self.server_configs = server_configs
        self._tools: list[ToolInfo] = []
        self._failed_servers: list[str] = []
        self._safe_to_namespaced: dict[str, str] = {}
        self._namespaced_to_safe: dict[str, str] = {}

    @property
    def tools(self) -> list[ToolInfo]:
        """Return the current aggregated list of namespaced tools."""
        return list(self._tools)

    @property
    def failed_servers(self) -> list[str]:
        """Return list of server names that failed to connect."""
        return list(self._failed_servers)

    @staticmethod
    def sanitize_name(raw_name: str, used_names: dict[str, str]) -> str:
        """Sanitize a tool name into a Gemini-compliant identifier.

        Pattern: ^[a-zA-Z_][a-zA-Z0-9_]{0,63}$
        1. Non-alphanumeric/non-underscore characters converted to '_'.
        2. Leading digits prefixed with 'tool_'.
        3. Max length 64 characters.
        4. Suffix disambiguation (_2, _3) on collisions.
        """
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", raw_name)
        if cleaned and cleaned[0].isdigit():
            cleaned = f"tool_{cleaned}"
        if not cleaned:
            cleaned = "unnamed_tool"
        base = cleaned[:64]

        candidate = base
        suffix = 2
        while candidate in used_names and used_names[candidate] != raw_name:
            suffix_str = f"_{suffix}"
            candidate = f"{base[: 64 - len(suffix_str)]}{suffix_str}"
            suffix += 1

        used_names[candidate] = raw_name
        return candidate

    def _build_sanitization_mappings(self) -> None:
        """Recompute bidirectional sanitization mappings for all aggregated tools."""
        self._safe_to_namespaced.clear()
        self._namespaced_to_safe.clear()

        for tool in self._tools:
            safe = self.sanitize_name(tool.name, self._safe_to_namespaced)
            self._namespaced_to_safe[tool.name] = safe

    def reverse_map_tool_name(self, safe_name: str) -> str:
        """Reverse-map a sanitized API function name back to its namespaced name."""
        return self._safe_to_namespaced.get(safe_name, safe_name)

    def get_tool(self, namespaced_name: str) -> ToolInfo | None:
        """Retrieve ToolInfo by its namespaced name."""
        for t in self._tools:
            if t.name == namespaced_name:
                return t
        return None

    async def aggregate_async(self) -> list[ToolInfo]:
        """Asynchronously connect to all configured servers and aggregate their tools.

        Tolerates individual server failures (logging a warning) as long as at least
        one server successfully connects.
        """
        self._tools.clear()
        self._failed_servers.clear()

        seen_names: set[str] = set()

        for srv in self.server_configs:
            client = MCPClient(srv)
            try:
                discovered = await client.list_tools_async()
                for tool in discovered:
                    namespaced_name = f"{srv.name}__{tool.name}"
                    if namespaced_name in seen_names:
                        logger.warning(
                            "Duplicate tool '%s' discovered from server '%s', skipping duplicate.",
                            namespaced_name,
                            srv.name,
                        )
                        continue

                    seen_names.add(namespaced_name)
                    namespaced_tool = ToolInfo(
                        name=namespaced_name,
                        description=tool.description,
                        input_schema=tool.input_schema,
                        server_name=srv.name,
                        original_name=tool.name,
                    )
                    self._tools.append(namespaced_tool)
            except Exception as err:
                logger.warning(
                    "MCP server '%s' failed during aggregation: %s",
                    srv.name,
                    err,
                )
                self._failed_servers.append(srv.name)

        if not self._tools and self.server_configs:
            raise RuntimeError(
                f"All {len(self.server_configs)} configured MCP servers failed to connect: "
                f"{', '.join(self._failed_servers)}"
            )

        if len(self._tools) < 40:
            logger.warning(
                "Aggregated tool count (%d) is below the recommended 40-tool threshold. "
                "Retrieval evaluation may encounter a ceiling effect.",
                len(self._tools),
            )

        self._build_sanitization_mappings()
        logger.info(
            "Successfully aggregated %d tools across %d servers (%d failed).",
            len(self._tools),
            len(self.server_configs) - len(self._failed_servers),
            len(self._failed_servers),
        )
        return list(self._tools)

    def aggregate(self) -> list[ToolInfo]:
        """Synchronously aggregate tools across all configured servers."""
        return anyio.run(self.aggregate_async)
