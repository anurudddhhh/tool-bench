"""Retrieval Proxy MCP Server (D3).

Provides a meta-MCP server that intercepts large toolsets and exposes:
1. `search_tools(query: str, k: int = 5)`: Semantic retrieval returning only the top-k
   relevant tools with their descriptions and input schemas.
2. `call_tool(server: str, tool: str, arguments: dict)`: Transparent pass-through execution
   to the target MCP backend server.
3. `list_all_tools()`: Meta-tool summarizing all managed tool endpoints.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Coroutine

from mcp.server.mcpserver import MCPServer

from toolbench.mcp_client import ToolInfo
from toolbench.retrieve import ToolRetriever

logger = logging.getLogger(__name__)

ToolCaller = Callable[[str, str, dict[str, Any]], Coroutine[Any, Any, Any] | Any]


class RetrievalProxyServer:
    """Orchestrates semantic retrieval and pass-through tool execution as an MCP server."""

    def __init__(
        self,
        tools: list[ToolInfo],
        retriever: ToolRetriever | None = None,
        tool_caller: ToolCaller | None = None,
        server_name: str = "toolbench-retrieval-proxy",
    ) -> None:
        """Initialize the retrieval proxy server.

        Args:
            tools: Full list of backend ToolInfo instances to index and route.
            retriever: ToolRetriever instance for semantic search.
            tool_caller: Optional callable (server, tool, args) to dispatch execution.
            server_name: MCP server name.
        """
        self.tools = list(tools)
        self.tool_map = {t.name: t for t in self.tools}
        self._retriever = retriever
        self.tool_caller = tool_caller
        self.server_name = server_name

        self.server = MCPServer(server_name)
        self._register_tools()

    @property
    def retriever(self) -> ToolRetriever:
        """Lazily initialize or return the ToolRetriever instance."""
        if self._retriever is None:
            self._retriever = ToolRetriever(self.tools)
        return self._retriever

    def _register_tools(self) -> None:
        """Register the meta-tools on the MCPServer instance."""

        @self.server.tool()
        def search_tools(query: str, k: int = 5) -> str:
            """Semantically search for the top-k most relevant tools for a user query.

            Args:
                query: User query or task description.
                k: Number of candidate tools to retrieve (default: 5).

            Returns:
                JSON string list of matched tools with descriptions and schemas.
            """
            k_val = max(1, k)
            matched = self.retriever.retrieve_top_k(query=query, k=k_val)
            results = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in matched
            ]
            return json.dumps(results, indent=2)

        @self.server.tool()
        async def call_tool(server: str, tool: str, arguments: dict[str, Any]) -> str:
            """Dispatch and execute a tool call on a target backend MCP server.

            Args:
                server: Target backend server identifier (e.g. 'filesystem', 'git').
                tool: Target tool name on that server.
                arguments: Dictionary of arguments conforming to the tool's input_schema.

            Returns:
                JSON string execution result or error description.
            """
            # Check if namespaced or raw tool name exists
            full_name = f"{server}__{tool}" if server else tool
            if tool not in self.tool_map and full_name not in self.tool_map:
                logger.warning("Attempted execution of unknown tool: %s on %s", tool, server)

            if self.tool_caller is None:
                # Standalone simulation response
                return json.dumps(
                    {
                        "status": "routed",
                        "server": server,
                        "tool": tool,
                        "arguments": arguments,
                        "message": "Tool call intercepted and routed by Toolbench retrieval proxy.",
                    },
                    indent=2,
                )

            try:
                res = self.tool_caller(server, tool, arguments)
                if hasattr(res, "__await__"):
                    res = await res
                return json.dumps({"status": "success", "result": res}, indent=2)
            except Exception as err:
                logger.exception("Error executing tool %s on server %s", tool, server)
                return json.dumps(
                    {"status": "error", "error": str(err), "tool": tool, "server": server},
                    indent=2,
                )

        @self.server.tool()
        def list_all_tools() -> str:
            """List all backend tools currently indexed and accessible through the proxy.

            Returns:
                JSON string summary of indexed tools.
            """
            summary = [
                {
                    "name": t.name,
                    "description": t.description[:120] + "..."
                    if len(t.description) > 120
                    else t.description,
                }
                for t in self.tools
            ]
            return json.dumps({"total_tools": len(self.tools), "tools": summary}, indent=2)

    def get_server(self) -> MCPServer:
        """Return the underlying MCPServer instance."""
        return self.server


def create_proxy_server(
    tools: list[ToolInfo],
    retriever: ToolRetriever | None = None,
    tool_caller: ToolCaller | None = None,
    server_name: str = "toolbench-retrieval-proxy",
) -> MCPServer:
    """Factory function to build and return an MCPServer instance with retrieval proxy tools."""
    proxy = RetrievalProxyServer(
        tools=tools,
        retriever=retriever,
        tool_caller=tool_caller,
        server_name=server_name,
    )
    return proxy.get_server()
