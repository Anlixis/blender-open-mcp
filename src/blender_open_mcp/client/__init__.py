"""
blender_open_mcp.client - async MCP client + CLI for blender-open-mcp.
"""

from __future__ import annotations

from .client import BlenderMCPClient, MCPError, main

__all__ = ["BlenderMCPClient", "MCPError", "main"]
