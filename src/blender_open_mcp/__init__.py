"""Blender integration through the Model Context Protocol."""

__version__ = "2.0.0"

# The primary public interface is the FastMCP server.
# Import from blender_open_mcp.server directly:
#   from blender_open_mcp.server import mcp, main
# For the client, use the separate client module:
#   from client.client import BlenderMCPClient