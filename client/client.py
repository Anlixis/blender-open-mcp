"""
Compatibility wrapper for the blender-open-mcp client.

The canonical implementation lives in ``src/blender_open_mcp/client/`` and is
what the installed ``blender-mcp-client`` console script runs. This module lets
older imports (``from client import BlenderMCPClient``) keep working when the
repo is used from source.
"""

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from blender_open_mcp.client.client import (  # noqa: E402
    BlenderMCPClient,
    MCPError,
    main,
)

__all__ = ["BlenderMCPClient", "MCPError", "main"]

if __name__ == "__main__":
    main()
