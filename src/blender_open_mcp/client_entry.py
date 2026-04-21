"""Entry point for the blender-mcp-client CLI."""

import sys
import os

# Add the client directory to the path
client_dir = os.path.join(os.path.dirname(__file__), "..", "..", "client")
sys.path.insert(0, os.path.abspath(client_dir))

from client import main

if __name__ == "__main__":
    main()