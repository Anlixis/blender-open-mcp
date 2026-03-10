"""
blender-open-mcp Client
========================
A Python client for interacting with the blender-open-mcp MCP server.

Provides:
  - BlenderMCPClient: Async Python API for all MCP tools
  - CLI: Interactive shell and one-shot command execution

Usage (CLI):
  python client/client.py --host http://localhost:8000 interactive
  python client/client.py --host http://localhost:8000 tool blender_get_scene_info
  python client/client.py --host http://localhost:8000 prompt "Create a metallic sphere at 0,0,2"

Usage (Python API):
  from client.client import BlenderMCPClient
  async with BlenderMCPClient("http://localhost:8000") as client:
      print(await client.get_scene_info())
      await client.create_object("SPHERE", name="MySphere", location=(0, 0, 2))
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

import httpx


# ---------------------------------------------------------------------------
# MCP HTTP client primitives
# ---------------------------------------------------------------------------

class MCPError(Exception):
    """Raised when the MCP server returns an error response."""
    pass


class BlenderMCPClient:
    """
    Async client for the blender-open-mcp MCP server.

    Communicates with the FastMCP HTTP transport (streamable HTTP).
    Each method corresponds to a tool registered on the server.

    Example:
        async with BlenderMCPClient("http://localhost:8000") as client:
            scene = await client.get_scene_info()
            print(scene)
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._http: Optional[httpx.AsyncClient] = None
        self._timeout = timeout
        self._session_id: Optional[str] = None

    async def __aenter__(self) -> "BlenderMCPClient":
        self._http = httpx.AsyncClient(timeout=self._timeout)
        await self._initialize()
        return self

    async def __aexit__(self, *_args) -> None:
        if self._http:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Low-level MCP JSON-RPC over HTTP
    # ------------------------------------------------------------------

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request to the MCP endpoint and return the result."""
        if self._http is None:
            raise RuntimeError("Client not initialized. Use 'async with BlenderMCPClient() as client:'")

        headers = {"Content-Type": "application/json"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        try:
            resp = await self._http.post(
                f"{self.base_url}/mcp",
                json=payload,
                headers=headers,
            )
        except httpx.ConnectError:
            raise MCPError(
                f"Cannot connect to MCP server at {self.base_url}. "
                "Run: blender-mcp --host 0.0.0.0 --port 8000"
            )

        # Capture session ID if issued
        if "Mcp-Session-Id" in resp.headers:
            self._session_id = resp.headers["Mcp-Session-Id"]

        if resp.status_code not in (200, 202):
            raise MCPError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        if resp.status_code == 202:
            return {"result": "(accepted, no content)"}

        # Handle newline-delimited JSON (streamable HTTP)
        content = resp.text.strip()
        last_response: Dict[str, Any] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("event:") or line.startswith("id:"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            try:
                last_response = json.loads(line)
            except json.JSONDecodeError:
                continue

        if "error" in last_response:
            err = last_response["error"]
            raise MCPError(f"MCP error {err.get('code', '')}: {err.get('message', err)}")

        return last_response.get("result", last_response)

    async def _initialize(self) -> None:
        """Perform MCP protocol initialization handshake."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "blender-open-mcp-client", "version": "2.0.0"},
            },
        }
        try:
            result = await self._post(payload)
            # Send initialized notification
            await self._notify("notifications/initialized")
        except MCPError:
            # Some transports don't require initialize; continue
            pass

    async def _notify(self, method: str, params: Optional[Dict] = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        payload = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        try:
            headers = {"Content-Type": "application/json"}
            if self._session_id:
                headers["Mcp-Session-Id"] = self._session_id
            await self._http.post(f"{self.base_url}/mcp", json=payload, headers=headers)
        except Exception:
            pass

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """
        Call any MCP tool by name with the given arguments.

        Returns the text content of the first content block in the response.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }
        result = await self._post(payload)
        # Extract text from content array
        content = result.get("content", [])
        if content and isinstance(content, list):
            texts = [block.get("text", "") for block in content if block.get("type") == "text"]
            return "\n".join(texts)
        return str(result)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all tools available on the MCP server."""
        payload = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
        result = await self._post(payload)
        return result.get("tools", [])

    # ------------------------------------------------------------------
    # Typed convenience methods
    # ------------------------------------------------------------------

    async def get_scene_info(self) -> str:
        """Get a full summary of the current Blender scene."""
        return await self.call_tool("blender_get_scene_info")

    async def get_object_info(self, object_name: str, response_format: str = "markdown") -> str:
        """Get detailed info about a specific Blender object."""
        return await self.call_tool("blender_get_object_info", {
            "object_name": object_name,
            "response_format": response_format,
        })

    async def create_object(
        self,
        primitive_type: str = "CUBE",
        name: Optional[str] = None,
        location: Optional[tuple] = None,
        rotation: Optional[tuple] = None,
        scale: Optional[tuple] = None,
    ) -> str:
        """Create a primitive mesh object in Blender."""
        args: Dict[str, Any] = {"primitive_type": primitive_type.upper()}
        if name:
            args["name"] = name
        if location:
            args["location"] = {"x": location[0], "y": location[1], "z": location[2]}
        if rotation:
            args["rotation"] = {"x": rotation[0], "y": rotation[1], "z": rotation[2]}
        if scale:
            args["scale"] = {"x": scale[0], "y": scale[1], "z": scale[2]}
        return await self.call_tool("blender_create_object", args)

    async def modify_object(
        self,
        name: str,
        location: Optional[tuple] = None,
        rotation: Optional[tuple] = None,
        scale: Optional[tuple] = None,
        visible: Optional[bool] = None,
    ) -> str:
        """Modify an existing object's transform or visibility."""
        args: Dict[str, Any] = {"name": name}
        if location:
            args["location"] = {"x": location[0], "y": location[1], "z": location[2]}
        if rotation:
            args["rotation"] = {"x": rotation[0], "y": rotation[1], "z": rotation[2]}
        if scale:
            args["scale"] = {"x": scale[0], "y": scale[1], "z": scale[2]}
        if visible is not None:
            args["visible"] = visible
        return await self.call_tool("blender_modify_object", args)

    async def delete_object(self, name: str) -> str:
        """Delete an object from the Blender scene."""
        return await self.call_tool("blender_delete_object", {"name": name})

    async def set_material(
        self,
        object_name: str,
        material_name: str,
        color: Optional[List[float]] = None,
    ) -> str:
        """Assign a material with optional RGBA color to a Blender object."""
        args: Dict[str, Any] = {"object_name": object_name, "material_name": material_name}
        if color:
            args["color"] = color
        return await self.call_tool("blender_set_material", args)

    async def render_image(self, file_path: str) -> str:
        """Render the current scene and save to file_path."""
        return await self.call_tool("blender_render_image", {"file_path": file_path})

    async def execute_code(self, code: str) -> str:
        """Execute Python (bpy) code inside Blender."""
        return await self.call_tool("blender_execute_code", {"code": code})

    async def ai_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a natural language prompt to Ollama via the MCP server."""
        args: Dict[str, Any] = {"prompt": prompt}
        if system_prompt:
            args["system_prompt"] = system_prompt
        return await self.call_tool("blender_ai_prompt", args)

    async def get_polyhaven_categories(self, asset_type: str = "textures") -> str:
        """List PolyHaven asset categories."""
        return await self.call_tool("blender_get_polyhaven_categories", {"asset_type": asset_type})

    async def search_polyhaven_assets(
        self,
        asset_type: str = "textures",
        categories: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> str:
        """Search PolyHaven assets."""
        args: Dict[str, Any] = {"asset_type": asset_type, "limit": limit, "offset": offset}
        if categories:
            args["categories"] = categories
        return await self.call_tool("blender_search_polyhaven_assets", args)

    async def download_polyhaven_asset(
        self,
        asset_id: str,
        asset_type: str = "textures",
        resolution: str = "1k",
        file_format: str = "jpg",
    ) -> str:
        """Download and import a PolyHaven asset into Blender."""
        return await self.call_tool("blender_download_polyhaven_asset", {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "resolution": resolution,
            "file_format": file_format,
        })

    async def set_texture(self, object_name: str, texture_id: str) -> str:
        """Apply a downloaded PolyHaven texture to a Blender object."""
        return await self.call_tool("blender_set_texture", {
            "object_name": object_name,
            "texture_id": texture_id,
        })

    async def get_ollama_models(self) -> str:
        """List available Ollama models."""
        return await self.call_tool("blender_get_ollama_models")

    async def set_ollama_model(self, model_name: str) -> str:
        """Switch the Ollama model."""
        return await self.call_tool("blender_set_ollama_model", {"model_name": model_name})

    async def set_ollama_url(self, url: str) -> str:
        """Change the Ollama server URL."""
        return await self.call_tool("blender_set_ollama_url", {"url": url})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

HELP_TEXT = """
blender-open-mcp Client
========================
Commands:
  scene             - Get current scene info
  object <name>     - Get info about an object
  create <type>     - Create a primitive (CUBE, SPHERE, CYLINDER, ...)
  delete <name>     - Delete an object
  material <obj> <mat> [r g b] - Set material with optional color (0.0-1.0)
  render <path>     - Render to file path
  exec <code>       - Execute Python code in Blender
  prompt <text>     - Send AI prompt to Ollama
  models            - List available Ollama models
  model <name>      - Switch Ollama model
  polyhaven cats    - List PolyHaven categories
  polyhaven search  - Search PolyHaven assets
  tools             - List all MCP tools
  help              - Show this help
  quit / exit       - Exit
"""


async def run_interactive(client: BlenderMCPClient) -> None:
    """Run an interactive REPL session."""
    print("\n🎨 blender-open-mcp Client — Interactive Mode")
    print(f"   Connected to: {client.base_url}")
    print("   Type 'help' for available commands.\n")

    while True:
        try:
            line = input("blender-mcp> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not line:
            continue

        parts = line.split(None, 3)
        cmd = parts[0].lower()

        try:
            if cmd in ("quit", "exit", "q"):
                print("Bye!")
                break
            elif cmd == "help":
                print(HELP_TEXT)
            elif cmd == "scene":
                print(await client.get_scene_info())
            elif cmd == "object" and len(parts) >= 2:
                print(await client.get_object_info(parts[1]))
            elif cmd == "create" and len(parts) >= 2:
                print(await client.create_object(primitive_type=parts[1].upper()))
            elif cmd == "delete" and len(parts) >= 2:
                print(await client.delete_object(parts[1]))
            elif cmd == "material" and len(parts) >= 3:
                color = None
                if len(parts) >= 4:
                    rgb = parts[3].split()
                    if len(rgb) >= 3:
                        color = [float(c) for c in rgb[:3]] + [1.0]
                print(await client.set_material(parts[1], parts[2], color))
            elif cmd == "render" and len(parts) >= 2:
                print(await client.render_image(parts[1]))
            elif cmd == "exec" and len(parts) >= 2:
                print(await client.execute_code(parts[1]))
            elif cmd == "prompt" and len(parts) >= 2:
                print(await client.ai_prompt(" ".join(parts[1:])))
            elif cmd == "models":
                print(await client.get_ollama_models())
            elif cmd == "model" and len(parts) >= 2:
                print(await client.set_ollama_model(parts[1]))
            elif cmd == "polyhaven":
                sub = parts[1].lower() if len(parts) >= 2 else "cats"
                if sub == "cats":
                    print(await client.get_polyhaven_categories())
                elif sub == "search":
                    print(await client.search_polyhaven_assets())
                else:
                    print(f"Unknown polyhaven sub-command: {sub}")
            elif cmd == "tools":
                tools = await client.list_tools()
                for t in tools:
                    desc = t.get("description", "")[:80]
                    print(f"  {t['name']:<40} {desc}")
            else:
                print(f"Unknown command: '{line}'. Type 'help'.")
        except MCPError as exc:
            print(f"[MCP Error] {exc}")
        except Exception as exc:
            print(f"[Error] {type(exc).__name__}: {exc}")


async def run_one_shot(client: BlenderMCPClient, mode: str, args: List[str]) -> None:
    """Execute a single command and print the result."""
    if mode == "tool":
        tool_name = args[0]
        tool_args: Dict[str, Any] = {}
        if len(args) > 1:
            try:
                tool_args = json.loads(args[1])
            except json.JSONDecodeError:
                print(f"Warning: could not parse tool args as JSON: {args[1]}", file=sys.stderr)
        print(await client.call_tool(tool_name, tool_args))

    elif mode == "prompt":
        prompt_text = " ".join(args)
        print(await client.ai_prompt(prompt_text))

    elif mode == "scene":
        print(await client.get_scene_info())

    elif mode == "tools":
        tools = await client.list_tools()
        for t in tools:
            print(f"  {t['name']}")

    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="blender-mcp-client",
        description="blender-open-mcp client — interact with your Blender MCP server",
    )
    parser.add_argument("--host", default="http://localhost:8000", help="MCP server URL (default: http://localhost:8000)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds")

    sub = parser.add_subparsers(dest="mode")
    sub.add_parser("interactive", aliases=["i", "shell"], help="Start interactive REPL")
    sub.add_parser("scene", help="Get current scene info")
    sub.add_parser("tools", help="List available MCP tools")

    tool_p = sub.add_parser("tool", help="Call a specific tool")
    tool_p.add_argument("tool_name", help="Tool name (e.g., blender_get_scene_info)")
    tool_p.add_argument("tool_args", nargs="?", help="JSON-encoded tool arguments")

    prompt_p = sub.add_parser("prompt", help="Send natural language prompt to Ollama")
    prompt_p.add_argument("text", nargs="+", help="Prompt text")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(0)

    async def _run():
        async with BlenderMCPClient(args.host, timeout=args.timeout) as client:
            if args.mode in ("interactive", "i", "shell"):
                await run_interactive(client)
            elif args.mode == "tool":
                tool_args_list = [args.tool_name]
                if hasattr(args, "tool_args") and args.tool_args:
                    tool_args_list.append(args.tool_args)
                await run_one_shot(client, "tool", tool_args_list)
            elif args.mode == "prompt":
                await run_one_shot(client, "prompt", args.text)
            else:
                await run_one_shot(client, args.mode, [])

    asyncio.run(_run())


if __name__ == "__main__":
    main()
