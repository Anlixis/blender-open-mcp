"""
blender-open-mcp Client
========================
A Python client for interacting with the blender-open-mcp MCP server.

Provides:
  - BlenderMCPClient: Async Python API for all MCP tools
  - CLI: interactive shell, one-shot tool calls, and `prompt` commands

Usage (CLI):
  blender-mcp-client --host http://localhost:8000 tool blender_get_scene_info
  blender-mcp-client --host http://localhost:8000 prompt "Create a metallic sphere at 0,0,2"
  blender-mcp-client --host http://localhost:8000 interactive

Usage (Python API):
  from blender_open_mcp.client.client import BlenderMCPClient
  async with BlenderMCPClient("http://localhost:8000") as client:
      print(await client.get_scene_info())
      await client.create_object("SPHERE", name="MySphere", location=(0, 0, 2))
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

import httpx

__all__ = ["BlenderMCPClient", "MCPError", "main"]


# ---------------------------------------------------------------------------
# MCP HTTP client primitives
# ---------------------------------------------------------------------------

class MCPError(Exception):
    """Raised when the MCP server returns an error response."""


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
            raise RuntimeError(
                "Client not initialized. Use 'async with BlenderMCPClient() as client:'"
            )

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
            raise MCPError(
                f"MCP error {err.get('code', '')}: {err.get('message', err)}"
            )

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
                "clientInfo": {"name": "blender-open-mcp-client", "version": "4.1.0"},
            },
        }
        try:
            await self._post(payload)
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

    async def call_tool(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> str:
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
        content = result.get("content", [])
        if content and isinstance(content, list):
            texts = [
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            ]
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

    async def get_object_info(
        self, object_name: str, response_format: str = "markdown"
    ) -> str:
        """Get detailed info about a specific Blender object."""
        return await self.call_tool(
            "blender_get_object_info",
            {
                "object_name": object_name,
                "response_format": response_format,
            },
        )

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

    async def get_selection(self) -> str:
        """Get active object, selected objects, and Blender mode."""
        return await self.call_tool("blender_get_selection")

    async def get_modifiers(self, object_name: str) -> str:
        """List modifiers on a Blender object."""
        return await self.call_tool(
            "blender_get_modifiers", {"object_name": object_name}
        )

    async def add_modifier(
        self,
        object_name: str,
        modifier_type: str,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        node_group: Optional[str] = None,
    ) -> str:
        """Add a modifier with optional properties."""
        args: Dict[str, Any] = {
            "object_name": object_name,
            "modifier_type": modifier_type,
        }
        if name:
            args["name"] = name
        if properties:
            args["properties"] = properties
        if node_group:
            args["node_group"] = node_group
        return await self.call_tool("blender_add_modifier", args)

    async def set_modifier_properties(
        self,
        object_name: str,
        modifier_name: str,
        properties: Dict[str, Any],
    ) -> str:
        """Update public RNA properties on an existing modifier."""
        return await self.call_tool(
            "blender_set_modifier_properties",
            {
                "object_name": object_name,
                "modifier_name": modifier_name,
                "properties": properties,
            },
        )

    async def gn_create_group(
        self,
        name: str,
        object_name: Optional[str] = None,
        modifier_name: Optional[str] = None,
    ) -> str:
        """Create/reuse a Geometry Nodes group and optionally attach it."""
        args: Dict[str, Any] = {"name": name}
        if object_name:
            args["object_name"] = object_name
        if modifier_name:
            args["modifier_name"] = modifier_name
        return await self.call_tool("blender_gn_create_group", args)

    async def gn_get_tree(self, node_group: str, include_sockets: bool = True) -> str:
        """Inspect a Geometry Nodes graph."""
        return await self.call_tool(
            "blender_gn_get_tree",
            {"node_group": node_group, "include_sockets": include_sockets},
        )

    async def gn_add_node(
        self,
        node_group: str,
        node_type: str,
        name: Optional[str] = None,
        label: Optional[str] = None,
        location: Optional[List[float]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a node to a Geometry Nodes group."""
        args: Dict[str, Any] = {
            "node_group": node_group,
            "node_type": node_type,
        }
        if name:
            args["name"] = name
        if label is not None:
            args["label"] = label
        if location is not None:
            args["location"] = location
        if properties:
            args["properties"] = properties
        return await self.call_tool("blender_gn_add_node", args)

    async def gn_connect(
        self,
        node_group: str,
        from_node: str,
        from_socket: Any,
        to_node: str,
        to_socket: Any,
        replace: bool = True,
    ) -> str:
        """Connect two Geometry Nodes sockets."""
        return await self.call_tool(
            "blender_gn_connect",
            {
                "node_group": node_group,
                "from_node": from_node,
                "from_socket": from_socket,
                "to_node": to_node,
                "to_socket": to_socket,
                "replace": replace,
            },
        )

    async def gn_set_input(
        self,
        node_group: str,
        node_name: str,
        input_socket: Any,
        value: Any,
    ) -> str:
        """Set a Geometry Nodes input default value."""
        return await self.call_tool(
            "blender_gn_set_input",
            {
                "node_group": node_group,
                "node_name": node_name,
                "input_socket": input_socket,
                "value": value,
            },
        )

    async def gn_set_modifier_input(
        self,
        object_name: str,
        modifier_name: str,
        input_socket: str,
        value: Any,
    ) -> str:
        """Set an exposed Geometry Nodes modifier input."""
        return await self.call_tool(
            "blender_gn_set_modifier_input",
            {
                "object_name": object_name,
                "modifier_name": modifier_name,
                "input_socket": input_socket,
                "value": value,
            },
        )

    async def gn_validate(self, node_group: str) -> str:
        """Validate a Geometry Nodes graph."""
        return await self.call_tool(
            "blender_gn_validate", {"node_group": node_group}
        )

    async def set_material(
        self,
        object_name: str,
        material_name: str,
        color: Optional[List[float]] = None,
    ) -> str:
        """Assign a material with optional RGBA color to a Blender object."""
        args: Dict[str, Any] = {
            "object_name": object_name,
            "material_name": material_name,
        }
        if color:
            args["color"] = color
        return await self.call_tool("blender_set_material", args)

    async def render_image(self, file_path: str) -> str:
        """Render the current scene and save to file_path."""
        return await self.call_tool("blender_render_image", {"file_path": file_path})

    async def execute_code(self, code: str) -> str:
        """Execute Python code in Blender's bpy environment."""
        return await self.call_tool("blender_execute_code", {"code": code})

    async def ai_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """
        Send a prompt to the configured LLM backend.

        provider: leave None for the server default, or pass one of the
        supported backend names (openai, ollama, lmstudio, llamacpp, azure).
        """
        args: Dict[str, Any] = {"prompt": prompt}
        if system_prompt:
            args["system_prompt"] = system_prompt
        if model:
            args["model"] = model
        if provider:
            args["provider"] = provider
        if base_url:
            args["base_url"] = base_url
        if api_key:
            args["api_key"] = api_key
        return await self.call_tool("blender_ai_prompt", args)

    async def get_llm_provider(self) -> str:
        """Return the current LLM provider configuration from the server."""
        return await self.call_tool("blender_get_llm_provider")

    async def set_llm_provider(
        self,
        provider: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Change the active LLM provider at runtime."""
        args: Dict[str, Any] = {"provider": provider}
        if base_url:
            args["base_url"] = base_url
        if api_key:
            args["api_key"] = api_key
        if model:
            args["model"] = model
        if extra:
            args["extra"] = extra
        return await self.call_tool("blender_set_llm_provider", args)

    async def list_llm_models(self, provider: Optional[str] = None) -> str:
        """List models available from a provider (defaults to the active one)."""
        args = {"provider": provider} if provider else {}
        return await self.call_tool("blender_list_llm_models", args)

    async def get_ollama_models(self) -> str:
        """List models available from the current Ollama endpoint."""
        return await self.call_tool("blender_get_ollama_models")

    async def search_polyhaven_assets(
        self,
        asset_type: str = "textures",
        categories: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> str:
        """Search the PolyHaven asset library."""
        args: Dict[str, Any] = {
            "asset_type": asset_type,
            "limit": limit,
            "offset": offset,
        }
        if categories:
            args["categories"] = categories
        return await self.call_tool("blender_search_polyhaven_assets", args)

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    async def _cmd_tool(self, name: str, args_json: Optional[str]) -> None:
        args = {}
        if args_json:
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON arguments: {exc}")
        print(await self.call_tool(name, args if args else None))

    async def _cmd_prompt(self, text: str) -> None:
        print(await self.ai_prompt(text))

    async def _cmd_interactive(self) -> None:
        """Simple interactive REPL."""
        print("blender-open-mcp client. Commands:")
        print("  tools                     List available tools")
        print("  tool <name> [json args]   Call a tool")
        print("  prompt <text>             Ask the configured LLM provider")
        print("  provider                  Show active LLM provider")
        print("  quit / exit               Leave")
        while True:
            try:
                line = (await asyncio.to_thread(input, "blender-mcp> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("quit", "exit"):
                break
            if line == "tools":
                for t in await self.list_tools():
                    print(f"- {t.get('name')}: {t.get('description', '')}")
            elif line == "provider":
                print(await self.get_llm_provider())
            elif line.startswith("tool "):
                parts = line[5:].strip().split(None, 1)
                if not parts:
                    print("Usage: tool <name> [json args]")
                    continue
                await self._cmd_tool(parts[0], parts[1] if len(parts) > 1 else None)
            elif line.startswith("prompt "):
                await self._cmd_prompt(line[7:].strip())
            else:
                # Bare text is treated as a prompt.
                await self._cmd_prompt(line)

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="blender-mcp-client",
            description="blender-open-mcp client CLI",
        )
        parser.add_argument("--host", default="http://localhost:8000",
                            help="MCP server base URL")
        sub = parser.add_subparsers(dest="command", required=False)
        p_tools = sub.add_parser("tools", help="list available tools")
        p_tool = sub.add_parser("tool", help="call a tool by name")
        p_tool.add_argument("name", help="tool name")
        p_tool.add_argument("json_args", nargs="?", default=None,
                            help="JSON arguments for the tool call")
        p_prompt = sub.add_parser("prompt", help="ask the configured LLM provider")
        p_prompt.add_argument("text", help="prompt text")
        sub.add_parser("interactive", help="open an interactive shell")
        return parser

    async def dispatch(self, args: argparse.Namespace) -> None:
        if args.command is None or args.command == "interactive":
            await self._cmd_interactive()
        elif args.command == "tools":
            for t in await self.list_tools():
                print(f"- {t.get('name')}: {t.get('description', '')}")
        elif args.command == "tool":
            await self._cmd_tool(args.name, args.json_args)
        elif args.command == "prompt":
            await self._cmd_prompt(args.text)


async def _main_async() -> None:
    parser = BlenderMCPClient._build_parser()
    args = parser.parse_args()
    async with BlenderMCPClient(args.host) as client:
        await client.dispatch(args)


def main() -> None:
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
    except MCPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


def main() -> None:
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
    except MCPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
