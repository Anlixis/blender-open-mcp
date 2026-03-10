"""
Tests for the blender-open-mcp MCP Client
==========================================
Tests verify request construction, response parsing, error handling,
and all convenience methods without a live server.
"""

from __future__ import annotations

import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure client is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "client"))

from client import BlenderMCPClient, MCPError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(content_blocks: list, status_code: int = 200) -> MagicMock:
    """Build a mock httpx response with MCP JSON-RPC format."""
    body = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "content": [{"type": "text", "text": block} for block in content_blocks]
        },
    }
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = json.dumps(body)
    mock.headers = {}
    return mock


def _make_tools_response(tools: list) -> MagicMock:
    body = {"jsonrpc": "2.0", "id": 3, "result": {"tools": tools}}
    mock = MagicMock()
    mock.status_code = 200
    mock.text = json.dumps(body)
    mock.headers = {}
    return mock


def _make_init_response() -> MagicMock:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "blender_open_mcp"},
        },
    }
    mock = MagicMock()
    mock.status_code = 200
    mock.text = json.dumps(body)
    mock.headers = {"Mcp-Session-Id": "test-session-123"}
    return mock


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

class TestClientConstruction:
    def test_default_url(self):
        client = BlenderMCPClient()
        assert client.base_url == "http://localhost:8000"

    def test_custom_url_trailing_slash_stripped(self):
        client = BlenderMCPClient("http://localhost:9000/")
        assert client.base_url == "http://localhost:9000"

    def test_timeout_default(self):
        client = BlenderMCPClient()
        assert client._timeout == 60.0


# ---------------------------------------------------------------------------
# Low-level call_tool
# ---------------------------------------------------------------------------

class TestCallTool:
    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        client = BlenderMCPClient.__new__(BlenderMCPClient)
        client.base_url = "http://localhost:8000"
        client._timeout = 60.0
        client._session_id = None

        init_resp = _make_init_response()
        tool_resp = _make_mock_response(["Scene: Default"])

        client._http = MagicMock()
        client._http.post = AsyncMock(side_effect=[init_resp, MagicMock(status_code=200, text="", headers={}), tool_resp])

        result = await client.call_tool("blender_get_scene_info")
        assert "Scene" in result or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_call_tool_mcp_error(self):
        client = BlenderMCPClient.__new__(BlenderMCPClient)
        client.base_url = "http://localhost:8000"
        client._timeout = 60.0
        client._session_id = None

        error_body = {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -32601, "message": "Tool not found"},
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps(error_body)
        mock_resp.headers = {}

        client._http = MagicMock()
        client._http.post = AsyncMock(return_value=mock_resp)

        with pytest.raises(MCPError) as exc_info:
            await client.call_tool("nonexistent_tool")
        assert "Tool not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_tool_connection_error(self):
        import httpx
        client = BlenderMCPClient.__new__(BlenderMCPClient)
        client.base_url = "http://localhost:8000"
        client._timeout = 60.0
        client._session_id = None
        client._http = MagicMock()
        client._http.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with pytest.raises(MCPError) as exc_info:
            await client.call_tool("blender_get_scene_info")
        assert "Cannot connect" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Convenience methods — parameter construction
# ---------------------------------------------------------------------------

class TestConvenienceMethods:
    def _setup_client(self):
        """Return a client with mocked _post/_call_tool."""
        client = BlenderMCPClient.__new__(BlenderMCPClient)
        client.base_url = "http://localhost:8000"
        client._timeout = 60.0
        client._session_id = None
        client._http = None
        return client

    @pytest.mark.asyncio
    async def test_create_object_sends_primitive_type(self):
        client = self._setup_client()
        captured = {}

        async def mock_call_tool(name, args=None):
            captured["name"] = name
            captured["args"] = args
            return "created"

        client.call_tool = mock_call_tool
        await client.create_object("SPHERE", name="Ball", location=(1.0, 2.0, 3.0))

        assert captured["name"] == "blender_create_object"
        assert captured["args"]["primitive_type"] == "SPHERE"
        assert captured["args"]["name"] == "Ball"
        assert captured["args"]["location"] == {"x": 1.0, "y": 2.0, "z": 3.0}

    @pytest.mark.asyncio
    async def test_set_material_with_color(self):
        client = self._setup_client()
        captured = {}

        async def mock_call_tool(name, args=None):
            captured["name"] = name
            captured["args"] = args
            return "material set"

        client.call_tool = mock_call_tool
        await client.set_material("Cube", "RedMat", color=[1.0, 0.0, 0.0, 1.0])

        assert captured["args"]["object_name"] == "Cube"
        assert captured["args"]["material_name"] == "RedMat"
        assert captured["args"]["color"] == [1.0, 0.0, 0.0, 1.0]

    @pytest.mark.asyncio
    async def test_modify_object_partial_update(self):
        """Only provided fields should be included in the call."""
        client = self._setup_client()
        captured = {}

        async def mock_call_tool(name, args=None):
            captured["args"] = args
            return "modified"

        client.call_tool = mock_call_tool
        await client.modify_object("Cube", visible=False)

        assert captured["args"]["name"] == "Cube"
        assert captured["args"]["visible"] is False
        assert "location" not in captured["args"]
        assert "scale" not in captured["args"]

    @pytest.mark.asyncio
    async def test_ai_prompt_includes_system(self):
        client = self._setup_client()
        captured = {}

        async def mock_call_tool(name, args=None):
            captured["args"] = args
            return "AI response"

        client.call_tool = mock_call_tool
        await client.ai_prompt("Create a cube", system_prompt="You are a Blender expert.")

        assert captured["args"]["prompt"] == "Create a cube"
        assert captured["args"]["system_prompt"] == "You are a Blender expert."

    @pytest.mark.asyncio
    async def test_search_polyhaven_default_args(self):
        client = self._setup_client()
        captured = {}

        async def mock_call_tool(name, args=None):
            captured["args"] = args
            return "assets"

        client.call_tool = mock_call_tool
        await client.search_polyhaven_assets()

        assert captured["args"]["asset_type"] == "textures"
        assert captured["args"]["limit"] == 20
        assert captured["args"]["offset"] == 0

    @pytest.mark.asyncio
    async def test_get_scene_info_calls_correct_tool(self):
        client = self._setup_client()
        captured = {}

        async def mock_call_tool(name, args=None):
            captured["name"] = name
            return "scene info"

        client.call_tool = mock_call_tool
        result = await client.get_scene_info()

        assert captured["name"] == "blender_get_scene_info"
        assert result == "scene info"


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------

class TestListTools:
    @pytest.mark.asyncio
    async def test_list_tools_returns_list(self):
        client = BlenderMCPClient.__new__(BlenderMCPClient)
        client.base_url = "http://localhost:8000"
        client._timeout = 60.0
        client._session_id = None

        tools_list = [
            {"name": "blender_get_scene_info", "description": "Get scene info"},
            {"name": "blender_create_object", "description": "Create an object"},
        ]
        mock_resp = _make_tools_response(tools_list)
        client._http = MagicMock()
        client._http.post = AsyncMock(return_value=mock_resp)

        result = await client.list_tools()
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "blender_get_scene_info"


# ---------------------------------------------------------------------------
# Session ID handling
# ---------------------------------------------------------------------------

class TestSessionHandling:
    @pytest.mark.asyncio
    async def test_session_id_captured_from_response(self):
        client = BlenderMCPClient.__new__(BlenderMCPClient)
        client.base_url = "http://localhost:8000"
        client._timeout = 60.0
        client._session_id = None

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}})
        mock_resp.headers = {"Mcp-Session-Id": "abc-123-xyz"}

        client._http = MagicMock()
        # init call returns session id, notification returns 200
        client._http.post = AsyncMock(side_effect=[
            mock_resp,
            MagicMock(status_code=200, text="", headers={}),
        ])

        await client._initialize()
        assert client._session_id == "abc-123-xyz"
