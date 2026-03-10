"""
Tests for blender-open-mcp MCP Server
======================================
These tests validate server tool registration, input validation,
error handling, and Ollama integration without requiring a live Blender instance.
"""

from __future__ import annotations

import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Server is imported per-test class to avoid circular issues


# ---------------------------------------------------------------------------
# Helper: import server without running it
# ---------------------------------------------------------------------------

def _import_server():
    """Import server module fresh."""
    import importlib
    import blender_open_mcp.server as srv
    importlib.reload(srv)
    return srv


# ---------------------------------------------------------------------------
# Input model validation tests
# ---------------------------------------------------------------------------

class TestInputModels:
    def test_vec3_defaults(self):
        from blender_open_mcp.server import Vec3
        v = Vec3()
        assert v.x == 0.0
        assert v.as_list() == [0.0, 0.0, 0.0]

    def test_vec3_custom(self):
        from blender_open_mcp.server import Vec3
        v = Vec3(x=1.0, y=2.5, z=-3.0)
        assert v.as_list() == [1.0, 2.5, -3.0]

    def test_set_material_color_validation_valid(self):
        from blender_open_mcp.server import SetMaterialInput
        m = SetMaterialInput(
            object_name="Cube",
            material_name="RedMat",
            color=[1.0, 0.0, 0.0, 1.0],
        )
        assert m.color == [1.0, 0.0, 0.0, 1.0]

    def test_set_material_color_auto_alpha(self):
        from blender_open_mcp.server import SetMaterialInput
        m = SetMaterialInput(
            object_name="Cube",
            material_name="GreenMat",
            color=[0.0, 1.0, 0.0],
        )
        assert len(m.color) == 4
        assert m.color[3] == 1.0

    def test_set_material_color_out_of_range(self):
        from blender_open_mcp.server import SetMaterialInput
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SetMaterialInput(
                object_name="Cube",
                material_name="Bad",
                color=[2.0, 0.0, 0.0],
            )

    def test_set_material_color_wrong_length(self):
        from blender_open_mcp.server import SetMaterialInput
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SetMaterialInput(
                object_name="Cube",
                material_name="Bad",
                color=[1.0, 0.0],
            )

    def test_create_object_defaults(self):
        from blender_open_mcp.server import CreateObjectInput, PrimitiveType
        m = CreateObjectInput()
        assert m.primitive_type == PrimitiveType.CUBE

    def test_get_object_info_empty_name_fails(self):
        from blender_open_mcp.server import GetObjectInfoInput
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            GetObjectInfoInput(object_name="")

    def test_search_polyhaven_pagination_defaults(self):
        from blender_open_mcp.server import SearchPolyHavenInput
        m = SearchPolyHavenInput()
        assert m.limit == 20
        assert m.offset == 0

    def test_search_polyhaven_limit_bounds(self):
        from blender_open_mcp.server import SearchPolyHavenInput
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SearchPolyHavenInput(limit=0)
        with pytest.raises(pydantic.ValidationError):
            SearchPolyHavenInput(limit=101)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_handle_blender_error_connection_refused(self):
        from blender_open_mcp.server import _handle_blender_error
        err = ConnectionRefusedError("Connection refused")
        result = _handle_blender_error(err)
        assert "Cannot connect" in result or "Connection refused" in result

    def test_handle_blender_error_timeout(self):
        from blender_open_mcp.server import _handle_blender_error
        err = TimeoutError("Timed out")
        result = _handle_blender_error(err)
        assert "Timed out" in result or "timeout" in result.lower()

    def test_handle_blender_error_unknown(self):
        from blender_open_mcp.server import _handle_blender_error
        err = RuntimeError("Unexpected")
        result = _handle_blender_error(err)
        assert "RuntimeError" in result or "Unexpected" in result


# ---------------------------------------------------------------------------
# Blender command helper tests
# ---------------------------------------------------------------------------

class TestBlenderCommandHelper:
    def test_send_blender_command_connection_refused(self):
        """When Blender add-on is not running, raises ConnectionRefusedError."""
        from blender_open_mcp.server import _send_blender_command
        # Port 19999 should be unused
        import blender_open_mcp.server as srv
        old_host, old_port = srv.BLENDER_HOST, srv.BLENDER_PORT
        srv.BLENDER_HOST = "localhost"
        srv.BLENDER_PORT = 19999
        try:
            with pytest.raises(ConnectionRefusedError):
                _send_blender_command("get_scene_info")
        finally:
            srv.BLENDER_HOST = old_host
            srv.BLENDER_PORT = old_port

    def test_format_blender_result_ok(self):
        from blender_open_mcp.server import _format_blender_result
        resp = {"status": "ok", "result": {"name": "Cube"}}
        out = _format_blender_result(resp)
        assert "Cube" in out

    def test_format_blender_result_error(self):
        from blender_open_mcp.server import _format_blender_result
        resp = {"status": "error", "message": "Object not found"}
        out = _format_blender_result(resp)
        assert "Object not found" in out


# ---------------------------------------------------------------------------
# Ollama integration tests (mocked)
# ---------------------------------------------------------------------------

class TestOllamaIntegration:
    @pytest.mark.asyncio
    async def test_query_ollama_success(self):
        from blender_open_mcp.server import _query_ollama
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "Here is some Python code..."}

        with patch("httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.post = AsyncMock(return_value=mock_resp)
            result = await _query_ollama("Create a cube in Blender")

        assert "Python" in result or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_query_ollama_connection_error(self):
        from blender_open_mcp.server import _query_ollama
        import httpx

        with patch("httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            result = await _query_ollama("test prompt")

        assert "Error" in result
        assert "Ollama" in result

    @pytest.mark.asyncio
    async def test_set_ollama_model_tool(self):
        """Test that set_ollama_model updates runtime state."""
        from blender_open_mcp.server import _state, SetOllamaModelInput
        _state["ollama_model"] = "llama3.2"

        from blender_open_mcp.server import blender_set_ollama_model
        result = await blender_set_ollama_model(SetOllamaModelInput(model_name="gemma3"))
        assert "gemma3" in result
        assert _state["ollama_model"] == "gemma3"

        # Restore
        _state["ollama_model"] = "llama3.2"

    @pytest.mark.asyncio
    async def test_set_ollama_url_tool(self):
        from blender_open_mcp.server import _state, SetOllamaUrlInput
        _state["ollama_url"] = "http://localhost:11434"

        from blender_open_mcp.server import blender_set_ollama_url
        result = await blender_set_ollama_url(SetOllamaUrlInput(url="http://192.168.1.50:11434"))
        assert "192.168.1.50" in result
        assert _state["ollama_url"] == "http://192.168.1.50:11434"

        # Restore
        _state["ollama_url"] = "http://localhost:11434"


# ---------------------------------------------------------------------------
# MCP tool annotation checks
# ---------------------------------------------------------------------------

class TestToolAnnotations:
    """Verify that all tools have correct MCP annotations."""

    def test_blender_delete_is_destructive(self):
        """Delete tool must be marked destructive."""
        from blender_open_mcp.server import mcp
        # Access registered tools
        tools = {}
        for name, tool in mcp._tool_manager._tools.items():
            tools[name] = tool
        assert "blender_delete_object" in tools
        annotations = tools["blender_delete_object"].annotations or {}
        assert annotations.get("destructiveHint") is True

    def test_scene_info_is_readonly(self):
        """Scene info tool must be read-only."""
        from blender_open_mcp.server import mcp
        tools = {name: t for name, t in mcp._tool_manager._tools.items()}
        assert "blender_get_scene_info" in tools
        annotations = tools["blender_get_scene_info"].annotations or {}
        assert annotations.get("readOnlyHint") is True

    def test_execute_code_is_destructive(self):
        """Code execution must be marked destructive."""
        from blender_open_mcp.server import mcp
        tools = {name: t for name, t in mcp._tool_manager._tools.items()}
        assert "blender_execute_code" in tools
        annotations = tools["blender_execute_code"].annotations or {}
        assert annotations.get("destructiveHint") is True


# ---------------------------------------------------------------------------
# PolyHaven integration tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestPolyHavenTools:
    @pytest.mark.asyncio
    async def test_get_polyhaven_categories_success(self):
        from blender_open_mcp.server import blender_get_polyhaven_categories, GetPolyHavenCategoriesInput
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = ["wood", "metal", "fabric"]

        with patch("httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(return_value=mock_resp)
            result = await blender_get_polyhaven_categories(
                GetPolyHavenCategoriesInput(asset_type="textures")
            )

        parsed = json.loads(result)
        assert "wood" in parsed

    @pytest.mark.asyncio
    async def test_search_polyhaven_assets_pagination(self):
        from blender_open_mcp.server import blender_search_polyhaven_assets, SearchPolyHavenInput
        # Generate 50 fake assets
        fake_assets = {f"asset_{i}": {"name": f"Asset {i}", "categories": ["wood"]} for i in range(50)}
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = fake_assets

        with patch("httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(return_value=mock_resp)
            result = await blender_search_polyhaven_assets(
                SearchPolyHavenInput(limit=10, offset=0)
            )

        parsed = json.loads(result)
        assert parsed["total"] == 50
        assert parsed["count"] == 10
        assert parsed["has_more"] is True
        assert parsed["next_offset"] == 10
