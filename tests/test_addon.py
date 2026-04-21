"""
Tests for the Blender MCP add-on.
=================================
Note: The addon.py architecture changed in v2.0.0 from a BlenderMCPServer class
to a module-level TCP server. These tests have been updated accordingly.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Mock bpy and its submodules before importing the addon module
bpy_mock = MagicMock()
bpy_mock.props = MagicMock()
sys.modules['bpy'] = bpy_mock
sys.modules['bpy.props'] = bpy_mock.props

# Add the root directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import the addon
import addon


class TestAddonHandlers(unittest.TestCase):
    """Test the command handler functions."""

    def setUp(self):
        """Set up mock bpy context for tests."""
        # Mock scene
        self.mock_scene = MagicMock()
        self.mock_scene.name = "Scene"
        self.mock_scene.frame_current = 1
        self.mock_scene.frame_start = 1
        self.mock_scene.frame_end = 250
        self.mock_scene.render.engine = "CYCLES"
        self.mock_scene.render.resolution_x = 1920
        self.mock_scene.render.resolution_y = 1080
        self.mock_scene.camera = MagicMock()
        self.mock_scene.camera.name = "Camera"
        self.mock_scene.objects = []

        addon.bpy.context.scene = self.mock_scene
        addon.bpy.data.objects = {}

    def test_handle_get_scene_info_empty_scene(self):
        """Test scene info handler with empty scene."""
        result = addon.handle_get_scene_info({})
        self.assertEqual(result["scene_name"], "Scene")
        self.assertEqual(result["object_count"], 0)
        self.assertEqual(result["objects"], [])

    def test_handle_get_object_info_not_found(self):
        """Test object info handler with non-existent object."""
        with self.assertRaises(ValueError) as context:
            addon.handle_get_object_info({"object_name": "NonExistent"})
        self.assertIn("not found", str(context.exception))

    def test_handle_create_object_invalid_type(self):
        """Test create object with invalid primitive type."""
        with self.assertRaises(ValueError) as context:
            addon.handle_create_object({"type": "INVALID"})
        self.assertIn("Unknown primitive type", str(context.exception))

    def test_handle_delete_object_not_found(self):
        """Test delete object with non-existent object."""
        with self.assertRaises(ValueError) as context:
            addon.handle_delete_object({"name": "NonExistent"})
        self.assertIn("not found", str(context.exception))

    def test_handle_modify_object_not_found(self):
        """Test modify object with non-existent object."""
        with self.assertRaises(ValueError) as context:
            addon.handle_modify_object({"name": "NonExistent"})
        self.assertIn("not found", str(context.exception))

    def test_handle_set_material_object_not_found(self):
        """Test set material with non-existent object."""
        with self.assertRaises(ValueError) as context:
            addon.handle_set_material({
                "object_name": "NonExistent",
                "material_name": "TestMat"
            })
        self.assertIn("not found", str(context.exception))

    def test_handle_set_material_unsupported_type(self):
        """Test set material on unsupported object type."""
        mock_obj = MagicMock()
        mock_obj.type = "EMPTY"
        addon.bpy.data.objects.get = MagicMock(return_value=mock_obj)

        with self.assertRaises(ValueError) as context:
            addon.handle_set_material({
                "object_name": "Empty",
                "material_name": "TestMat"
            })
        self.assertIn("does not support materials", str(context.exception))

    def test_vec3_from_list_helper(self):
        """Test the vec3 from list helper function."""
        self.assertEqual(addon._vec3_from_list([1, 2, 3]), (1.0, 2.0, 3.0))
        self.assertEqual(addon._vec3_from_list([1, 2]), (0.0, 0.0, 0.0))
        self.assertEqual(addon._vec3_from_list(None), (0.0, 0.0, 0.0))
        self.assertEqual(addon._vec3_from_list([1, 2, 3, 4, 5]), (1.0, 2.0, 3.0))

    def test_ok_response_helper(self):
        """Test the ok response formatter."""
        result = addon._ok({"test": "data"})
        self.assertEqual(
            result,
            b'{"status": "ok", "result": {"test": "data"}}\n'
        )

    def test_err_response_helper(self):
        """Test the error response formatter."""
        result = addon._err("Something went wrong")
        self.assertEqual(
            result,
            b'{"status": "error", "message": "Something went wrong"}\n'
        )

    def test_handlers_dict_contains_all_commands(self):
        """Test that all expected handlers are registered."""
        expected_handlers = [
            "get_scene_info",
            "get_object_info",
            "create_object",
            "modify_object",
            "delete_object",
            "set_material",
            "render_image",
            "execute_blender_code",
            "get_polyhaven_categories",
            "search_polyhaven_assets",
            "download_polyhaven_asset",
            "set_texture",
            "set_ollama_model",
            "set_ollama_url",
            "get_ollama_models",
        ]
        for handler in expected_handlers:
            self.assertIn(handler, addon.HANDLERS)
            self.assertTrue(callable(addon.HANDLERS[handler]))


class TestAddonDispatch(unittest.TestCase):
    """Test the command dispatch function."""

    def test_dispatch_unknown_command(self):
        """Test dispatch with unknown command returns error."""
        result = addon._dispatch("unknown_command", {})
        self.assertIn(b"Unknown command", result)
        self.assertIn(b"error", result)

    def test_dispatch_valid_command(self):
        """Test dispatch with valid command returns ok."""
        # Mock a simple handler
        original_handler = addon.HANDLERS.get("get_scene_info")
        mock_handler = MagicMock(return_value={"test": "result"})
        addon.HANDLERS["get_scene_info"] = mock_handler

        try:
            result = addon._dispatch("get_scene_info", {})
            self.assertIn(b"ok", result)
            mock_handler.assert_called_once_with({})
        finally:
            if original_handler:
                addon.HANDLERS["get_scene_info"] = original_handler

    def test_dispatch_handler_raises_exception(self):
        """Test dispatch when handler raises an exception."""
        original_handler = addon.HANDLERS.get("get_scene_info")

        def failing_handler(params):
            raise ValueError("Test error")

        addon.HANDLERS["get_scene_info"] = failing_handler

        try:
            result = addon._dispatch("get_scene_info", {})
            self.assertIn(b"error", result)
            self.assertIn(b"ValueError", result)
            self.assertIn(b"Test error", result)
        finally:
            if original_handler:
                addon.HANDLERS["get_scene_info"] = original_handler


class TestAddonPolyHaven(unittest.TestCase):
    """Test PolyHaven integration handlers."""

    @patch('addon.urllib.request.urlopen')
    def test_handle_get_polyhaven_categories(self, mock_urlopen):
        """Test fetching PolyHaven categories."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'["wood", "metal", "fabric"]'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = addon.handle_get_polyhaven_categories({"asset_type": "textures"})
        self.assertEqual(result, ["wood", "metal", "fabric"])

    @patch('addon.urllib.request.urlopen')
    def test_handle_search_polyhaven_assets(self, mock_urlopen):
        """Test searching PolyHaven assets."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"asset1": {}, "asset2": {}}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = addon.handle_search_polyhaven_assets({"asset_type": "textures"})
        self.assertEqual(result["total"], 2)
        self.assertEqual(len(result["assets"]), 2)


if __name__ == '__main__':
    unittest.main()