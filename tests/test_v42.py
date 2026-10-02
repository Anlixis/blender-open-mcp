"""Tests for v4.2 transactions, generic node API, and viewport helpers."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

bpy_mock = MagicMock()
bpy_mock.props = MagicMock()
sys.modules.setdefault("bpy", bpy_mock)
sys.modules.setdefault("bpy.props", bpy_mock.props)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import addon  # noqa: E402


class NamedCollection(list):
    def get(self, name):
        for item in self:
            if getattr(item, "name", None) == name:
                return item
        return None


class FakeLinks(list):
    def new(self, from_socket, to_socket):
        link = SimpleNamespace(
            from_node=from_socket.node,
            from_socket=from_socket,
            to_node=to_socket.node,
            to_socket=to_socket,
            is_valid=True,
        )
        self.append(link)
        return link


def make_socket(name, identifier=None, default_value=0.0):
    return SimpleNamespace(
        name=name,
        identifier=identifier or name,
        bl_idname="NodeSocketFloat",
        enabled=True,
        is_linked=False,
        is_multi_input=False,
        default_value=default_value,
        node=None,
    )


def make_node(name, bl_idname="ShaderNodeValue", inputs=None, outputs=None):
    node = SimpleNamespace(
        name=name,
        label="",
        bl_idname=bl_idname,
        location=(0.0, 0.0),
        hide=False,
        inputs=NamedCollection(inputs or []),
        outputs=NamedCollection(outputs or []),
    )
    for socket in list(node.inputs) + list(node.outputs):
        socket.node = node
    return node


def test_generic_material_tree_inspection():
    out_socket = make_socket("Value", "Value")
    in_socket = make_socket("Surface", "Surface")
    source = make_node("Value", outputs=[out_socket])
    target = make_node("Output", "ShaderNodeOutputMaterial", inputs=[in_socket])
    links = FakeLinks()
    links.new(out_socket, in_socket)
    tree = SimpleNamespace(
        name="Material NodeTree",
        bl_idname="ShaderNodeTree",
        nodes=NamedCollection([source, target]),
        links=links,
    )
    material = SimpleNamespace(name="TestMat", use_nodes=True, node_tree=tree)
    materials = MagicMock()
    materials.get.return_value = material
    addon.bpy.data.materials = materials

    result = addon.handle_node_get_tree(
        {"tree_type": "MATERIAL", "tree_name": "TestMat"}
    )

    assert result["tree_type"] == "MATERIAL"
    assert result["owner"] == "TestMat"
    assert result["node_count"] == 2
    assert result["link_count"] == 1


def test_generic_node_set_input():
    value_socket = make_socket("Scale", "Scale", 1.0)
    node = make_node("Math", "ShaderNodeMath", inputs=[value_socket])
    tree = SimpleNamespace(
        name="Material NodeTree",
        bl_idname="ShaderNodeTree",
        nodes=NamedCollection([node]),
        links=FakeLinks(),
    )
    material = SimpleNamespace(name="TestMat", use_nodes=True, node_tree=tree)
    materials = MagicMock()
    materials.get.return_value = material
    addon.bpy.data.materials = materials

    result = addon.handle_node_set_input(
        {
            "tree_type": "MATERIAL",
            "tree_name": "TestMat",
            "node_name": "Math",
            "input_socket": "Scale",
            "value": 2.0,
        }
    )

    assert value_socket.default_value == 2.0
    assert result["value"] == 2.0


class FakeScene(dict):
    def __init__(self, name="Scene"):
        super().__init__()
        self.name = name


def test_transaction_begin_and_commit():
    scene = FakeScene()
    addon.bpy.context.scene = scene
    addon._active_transaction_id = None
    addon.bpy.ops.ed.undo_push = MagicMock(return_value={"FINISHED"})

    started = addon.handle_transaction_begin({"label": "test"})
    tx_id = started["transaction_id"]

    assert addon._active_transaction_id == tx_id
    assert scene[addon._TRANSACTION_PROP] == f"{tx_id}:active"

    committed = addon.handle_transaction_commit({"transaction_id": tx_id})

    assert committed["status"] == "committed"
    assert addon._active_transaction_id is None
    assert addon._TRANSACTION_PROP not in scene


def test_viewport_screenshot_rejects_invalid_shading_before_context_lookup():
    try:
        addon.handle_viewport_screenshot({"shading": "NOT_A_MODE"})
    except ValueError as exc:
        assert "shading must be one of" in str(exc)
    else:
        raise AssertionError("Expected invalid shading to be rejected")
