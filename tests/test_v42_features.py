"""Tests for v4.2 generic nodes, transactions, and viewport discovery."""

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


class Nodes(list):
    def get(self, name):
        for node in self:
            if getattr(node, "name", None) == name:
                return node
        return None


def test_generic_material_tree_inspection():
    tree = SimpleNamespace(name="MatTree", nodes=Nodes(), links=[])
    material = SimpleNamespace(name="AgentMat", node_tree=tree, use_nodes=True)
    materials = MagicMock()
    materials.get.return_value = material
    addon.bpy.data.materials = materials

    result = addon.handle_node_get_tree(
        {
            "tree_type": "MATERIAL",
            "target": "AgentMat",
            "include_sockets": True,
        }
    )

    assert result["tree_type"] == "MATERIAL"
    assert result["tree_name"] == "MatTree"
    assert result["owner"] == {"type": "MATERIAL", "name": "AgentMat"}
    assert result["node_count"] == 0


def test_generic_tree_type_validation():
    try:
        addon._normalize_node_tree_type("NOT_A_TREE")
    except ValueError as exc:
        assert "Unknown tree_type" in str(exc)
    else:
        raise AssertionError("Expected invalid node tree type to fail")


def test_transaction_begin_and_rollback_use_single_checkpoint_boundary():
    addon._transaction_state = None
    addon.bpy.ops.ed.undo_push = MagicMock()
    addon.bpy.ops.ed.undo = MagicMock()

    started = addon.handle_transaction_begin({"label": "Test Edit"})
    assert started["label"] == "Test Edit"
    assert addon._transaction_state is not None

    rolled_back = addon.handle_transaction_rollback({})

    assert rolled_back["rolled_back"] is True
    assert addon._transaction_state is None
    assert addon.bpy.ops.ed.undo_push.call_count == 2
    addon.bpy.ops.ed.undo.assert_called_once_with()


def test_transaction_blocks_operator_heavy_commands():
    addon._transaction_state = {
        "id": "tx-test",
        "label": "Safe Node Edit",
        "started_at": 0.0,
        "mutation_count": 0,
    }
    try:
        response = addon._dispatch("create_object", {"type": "CUBE"})
        assert b"blocked while transaction" in response
        assert b"error" in response
    finally:
        addon._transaction_state = None


def test_transaction_counts_typed_mutations():
    addon._transaction_state = {
        "id": "tx-test",
        "label": "Count",
        "started_at": 0.0,
        "mutation_count": 0,
    }
    original = addon.HANDLERS["modify_object"]
    addon.HANDLERS["modify_object"] = MagicMock(return_value={"modified": "Cube"})
    try:
        response = addon._dispatch("modify_object", {"name": "Cube"})
        assert b'"status": "ok"' in response
        assert addon._transaction_state["mutation_count"] == 1
    finally:
        addon.HANDLERS["modify_object"] = original
        addon._transaction_state = None


def test_find_view3d_region_chooses_largest_viewport():
    small_region = SimpleNamespace(type="WINDOW")
    large_region = SimpleNamespace(type="WINDOW")
    small = SimpleNamespace(
        type="VIEW_3D",
        width=100,
        height=100,
        regions=[small_region],
    )
    large = SimpleNamespace(
        type="VIEW_3D",
        width=800,
        height=600,
        regions=[large_region],
    )
    window = SimpleNamespace(screen=SimpleNamespace(areas=[small, large]))
    addon.bpy.context.window_manager = SimpleNamespace(windows=[window])

    found_window, found_area, found_region = addon._find_view3d_region()

    assert found_window is window
    assert found_area is large
    assert found_region is large_region
