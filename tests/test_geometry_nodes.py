"""Unit tests for the typed Geometry Nodes MCP bridge."""

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


class SocketCollection(list):
    def get(self, key):
        for socket in self:
            if socket.name == key or socket.identifier == key:
                return socket
        return None


class NodeCollection(list):
    def get(self, key):
        for node in self:
            if node.name == key:
                return node
        return None


class LinkCollection(list):
    def new(self, from_socket, to_socket):
        link = SimpleNamespace(
            from_node=from_socket.node,
            from_socket=from_socket,
            to_node=to_socket.node,
            to_socket=to_socket,
            is_valid=True,
        )
        self.append(link)
        to_socket.is_linked = True
        return link


def make_socket(name, identifier=None, default_value=None):
    socket = SimpleNamespace(
        name=name,
        identifier=identifier or name,
        bl_idname="NodeSocketFloat",
        enabled=True,
        is_linked=False,
        is_multi_input=False,
        default_value=default_value,
        node=None,
    )
    return socket


def make_node(name, bl_idname, inputs=None, outputs=None):
    node = SimpleNamespace(
        name=name,
        label="",
        bl_idname=bl_idname,
        location=(0.0, 0.0),
        hide=False,
        inputs=SocketCollection(inputs or []),
        outputs=SocketCollection(outputs or []),
    )
    for socket in list(node.inputs) + list(node.outputs):
        socket.node = node
    return node


def install_tree(tree):
    node_groups = MagicMock()
    node_groups.get.return_value = tree
    addon.bpy.data.node_groups = node_groups


def test_get_selection_reports_active_and_selected():
    cube = SimpleNamespace(name="Cube")
    sphere = SimpleNamespace(name="Sphere")
    addon.bpy.context.selected_objects = [cube, sphere]
    addon.bpy.context.view_layer = SimpleNamespace(
        objects=SimpleNamespace(active=sphere)
    )
    addon.bpy.context.mode = "OBJECT"

    result = addon.handle_get_selection({})

    assert result == {
        "active": "Sphere",
        "selected": ["Cube", "Sphere"],
        "mode": "OBJECT",
    }


def test_gn_get_tree_returns_structured_graph():
    out_socket = make_socket("Geometry", "Geometry")
    in_socket = make_socket("Geometry", "Geometry")
    input_node = make_node("Group Input", "NodeGroupInput", outputs=[out_socket])
    output_node = make_node("Group Output", "NodeGroupOutput", inputs=[in_socket])
    links = LinkCollection()
    links.new(out_socket, in_socket)
    tree = SimpleNamespace(
        name="TestGN",
        bl_idname="GeometryNodeTree",
        nodes=NodeCollection([input_node, output_node]),
        links=links,
    )
    install_tree(tree)

    result = addon.handle_gn_get_tree(
        {"node_group": "TestGN", "include_sockets": True}
    )

    assert result["node_group"] == "TestGN"
    assert result["node_count"] == 2
    assert result["link_count"] == 1
    assert result["nodes"][0]["type"] == "NodeGroupInput"
    assert result["links"][0]["from_node"] == "Group Input"
    assert result["links"][0]["to_node"] == "Group Output"


def test_gn_set_input_sets_default_value():
    value_socket = make_socket("Size", "Size", 1.0)
    node = make_node("Cube", "GeometryNodeMeshCube", inputs=[value_socket])
    tree = SimpleNamespace(
        name="TestGN",
        bl_idname="GeometryNodeTree",
        nodes=NodeCollection([node]),
        links=LinkCollection(),
    )
    install_tree(tree)

    result = addon.handle_gn_set_input(
        {
            "node_group": "TestGN",
            "node_name": "Cube",
            "input_socket": "Size",
            "value": 2.5,
        }
    )

    assert value_socket.default_value == 2.5
    assert result["value"] == 2.5


def test_gn_connect_replaces_existing_single_input_link():
    old_out = make_socket("Geometry", "old_out")
    new_out = make_socket("Geometry", "new_out")
    target = make_socket("Geometry", "target")
    old_node = make_node("Old", "GeometryNodeJoinGeometry", outputs=[old_out])
    new_node = make_node("New", "GeometryNodeJoinGeometry", outputs=[new_out])
    target_node = make_node("Target", "NodeGroupOutput", inputs=[target])

    links = LinkCollection()
    links.new(old_out, target)
    tree = SimpleNamespace(
        name="TestGN",
        bl_idname="GeometryNodeTree",
        nodes=NodeCollection([old_node, new_node, target_node]),
        links=links,
    )
    install_tree(tree)

    result = addon.handle_gn_connect(
        {
            "node_group": "TestGN",
            "from_node": "New",
            "from_socket": "new_out",
            "to_node": "Target",
            "to_socket": "target",
            "replace": True,
        }
    )

    assert len(links) == 1
    assert links[0].from_node.name == "New"
    assert result["is_valid"] is True


def test_set_rna_properties_rejects_private_names():
    target = SimpleNamespace(value=1)
    try:
        addon._set_rna_properties(target, {"_private": 2})
    except ValueError as exc:
        assert "not writable" in str(exc)
    else:
        raise AssertionError("Expected private property to be rejected")


class FakeModifier(dict):
    def __init__(self, name, node_group):
        super().__init__()
        self.name = name
        self.type = "NODES"
        self.node_group = node_group


def test_gn_set_modifier_input_uses_interface_identifier():
    interface_socket = SimpleNamespace(
        item_type="SOCKET",
        name="Density",
        identifier="Socket_2",
        in_out="INPUT",
        bl_socket_idname="NodeSocketFloat",
        default_value=1.0,
    )
    tree = SimpleNamespace(
        name="TestGN",
        bl_idname="GeometryNodeTree",
        interface=SimpleNamespace(items_tree=[interface_socket]),
    )
    modifier = FakeModifier("GeometryNodes", tree)
    modifiers = MagicMock()
    modifiers.get.return_value = modifier
    obj = SimpleNamespace(name="Cube", modifiers=modifiers, update_tag=MagicMock())
    objects = MagicMock()
    objects.get.return_value = obj
    addon.bpy.data.objects = objects

    result = addon.handle_gn_set_modifier_input(
        {
            "object_name": "Cube",
            "modifier_name": "GeometryNodes",
            "input_socket": "Density",
            "value": 12.0,
        }
    )

    assert modifier["Socket_2"] == 12.0
    assert result["identifier"] == "Socket_2"
    obj.update_tag.assert_called_once()


def test_gn_disconnect_removes_targeted_link():
    source_socket = make_socket("Geometry", "source")
    target_socket = make_socket("Geometry", "target")
    source = make_node("Source", "GeometryNodeJoinGeometry", outputs=[source_socket])
    target = make_node("Target", "NodeGroupOutput", inputs=[target_socket])
    links = LinkCollection()
    links.new(source_socket, target_socket)
    tree = SimpleNamespace(
        name="TestGN",
        bl_idname="GeometryNodeTree",
        nodes=NodeCollection([source, target]),
        links=links,
    )
    install_tree(tree)

    result = addon.handle_gn_disconnect(
        {
            "node_group": "TestGN",
            "to_node": "Target",
            "to_socket": "target",
        }
    )

    assert result["removed_count"] == 1
    assert links == []
