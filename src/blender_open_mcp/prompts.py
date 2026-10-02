"""
blender_open_mcp.prompts - MCP Prompts registered on the blender-open-mcp server.

MCP Prompts are reusable templates an agent (or any MCP client) can list and
instantiate via ``prompts/list`` / ``prompts/get``. They give agents a guided
starting point for common workflows and expose the provider-switching story.

Each decorated function's parameters become the prompt's arguments.
"""

from __future__ import annotations

from typing import Optional

from fastmcp import FastMCP
from fastmcp.prompts import prompt

SUPPORTED_PROVIDERS = (
    "openai, openai_compat, ollama, lmstudio, llamacpp, azure (aliases accepted)"
)


@prompt(
    name="blender_build_scene",
    title="Build a Blender scene from a description",
    description=(
        "Render a step-by-step plan for creating a Blender scene that matches "
        "the user's description, using the available blender_* tools."
    ),
)
def blender_build_scene(description: str) -> str:
    """Instructions for building a scene from ``description``."""
    return (
        "You are driving Blender through the blender_open_mcp MCP server.\n\n"
        "User scene request:\n"
        f"{description}\n\n"
        "Plan of action:\n"
        "1. Call blender_get_scene_info first to inspect the current scene and "
        "object names (names are case-sensitive).\n"
        "2. Break the request into small steps. Prefer specific tools "
        "(blender_create_object, blender_modify_object, blender_set_material, "
        "blender_render_image) over blender_execute_code.\n"
        "3. Use blender_execute_code only for complex or algorithmic operations; "
        "variables persist between calls.\n"
        "4. After significant changes, describe what was created, the object "
        "names, and their locations so the user can verify.\n"
    )


@prompt(
    name="blender_review_scene",
    title="Inspect and summarize the current Blender scene",
    description=(
        "Render instructions for inspecting the current Blender scene "
        "read-only and summarizing it for the user."
    ),
)
def blender_review_scene() -> str:
    """Read-only scene review instructions."""
    return (
        "You are inspecting a live Blender session through blender_open_mcp.\n\n"
        "1. Call blender_get_scene_info to enumerate objects, the active "
        "camera, frame range, and render engine.\n"
        "2. For any object the user asks about, call blender_get_object_info "
        "with its exact name.\n"
        "3. Do not modify, create, or delete anything during a review.\n"
        "4. Summarize concisely: object count and types, notable locations, "
        "render settings, and the active camera.\n"
    )



@prompt(
    name="blender_build_geometry_nodes",
    title="Build or edit a Geometry Nodes graph",
    description=(
        "Render a safe tool-first workflow for inspecting, creating, and "
        "validating Geometry Nodes without relying on arbitrary Python."
    ),
)
def blender_build_geometry_nodes(object_name: str, description: str) -> str:
    """Instructions for building Geometry Nodes on ``object_name``."""
    return (
        "You are editing Geometry Nodes through blender_open_mcp.\n\n"
        f"Target object: {object_name}\n"
        f"Requested procedural setup: {description}\n\n"
        "Workflow:\n"
        "1. Call blender_get_selection and blender_get_modifiers to confirm the "
        "target object and existing modifier stack.\n"
        "2. If a Geometry Nodes group already exists, inspect it with "
        "blender_gn_get_tree before changing anything.\n"
        "3. Otherwise create and attach a group with blender_gn_create_group.\n"
        "4. Build the graph incrementally using blender_gn_add_node, "
        "blender_gn_set_node_property, blender_gn_set_input, "
        "blender_gn_connect, and blender_gn_disconnect. Use exact Blender "
        "node bl_idnames.\n"
        "5. Add exposed group inputs/outputs with blender_gn_add_interface_socket "
        "when the user needs modifier-level controls, and set their per-object "
        "values with blender_gn_set_modifier_input.\n"
        "6. Call blender_gn_validate and then blender_gn_get_tree after meaningful "
        "changes to verify the graph.\n"
        "7. Prefer these typed tools over blender_execute_code. Use arbitrary "
        "Python only when the requested operation cannot be expressed with the "
        "Geometry Nodes tools.\n"
    )


@prompt(
    name="blender_configure_llm",
    title="Configure the LLM backend provider",
    description=(
        "Render instructions (with examples) for switching the active LLM "
        "provider used by blender_ai_prompt."
    ),
)
def blender_configure_llm(provider: str, base_url: Optional[str] = None) -> str:
    """Instructions for switching providers, with the target in ``provider``."""
    extra = (
        f"\nTarget base URL: {base_url}" if base_url else "\nUse the provider's default base URL."
    )
    return (
        "You are configuring the LLM backend of the blender_open_mcp server.\n\n"
        f"Requested provider: {provider}{extra}\n\n"
        "1. Call blender_set_llm_provider with provider=<requested provider>, "
        "plus base_url/model/api_key when supplied by the user.\n"
        "2. Verify with blender_get_llm_provider (the API key is masked).\n"
        "3. If the user asks which models are available, call "
        "blender_list_llm_models.\n"
        "4. Test with a short blender_ai_prompt call.\n\n"
        f"Supported providers: {SUPPORTED_PROVIDERS}.\n"
        "Examples:\n"
        '  - provider="ollama", base_url="http://localhost:11434", model="llama3.2"\n'
        '  - provider="lmstudio", base_url="http://localhost:1234/v1"\n'
        '  - provider="llamacpp", base_url="http://localhost:8080/v1"\n'
        '  - provider="openai", api_key="sk-...", model="gpt-4o-mini"\n'
        '  - provider="azure", extra={"resource": "...", "deployment": "..."}\n'
    )


def register_prompts(mcp_server: FastMCP) -> None:
    """Register all prompt templates on a FastMCP instance."""
    mcp_server.add_prompt(blender_build_scene)
    mcp_server.add_prompt(blender_review_scene)
    mcp_server.add_prompt(blender_build_geometry_nodes)
    mcp_server.add_prompt(blender_configure_llm)


__all__ = ["register_prompts"]
