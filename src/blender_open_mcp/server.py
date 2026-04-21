"""
blender_open_mcp - MCP Server for Blender3D with Ollama Integration

Architecture:
  - FastMCP server (port 8000): Exposes tools to MCP clients (Claude, Cursor, etc.)
  - Blender add-on socket (port 9876): TCP server inside Blender that executes commands
  - Ollama HTTP API (port 11434): Local LLM for natural language prompt handling

Communication flow:
  MCP Client → FastMCP Server → (TCP) → Blender Add-on → bpy execution → result
  MCP Client → FastMCP Server → (HTTP) → Ollama → LLM response
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import sys
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Logging – use stderr so it doesn't pollute stdio transport
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("blender_mcp")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLENDER_HOST: str = "localhost"
BLENDER_PORT: int = 9876
BLENDER_TIMEOUT: float = 30.0

DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL: str = "llama3.2"

POLYHAVEN_API_BASE: str = "https://api.polyhaven.com"

# ---------------------------------------------------------------------------
# Runtime state (mutable at runtime via set_* tools)
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "ollama_url": DEFAULT_OLLAMA_URL,
    "ollama_model": DEFAULT_OLLAMA_MODEL,
}

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP("blender_open_mcp")


# ===========================================================================
# Shared helpers
# ===========================================================================

def _send_blender_command(command_type: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Send a JSON command to the Blender add-on TCP server and return the parsed response.

    The Blender add-on listens on BLENDER_HOST:BLENDER_PORT.
    Each command is a newline-terminated JSON object:
        {"type": "<command_type>", "params": {...}}
    The response is a JSON object:
        {"status": "ok" | "error", "result": <any> | "message": "<error>"}

    Raises:
        ConnectionRefusedError: If Blender add-on is not running.
        TimeoutError: If the command takes too long.
        ValueError: If the response cannot be parsed.
    """
    payload = json.dumps({"type": command_type, "params": params or {}}) + "\n"
    try:
        with socket.create_connection((BLENDER_HOST, BLENDER_PORT), timeout=BLENDER_TIMEOUT) as sock:
            sock.sendall(payload.encode("utf-8"))
            # Read until connection closes
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks).decode("utf-8").strip()
        response: Dict[str, Any] = json.loads(raw)
        return response
    except ConnectionRefusedError:
        raise ConnectionRefusedError(
            f"Cannot connect to Blender add-on at {BLENDER_HOST}:{BLENDER_PORT}. "
            "Make sure Blender is open with the Blender MCP add-on enabled and the server started "
            "(N-key sidebar → Blender MCP → Start MCP Server)."
        )
    except TimeoutError:
        raise TimeoutError(
            f"Blender add-on did not respond within {BLENDER_TIMEOUT}s. "
            "The operation may still be running in Blender."
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON response from Blender: {exc}. Raw: {raw!r}")


def _format_blender_result(response: Dict[str, Any]) -> str:
    """Convert a Blender TCP response into a human-readable string."""
    if response.get("status") == "error":
        return f"Error from Blender: {response.get('message', 'Unknown error')}"
    result = response.get("result", response)
    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=2)
    return str(result)


async def _query_ollama(prompt: str) -> str:
    """
    Send a prompt to the local Ollama server and return the assistant's reply.

    Uses the /api/generate endpoint with stream=False.
    """
    url = f"{_state['ollama_url']}/api/generate"
    payload = {
        "model": _state["ollama_model"],
        "prompt": prompt,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except httpx.ConnectError:
            return (
                f"Error: Cannot connect to Ollama at {_state['ollama_url']}. "
                "Make sure Ollama is running: `ollama serve`."
            )
        except httpx.HTTPStatusError as exc:
            return f"Error: Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
        except Exception as exc:
            return f"Error communicating with Ollama: {type(exc).__name__}: {exc}"


def _handle_blender_error(exc: Exception) -> str:
    """Produce a friendly, actionable error string from common exceptions."""
    if isinstance(exc, ConnectionRefusedError):
        return str(exc)
    if isinstance(exc, TimeoutError):
        return str(exc)
    if isinstance(exc, ValueError):
        return str(exc)
    return (
        f"Unexpected error communicating with Blender: {type(exc).__name__}: {exc}. "
        "Check the server logs for details."
    )


# ===========================================================================
# Input models (Pydantic v2)
# ===========================================================================

class ResponseFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"


class GetObjectInfoInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    object_name: str = Field(..., description="Name of the Blender object (e.g., 'Cube', 'Camera')", min_length=1, max_length=256)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format: 'markdown' or 'json'")


class ObjectType(str, Enum):
    MESH = "MESH"
    CURVE = "CURVE"
    SURFACE = "SURFACE"
    META = "META"
    FONT = "FONT"
    ARMATURE = "ARMATURE"
    LATTICE = "LATTICE"
    EMPTY = "EMPTY"
    LIGHT = "LIGHT"
    CAMERA = "CAMERA"
    LIGHT_PROBE = "LIGHT_PROBE"
    SPEAKER = "SPEAKER"
    GPENCIL = "GPENCIL"


class PrimitiveType(str, Enum):
    CUBE = "CUBE"
    SPHERE = "SPHERE"
    CYLINDER = "CYLINDER"
    CONE = "CONE"
    TORUS = "TORUS"
    PLANE = "PLANE"
    CIRCLE = "CIRCLE"
    ICO_SPHERE = "ICO_SPHERE"
    GRID = "GRID"
    MONKEY = "MONKEY"


class Vec3(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    x: float = Field(default=0.0, description="X coordinate")
    y: float = Field(default=0.0, description="Y coordinate")
    z: float = Field(default=0.0, description="Z coordinate")

    def as_list(self) -> List[float]:
        return [self.x, self.y, self.z]


class CreateObjectInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    primitive_type: PrimitiveType = Field(
        default=PrimitiveType.CUBE,
        description="Type of primitive mesh to create: CUBE, SPHERE, CYLINDER, CONE, TORUS, PLANE, CIRCLE, ICO_SPHERE, GRID, MONKEY"
    )
    name: Optional[str] = Field(default=None, description="Name for the new object. If None, Blender assigns a default.", max_length=256)
    location: Optional[Vec3] = Field(default=None, description="World-space location {x, y, z}")
    rotation: Optional[Vec3] = Field(default=None, description="Euler rotation in radians {x, y, z}")
    scale: Optional[Vec3] = Field(default=None, description="Scale {x, y, z}")


class ModifyObjectInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    name: str = Field(..., description="Name of the Blender object to modify", min_length=1, max_length=256)
    location: Optional[Vec3] = Field(default=None, description="New world-space location {x, y, z}")
    rotation: Optional[Vec3] = Field(default=None, description="New Euler rotation in radians {x, y, z}")
    scale: Optional[Vec3] = Field(default=None, description="New scale {x, y, z}")
    visible: Optional[bool] = Field(default=None, description="Set viewport visibility (True = visible, False = hidden)")


class DeleteObjectInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    name: str = Field(..., description="Name of the Blender object to delete", min_length=1, max_length=256)


class SetMaterialInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    object_name: str = Field(..., description="Name of the Blender object", min_length=1, max_length=256)
    material_name: str = Field(..., description="Name of the material to create/assign", min_length=1, max_length=256)
    color: Optional[List[float]] = Field(
        default=None,
        description="RGBA color as [R, G, B, A] with values in [0.0, 1.0]. Example: [1.0, 0.0, 0.0, 1.0] for red."
    )

    @field_validator("color")
    @classmethod
    def validate_color(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if len(v) not in (3, 4):
            raise ValueError("color must be [R, G, B] or [R, G, B, A]")
        for c in v:
            if not (0.0 <= c <= 1.0):
                raise ValueError(f"Color channel {c} out of range [0.0, 1.0]")
        if len(v) == 3:
            v = v + [1.0]
        return v


class RenderImageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    file_path: str = Field(
        ...,
        description="Absolute file path to save the render output (e.g., '/tmp/render.png'). Blender must have write access.",
        min_length=1,
        max_length=1024
    )


class ExecuteCodeInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    code: str = Field(
        ...,
        description="Python code to execute inside Blender's Python environment via bpy. Use with caution.",
        min_length=1
    )


class PolyHavenAssetType(str, Enum):
    HDRIs = "hdris"
    TEXTURES = "textures"
    MODELS = "models"
    ALL = "all"


class GetPolyHavenCategoriesInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    asset_type: PolyHavenAssetType = Field(
        default=PolyHavenAssetType.TEXTURES,
        description="Asset type to list categories for: 'hdris', 'textures', 'models', or 'all'"
    )


class SearchPolyHavenInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    asset_type: PolyHavenAssetType = Field(
        default=PolyHavenAssetType.TEXTURES,
        description="Asset type to search: 'hdris', 'textures', 'models', or 'all'"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Category filter list (e.g., ['wood', 'metal']). If None, returns all assets."
    )
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip for pagination")


class DownloadPolyHavenInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    asset_id: str = Field(..., description="PolyHaven asset ID (e.g., 'brick_wall_001')", min_length=1, max_length=128)
    asset_type: PolyHavenAssetType = Field(
        default=PolyHavenAssetType.TEXTURES,
        description="Type of asset: 'hdris', 'textures', or 'models'"
    )
    resolution: str = Field(
        default="1k",
        description="Resolution string (e.g., '1k', '2k', '4k', '8k')"
    )
    file_format: str = Field(
        default="jpg",
        description="File format (e.g., 'jpg', 'png', 'exr', 'blend', 'gltf', 'fbx')"
    )


class SetTextureInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    object_name: str = Field(..., description="Name of the Blender object to texture", min_length=1, max_length=256)
    texture_id: str = Field(..., description="PolyHaven asset ID of the already-downloaded texture", min_length=1, max_length=128)


class SetOllamaModelInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    model_name: str = Field(..., description="Ollama model name (e.g., 'llama3.2', 'gemma3', 'mistral')", min_length=1, max_length=128)


class SetOllamaUrlInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    url: str = Field(
        ...,
        description="Ollama server base URL (e.g., 'http://localhost:11434')",
        min_length=1,
        max_length=512
    )


class PromptInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    prompt: str = Field(..., description="Natural language prompt to send to the Ollama model", min_length=1)
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to prepend. Defaults to a Blender-expert system prompt."
    )


# ===========================================================================
# MCP Tools — Scene & Object Management
# ===========================================================================

@mcp.tool(
    name="blender_get_scene_info",
    annotations={
        "title": "Get Blender Scene Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def blender_get_scene_info() -> str:
    """
    Retrieve a full summary of the current Blender scene.

    Returns scene name, frame range, render settings, list of all objects
    (with their types, locations, and visibility), active camera, and world info.

    Returns:
        str: JSON-formatted scene summary or an error message.
    """
    try:
        response = _send_blender_command("get_scene_info")
        return _format_blender_result(response)
    except Exception as exc:
        return _handle_blender_error(exc)


@mcp.tool(
    name="blender_get_object_info",
    annotations={
        "title": "Get Blender Object Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def blender_get_object_info(params: GetObjectInfoInput) -> str:
    """
    Retrieve detailed information about a specific Blender object by name.

    Returns the object's type, world matrix, location, rotation, scale,
    material slots, mesh vertex count (for MESH objects), and custom properties.

    Args:
        params (GetObjectInfoInput):
            - object_name (str): Exact name of the object in the Blender scene.
            - response_format (ResponseFormat): 'markdown' or 'json'.

    Returns:
        str: Formatted object info or an error message if the object doesn't exist.
    """
    try:
        response = _send_blender_command("get_object_info", {"object_name": params.object_name})
        result = _format_blender_result(response)
        if params.response_format == ResponseFormat.MARKDOWN:
            return f"## Object: {params.object_name}\n\n```json\n{result}\n```"
        return result
    except Exception as exc:
        return _handle_blender_error(exc)


@mcp.tool(
    name="blender_create_object",
    annotations={
        "title": "Create Blender 3D Object",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def blender_create_object(params: CreateObjectInput) -> str:
    """
    Create a new primitive mesh object in the Blender scene.

    Supported primitives: CUBE, SPHERE, CYLINDER, CONE, TORUS, PLANE,
    CIRCLE, ICO_SPHERE, GRID, MONKEY.

    Args:
        params (CreateObjectInput):
            - primitive_type (PrimitiveType): Mesh primitive to add.
            - name (Optional[str]): Desired object name. Blender assigns a default if None.
            - location (Optional[Vec3]): World-space position {x, y, z}.
            - rotation (Optional[Vec3]): Euler angles in radians {x, y, z}.
            - scale (Optional[Vec3]): Scale factors {x, y, z}.

    Returns:
        str: Confirmation with the created object's name, or an error message.
    """
    cmd_params: Dict[str, Any] = {"type": params.primitive_type.value}
    if params.name:
        cmd_params["name"] = params.name
    if params.location:
        cmd_params["location"] = params.location.as_list()
    if params.rotation:
        cmd_params["rotation"] = params.rotation.as_list()
    if params.scale:
        cmd_params["scale"] = params.scale.as_list()
    try:
        response = _send_blender_command("create_object", cmd_params)
        return _format_blender_result(response)
    except Exception as exc:
        return _handle_blender_error(exc)


@mcp.tool(
    name="blender_modify_object",
    annotations={
        "title": "Modify Blender Object Properties",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def blender_modify_object(params: ModifyObjectInput) -> str:
    """
    Modify an existing Blender object's transform properties or visibility.

    Only the fields you provide will be updated; omitted fields remain unchanged.

    Args:
        params (ModifyObjectInput):
            - name (str): Exact name of the object to modify.
            - location (Optional[Vec3]): New world-space location {x, y, z}.
            - rotation (Optional[Vec3]): New Euler rotation in radians {x, y, z}.
            - scale (Optional[Vec3]): New scale {x, y, z}.
            - visible (Optional[bool]): Viewport visibility toggle.

    Returns:
        str: Confirmation of changes applied, or an error message.
    """
    cmd_params: Dict[str, Any] = {"name": params.name}
    if params.location:
        cmd_params["location"] = params.location.as_list()
    if params.rotation:
        cmd_params["rotation"] = params.rotation.as_list()
    if params.scale:
        cmd_params["scale"] = params.scale.as_list()
    if params.visible is not None:
        cmd_params["visible"] = params.visible
    try:
        response = _send_blender_command("modify_object", cmd_params)
        return _format_blender_result(response)
    except Exception as exc:
        return _handle_blender_error(exc)


@mcp.tool(
    name="blender_delete_object",
    annotations={
        "title": "Delete Blender Object",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def blender_delete_object(params: DeleteObjectInput) -> str:
    """
    Permanently delete an object from the current Blender scene.

    Warning: This operation is destructive. The object and its data will be removed.
    Use blender_get_scene_info first to confirm the correct object name.

    Args:
        params (DeleteObjectInput):
            - name (str): Exact name of the object to delete.

    Returns:
        str: Confirmation message or an error if the object was not found.
    """
    try:
        response = _send_blender_command("delete_object", {"name": params.name})
        return _format_blender_result(response)
    except Exception as exc:
        return _handle_blender_error(exc)


# ===========================================================================
# MCP Tools — Materials & Rendering
# ===========================================================================

@mcp.tool(
    name="blender_set_material",
    annotations={
        "title": "Set / Create Material on Object",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def blender_set_material(params: SetMaterialInput) -> str:
    """
    Create a new Principled BSDF material and assign it to a Blender object.

    If a material with the given name already exists, it will be reused and its
    base color updated if a color is provided.

    Args:
        params (SetMaterialInput):
            - object_name (str): Object to receive the material.
            - material_name (str): Name for the material.
            - color (Optional[List[float]]): RGBA color [R, G, B, A] in [0.0, 1.0].
              Example: [0.8, 0.2, 0.1, 1.0] for a reddish color.

    Returns:
        str: Confirmation or error message.
    """
    cmd_params: Dict[str, Any] = {
        "object_name": params.object_name,
        "material_name": params.material_name,
    }
    if params.color:
        cmd_params["color"] = params.color
    try:
        response = _send_blender_command("set_material", cmd_params)
        return _format_blender_result(response)
    except Exception as exc:
        return _handle_blender_error(exc)


@mcp.tool(
    name="blender_render_image",
    annotations={
        "title": "Render Current Scene to File",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def blender_render_image(params: RenderImageInput) -> str:
    """
    Trigger a render of the current Blender scene and save the result to disk.

    Uses the scene's current render engine, resolution, and camera settings.
    The file_path must use an absolute path that Blender can write to.

    Args:
        params (RenderImageInput):
            - file_path (str): Absolute output path (e.g., '/tmp/output.png').

    Returns:
        str: Confirmation with the saved path, or an error message.
    """
    try:
        response = _send_blender_command("render_image", {"file_path": params.file_path})
        return _format_blender_result(response)
    except Exception as exc:
        return _handle_blender_error(exc)


# ===========================================================================
# MCP Tools — Advanced / Code Execution
# ===========================================================================

@mcp.tool(
    name="blender_execute_code",
    annotations={
        "title": "Execute Python Code in Blender",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def blender_execute_code(params: ExecuteCodeInput) -> str:
    """
    Execute arbitrary Python code inside Blender using the bpy module.

    Use this for complex operations not covered by specific tools, such as
    node graph manipulation, animation keyframing, or batch operations.

    WARNING: This tool runs with full Blender Python access. Use with caution.
    Avoid destructive operations unless intentional.

    Args:
        params (ExecuteCodeInput):
            - code (str): Valid Python code using bpy, mathutils, etc.
              Example: "import bpy\\nbpy.ops.object.select_all(action='SELECT')\\nprint(len(bpy.context.selected_objects))"

    Returns:
        str: stdout/result from the executed code, or an error with traceback info.
    """
    try:
        response = _send_blender_command("execute_blender_code", {"code": params.code})
        return _format_blender_result(response)
    except Exception as exc:
        return _handle_blender_error(exc)


# ===========================================================================
# MCP Tools — PolyHaven Asset Integration
# ===========================================================================

@mcp.tool(
    name="blender_get_polyhaven_categories",
    annotations={
        "title": "List PolyHaven Asset Categories",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def blender_get_polyhaven_categories(params: GetPolyHavenCategoriesInput) -> str:
    """
    Fetch the list of available asset categories from the PolyHaven API.

    PolyHaven provides free CC0 HDRIs, textures, and 3D models. Use this tool
    to discover available categories before searching or downloading assets.

    Args:
        params (GetPolyHavenCategoriesInput):
            - asset_type (PolyHavenAssetType): 'hdris', 'textures', 'models', or 'all'.

    Returns:
        str: JSON list of category names or an error message.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            url = f"{POLYHAVEN_API_BASE}/categories/{params.asset_type.value}"
            resp = await client.get(url)
            resp.raise_for_status()
            categories = resp.json()
        return json.dumps(categories, indent=2)
    except httpx.HTTPStatusError as exc:
        return f"Error: PolyHaven API returned HTTP {exc.response.status_code}: {exc.response.text}"
    except httpx.ConnectError:
        return "Error: Cannot reach PolyHaven API. Check your internet connection."
    except Exception as exc:
        return f"Error fetching PolyHaven categories: {type(exc).__name__}: {exc}"


@mcp.tool(
    name="blender_search_polyhaven_assets",
    annotations={
        "title": "Search PolyHaven Assets",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def blender_search_polyhaven_assets(params: SearchPolyHavenInput) -> str:
    """
    Search the PolyHaven asset library by type and optional category filters.

    Returns paginated asset metadata including IDs, names, categories, and
    download availability. Use the asset_id from results with blender_download_polyhaven_asset.

    Args:
        params (SearchPolyHavenInput):
            - asset_type (PolyHavenAssetType): Asset type to search.
            - categories (Optional[List[str]]): Category filters (e.g., ['wood', 'floor']).
            - limit (int): Max results to return (1-100, default 20).
            - offset (int): Pagination offset.

    Returns:
        str: JSON object with items, total, has_more, next_offset.
    """
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            url = f"{POLYHAVEN_API_BASE}/assets"
            query: Dict[str, Any] = {"type": params.asset_type.value}
            if params.categories:
                query["categories"] = ",".join(params.categories)
            resp = await client.get(url, params=query)
            resp.raise_for_status()
            all_assets: Dict[str, Any] = resp.json()

        items = list(all_assets.items())
        total = len(items)
        page = items[params.offset: params.offset + params.limit]
        result = {
            "total": total,
            "count": len(page),
            "offset": params.offset,
            "has_more": total > params.offset + len(page),
            "next_offset": params.offset + len(page) if total > params.offset + len(page) else None,
            "items": [
                {"id": k, **v}
                for k, v in page
            ],
        }
        return json.dumps(result, indent=2)
    except httpx.HTTPStatusError as exc:
        return f"Error: PolyHaven API returned HTTP {exc.response.status_code}"
    except httpx.ConnectError:
        return "Error: Cannot reach PolyHaven API. Check your internet connection."
    except Exception as exc:
        return f"Error searching PolyHaven: {type(exc).__name__}: {exc}"


@mcp.tool(
    name="blender_download_polyhaven_asset",
    annotations={
        "title": "Download PolyHaven Asset into Blender",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def blender_download_polyhaven_asset(params: DownloadPolyHavenInput) -> str:
    """
    Download a PolyHaven asset and import it into the active Blender scene.

    This tool first resolves the asset's download URL from the PolyHaven API,
    then instructs the Blender add-on to download and import the asset.
    For HDRIs, sets the world environment. For textures/models, places them in the scene.

    Args:
        params (DownloadPolyHavenInput):
            - asset_id (str): PolyHaven asset slug (e.g., 'brick_wall_001').
            - asset_type (PolyHavenAssetType): 'hdris', 'textures', or 'models'.
            - resolution (str): Download resolution ('1k', '2k', '4k', '8k').
            - file_format (str): File format ('jpg', 'png', 'exr', 'blend', 'gltf', 'fbx').

    Returns:
        str: Confirmation that the asset was downloaded and imported, or an error.
    """
    # First resolve the download URL from PolyHaven API
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            files_url = f"{POLYHAVEN_API_BASE}/files/{params.asset_id}"
            resp = await client.get(files_url)
            resp.raise_for_status()
            files_data: Dict[str, Any] = resp.json()
    except httpx.ConnectError:
        return "Error: Cannot reach PolyHaven API. Check your internet connection."
    except httpx.HTTPStatusError as exc:
        return f"Error: PolyHaven API returned HTTP {exc.response.status_code} for asset '{params.asset_id}'"
    except Exception as exc:
        return f"Error fetching PolyHaven asset info: {type(exc).__name__}: {exc}"

    # Instruct Blender to download and import
    try:
        cmd_params: Dict[str, Any] = {
            "asset_id": params.asset_id,
            "asset_type": params.asset_type.value,
            "resolution": params.resolution,
            "file_format": params.file_format,
            "files_data": files_data,
        }
        response = _send_blender_command("download_polyhaven_asset", cmd_params)
        return _format_blender_result(response)
    except Exception as exc:
        return _handle_blender_error(exc)


@mcp.tool(
    name="blender_set_texture",
    annotations={
        "title": "Apply Downloaded PolyHaven Texture to Object",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def blender_set_texture(params: SetTextureInput) -> str:
    """
    Apply a previously downloaded PolyHaven texture to a Blender object.

    The texture must have been downloaded first using blender_download_polyhaven_asset.
    Creates a new material with the texture connected to a Principled BSDF shader.

    Args:
        params (SetTextureInput):
            - object_name (str): Name of the Blender object to texture.
            - texture_id (str): PolyHaven asset ID of the downloaded texture.

    Returns:
        str: Confirmation or error message.
    """
    try:
        response = _send_blender_command("set_texture", {
            "object_name": params.object_name,
            "texture_id": params.texture_id,
        })
        return _format_blender_result(response)
    except Exception as exc:
        return _handle_blender_error(exc)


# ===========================================================================
# MCP Tools — Ollama / AI Integration
# ===========================================================================

@mcp.tool(
    name="blender_ai_prompt",
    annotations={
        "title": "Send Natural Language Prompt to Ollama",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def blender_ai_prompt(params: PromptInput) -> str:
    """
    Send a natural language prompt to the local Ollama model.

    Ollama processes the prompt and returns an AI-generated response. This tool
    is useful for getting Blender scripting help, asset suggestions, or generating
    bpy code that can then be executed with blender_execute_code.

    The default system prompt positions the model as a Blender expert.

    Args:
        params (PromptInput):
            - prompt (str): Your question or instruction in natural language.
            - system_prompt (Optional[str]): Override the default Blender-expert system prompt.

    Returns:
        str: The Ollama model's response text.
    """
    system = params.system_prompt or (
        "You are an expert Blender 3D artist and Python developer. "
        "You help users control Blender using the bpy Python API. "
        "Provide concise, runnable Python code examples when appropriate. "
        f"Current Ollama model: {_state['ollama_model']}."
    )
    full_prompt = f"{system}\n\nUser: {params.prompt}\n\nAssistant:"
    return await _query_ollama(full_prompt)


@mcp.tool(
    name="blender_set_ollama_model",
    annotations={
        "title": "Set Ollama Model",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def blender_set_ollama_model(params: SetOllamaModelInput) -> str:
    """
    Switch the Ollama language model used for AI prompts.

    The model must already be pulled in Ollama (use `ollama pull <model>`).
    Use blender_get_ollama_models to list available models.

    Args:
        params (SetOllamaModelInput):
            - model_name (str): Ollama model name (e.g., 'llama3.2', 'gemma3', 'mistral').

    Returns:
        str: Confirmation of the model change.
    """
    _state["ollama_model"] = params.model_name
    return f"Ollama model set to: {params.model_name}"


@mcp.tool(
    name="blender_set_ollama_url",
    annotations={
        "title": "Set Ollama Server URL",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def blender_set_ollama_url(params: SetOllamaUrlInput) -> str:
    """
    Update the Ollama server base URL.

    Useful if Ollama is running on a non-default host or port, or on a remote machine.

    Args:
        params (SetOllamaUrlInput):
            - url (str): New base URL (e.g., 'http://192.168.1.100:11434').

    Returns:
        str: Confirmation of the URL change.
    """
    _state["ollama_url"] = params.url
    return f"Ollama URL set to: {params.url}"


@mcp.tool(
    name="blender_get_ollama_models",
    annotations={
        "title": "List Available Ollama Models",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def blender_get_ollama_models() -> str:
    """
    List all locally available Ollama models.

    Queries the Ollama /api/tags endpoint to show models that have been pulled
    and are ready to use. Returns name, size, and modification date for each model.

    Returns:
        str: JSON-formatted list of available models or an error message.
    """
    url = f"{_state['ollama_url']}/api/tags"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            models = [
                {
                    "name": m.get("name"),
                    "size_gb": round(m.get("size", 0) / 1e9, 2),
                    "modified_at": m.get("modified_at", ""),
                }
                for m in data.get("models", [])
            ]
            return json.dumps({"current_model": _state["ollama_model"], "available_models": models}, indent=2)
        except httpx.ConnectError:
            return (
                f"Error: Cannot connect to Ollama at {_state['ollama_url']}. "
                "Make sure Ollama is running: `ollama serve`."
            )
        except httpx.HTTPStatusError as exc:
            return f"Error: Ollama returned HTTP {exc.response.status_code}"
        except Exception as exc:
            return f"Error listing Ollama models: {type(exc).__name__}: {exc}"


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """CLI entry point registered as `blender-mcp` in pyproject.toml."""
    global BLENDER_HOST, BLENDER_PORT
    import argparse

    parser = argparse.ArgumentParser(
        prog="blender-mcp",
        description="blender-open-mcp: MCP server for controlling Blender3D with local AI models.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="FastMCP server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="FastMCP server port (default: 8000)")
    parser.add_argument("--blender-host", default=BLENDER_HOST, help="Blender add-on TCP host (default: localhost)")
    parser.add_argument("--blender-port", type=int, default=BLENDER_PORT, help="Blender add-on TCP port (default: 9876)")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help="Default Ollama model (default: llama3.2)")
    parser.add_argument("--transport", choices=["streamable_http", "stdio"], default="streamable_http",
                        help="MCP transport (default: streamable_http)")
    args = parser.parse_args()

    # Apply runtime configuration
    BLENDER_HOST = args.blender_host
    BLENDER_PORT = args.blender_port
    _state["ollama_url"] = args.ollama_url
    _state["ollama_model"] = args.ollama_model

    logger.info("Starting blender-open-mcp server")
    logger.info("  FastMCP transport : %s", args.transport)
    if args.transport == "streamable_http":
        logger.info("  FastMCP endpoint  : http://%s:%d", args.host, args.port)
    logger.info("  Blender add-on    : %s:%d", args.blender_host, args.blender_port)
    logger.info("  Ollama URL        : %s", _state["ollama_url"])
    logger.info("  Ollama model      : %s", _state["ollama_model"])

    if args.transport == "streamable_http":
        mcp.run(transport="streamable_http", host=args.host, port=args.port)
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()