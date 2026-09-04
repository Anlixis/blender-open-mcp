"""
blender_open_mcp - provider-agnostic MCP server for Blender.

Supported LLM backends:
- OpenAI-compatible REST APIs (OpenAI, Azure AI Foundry, LM Studio, llama.cpp server, etc.)
- Ollama via both chat/completions and native Ollama endpoints
- Runtime provider switching via MCP tools
"""

from __future__ import annotations

from .server import main

__all__ = ["main"]
