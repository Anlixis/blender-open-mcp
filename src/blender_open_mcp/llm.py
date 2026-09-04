"""
blender_open_mcp.llm - provider-agnostic LLM backend.

Supported providers (extensible via PROVIDERS registry):

- ``openai``            : OpenAI REST API (chat/completions + /models)
- ``openai_compat``     : any OpenAI-compatible endpoint (vLLM, TGI, OpenRouter,
                          Groq, Together, llama.cpp server, LM Studio, ...)
- ``lmstudio``          : LM Studio local server (OpenAI-compatible, port 1234)
- ``llamacpp``          : llama.cpp server (OpenAI-compatible, port 8080)
- ``ollama``            : Ollama native /api/chat (+ /api/tags), falls back to
                          OpenAI-compatible /v1/chat/completions when the
                          configured base URL already ends in /v1
- ``azure``             : Azure AI Foundry / Azure OpenAI (api-key auth,
                          deployment-scoped chat/completions URL)

Provider-specific helpers live next to the generic OpenAI-compatible path so
the same MCP tools work against every backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

__all__ = [
    "ProviderError",
    "PROVIDERS",
    "resolve_provider",
    "chat",
    "list_models",
]

DEFAULT_TIMEOUT = 60.0


class ProviderError(Exception):
    """Raised when a configured LLM provider cannot be reached or replies badly."""


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
# style: "openai" (chat/completions), "ollama" (/api/chat), "azure" (deployment URL)
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "style": "openai",
        "default_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "auth": "bearer",
        "label": "OpenAI",
    },
    "openai_compat": {
        "style": "openai",
        "default_url": "http://localhost:8000/v1",
        "default_model": None,
        "auth": "optional_bearer",
        "label": "OpenAI-compatible endpoint",
    },
    "lmstudio": {
        "style": "openai",
        "default_url": "http://localhost:1234/v1",
        "default_model": None,
        "auth": "optional_bearer",
        "label": "LM Studio",
    },
    "llamacpp": {
        "style": "openai",
        "default_url": "http://localhost:8080/v1",
        "default_model": None,
        "auth": "optional_bearer",
        "label": "llama.cpp server",
    },
    "ollama": {
        "style": "ollama",
        "default_url": "http://localhost:11434",
        "default_model": "llama3.2",
        "auth": None,
        "label": "Ollama",
    },
    "azure": {
        "style": "azure",
        "default_url": "https://RESOURCE.openai.azure.com",
        "default_model": None,
        "auth": "api_key",
        "label": "Azure AI Foundry / Azure OpenAI",
    },
}

# Aliases accepted anywhere a provider name is passed.
PROVIDER_ALIASES: Dict[str, str] = {
    "openai-compatible": "openai_compat",
    "openai_compatible": "openai_compat",
    "openai-compat": "openai_compat",
    "generic": "openai_compat",
    "vllm": "openai_compat",
    "tgi": "openai_compat",
    "text-generation-inference": "openai_compat",
    "openrouter": "openai_compat",
    "groq": "openai_compat",
    "together": "openai_compat",
    "mistral": "openai_compat",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "llama.cpp": "llamacpp",
    "llama_cpp": "llamacpp",
    "llama-cpp": "llamacpp",
    "azure-openai": "azure",
    "azure_openai": "azure",
    "azure_ai_foundry": "azure",
}


def resolve_provider(name: Optional[str]) -> str:
    """Return the canonical provider key, normalizing common aliases."""
    if not name:
        raise ProviderError("No LLM provider specified.")
    key = str(name).strip().lower()
    key = PROVIDER_ALIASES.get(key, key)
    if key not in PROVIDERS:
        raise ProviderError(
            f"Unknown LLM provider '{name}'. Supported: {sorted(PROVIDERS)}"
        )
    return key


def provider_spec(name: str) -> Dict[str, Any]:
    """Return the provider spec dict for a (possibly aliased) provider name."""
    return PROVIDERS[resolve_provider(name)]


# ---------------------------------------------------------------------------
# Headers / request builders
# ---------------------------------------------------------------------------

def _base_url(url: str) -> str:
    return url.rstrip("/")


def _headers(style: str, api_key: Optional[str], extra: Optional[Dict[str, Any]]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if style == "azure":
        key = api_key or (extra or {}).get("api_key")
        if key:
            headers["api-key"] = key
    elif style in ("openai", "ollama"):
        # OpenAI-compatible servers commonly accept a Bearer token; Ollama's
        # OpenAI compatibility layer does too. Native Ollama ignores it.
        key = api_key or (extra or {}).get("api_key")
        if key:
            headers["Authorization"] = f"Bearer {key}"
    return headers


def _chat_url(
    provider: str,
    spec: Dict[str, Any],
    base_url: Optional[str],
    extra: Optional[Dict[str, Any]],
) -> str:
    style = spec["style"]
    url = _base_url(base_url or spec["default_url"])
    extra = extra or {}

    if style == "ollama":
        # If the user pointed us at an OpenAI-compatible /v1 URL (e.g. a proxy
        # or Ollama's own /v1), use the OpenAI-compatible endpoint instead.
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/api/chat"

    if style == "azure":
        resource = extra.get("resource")
        deployment = extra.get("deployment")
        api_version = extra.get("api_version", "2024-06-01")
        if not resource or not deployment:
            raise ProviderError(
                "Azure provider requires 'extra' params: resource, deployment "
                "(api_version optional). Example: "
                'extra={"resource": "myres", "deployment": "gpt-4o"}'
            )
        base = url if "RESOURCE" not in url else f"https://{resource}.openai.azure.com"
        return (
            f"{_base_url(base)}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={api_version}"
        )

    # openai style
    path = "chat/completions"
    if not url.endswith("/chat/completions"):
        path = f"{url}/chat/completions"
        if url.endswith("/v1"):
            pass  # already correct
    return path if url.endswith("/chat/completions") else f"{url}/chat/completions"


def _chat_payload(
    provider: str,
    spec: Dict[str, Any],
    model: Optional[str],
    messages: List[Dict[str, str]],
    temperature: Optional[float],
    max_tokens: Optional[int],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    extra = extra or {}
    style = spec["style"]
    model = model or extra.get("model") or spec.get("default_model")

    payload: Dict[str, Any] = {"messages": messages, "stream": False}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    if style in ("openai", "ollama"):
        if model:
            payload["model"] = model
        # Pass through any extra keys the server accepts (top_p, seed, ...)
        for k, v in extra.items():
            if k not in ("model", "api_key", "resource", "deployment", "api_version"):
                payload.setdefault(k, v)
    elif style == "azure":
        # Model name is the deployment; include it as model too for parity.
        if model:
            payload["model"] = model
    return payload


def _extract_content(data: Dict[str, Any]) -> str:
    """Normalize responses from OpenAI-style and Ollama-native endpoints."""
    if "choices" in data and isinstance(data.get("choices"), list) and data["choices"]:
        choice = data["choices"][0]
        if isinstance(choice, dict):
            msg = choice.get("message") or {}
            if isinstance(msg, dict):
                text = msg.get("content")
                if text:
                    return str(text)
            # Some servers return text directly on the choice.
            text = choice.get("text")
            if text:
                return str(text)
    if isinstance(data.get("message"), dict):
        text = data["message"].get("content")
        if text:
            return str(text)
    if data.get("response"):  # Ollama /api/generate fallback
        return str(data["response"])
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def chat(
    messages: List[Dict[str, str]],
    *,
    provider: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """
    Send a chat completion request to the given provider and return the text.

    ``provider`` may be any key in PROVIDERS (or an alias). ``extra`` carries
    provider-specific settings (e.g. Azure resource/deployment/api_version).
    """
    provider_key = resolve_provider(provider)
    spec = PROVIDERS[provider_key]
    extra = dict(extra or {})

    url = _chat_url(provider_key, spec, base_url, extra)
    payload = _chat_payload(
        provider_key, spec, model, messages, temperature, max_tokens, extra
    )
    headers = _headers(spec["style"], api_key, extra)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError as exc:
        raise ProviderError(
            f"Cannot connect to {provider_key} provider at {url}. "
            f"Make sure the server is running. ({exc})"
        )
    except httpx.TimeoutException:
        raise ProviderError(
            f"Provider {provider_key} at {url} timed out after {timeout}s."
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response is not None else ""
        raise ProviderError(
            f"Provider {provider_key} returned HTTP {exc.response.status_code}: {detail}"
        )
    except Exception as exc:
        raise ProviderError(
            f"Error calling provider {provider_key}: {type(exc).__name__}: {exc}"
        )

    text = _extract_content(data)
    if not text:
        raise ProviderError(
            f"Provider {provider_key} returned an empty or unparseable response: "
            f"{str(data)[:500]}"
        )
    return text.strip()


async def list_models(
    *,
    provider: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> List[Dict[str, Any]]:
    """
    List locally/remotely available models for a provider.

    - Ollama native: GET /api/tags
    - Everything else (OpenAI-compatible): GET /models
    - Azure: not currently exposed (deployments are managed in the portal);
      raises ProviderError with guidance.
    """
    provider_key = resolve_provider(provider)
    spec = PROVIDERS[provider_key]
    extra = dict(extra or {})
    url = _base_url(base_url or spec["default_url"])

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if provider_key == "ollama" and not url.endswith("/v1"):
                resp = await client.get(
                    f"{url}/api/tags", headers=_headers("ollama", api_key, extra)
                )
                resp.raise_for_status()
                data = resp.json()
                models = [
                    {"id": m.get("name") or m.get("model"), "name": m.get("name")}
                    for m in data.get("models", [])
                ]
                return [m for m in models if m.get("id")]
            if provider_key == "azure":
                raise ProviderError(
                    "Azure model listing is not supported; deployments are "
                    "configured in the Azure portal and referenced via "
                    "extra.deployment."
                )
            resp = await client.get(
                f"{url}/models", headers=_headers(spec["style"], api_key, extra)
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])
            return [
                {"id": m.get("id"), "name": m.get("id")}
                for m in items
                if m.get("id")
            ]
    except ProviderError:
        raise
    except httpx.ConnectError as exc:
        raise ProviderError(
            f"Cannot connect to {provider_key} at {url}. "
            f"Make sure the server is running. ({exc})"
        )
    except httpx.HTTPStatusError as exc:
        raise ProviderError(
            f"Provider {provider_key} returned HTTP "
            f"{exc.response.status_code} listing models: {exc.response.text[:500]}"
        )
    except Exception as exc:
        raise ProviderError(
            f"Error listing models from {provider_key}: {type(exc).__name__}: {exc}"
        )
