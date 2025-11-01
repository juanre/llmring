"""Utilities to declaratively apply MCP configuration (servers, tools, resources, prompts).

Intended for both CLI (llmring mcp apply -f manifest.json) and code-first ensure
at app startup. Uses MCPHttpClient to persist state via llmring-server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmring.mcp.http_client import MCPHttpClient


def _to_tool_capability(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool entry to server refresh shape (expects inputSchema)."""
    return {
        "name": obj.get("name"),
        "description": obj.get("description"),
        # Accept either inputSchema (preferred) or input_schema (compat)
        "inputSchema": obj.get("inputSchema") or obj.get("input_schema") or {},
    }


def _to_resource_capability(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize resource entry to server refresh shape (expects mimeType)."""
    return {
        "uri": obj.get("uri"),
        "name": obj.get("name"),
        "description": obj.get("description"),
        # Accept either mimeType (preferred) or mime_type (compat)
        "mimeType": obj.get("mimeType") or obj.get("mime_type"),
    }


def _to_prompt_capability(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize prompt entry to server refresh shape."""
    return {
        "name": obj.get("name"),
        "description": obj.get("description"),
        "arguments": obj.get("arguments") or {},
    }


async def ensure_server_and_capabilities(
    client: MCPHttpClient,
    name: str,
    url: str,
    transport: str = "http",
    tools: Optional[List[Dict[str, Any]]] = None,
    resources: Optional[List[Dict[str, Any]]] = None,
    prompts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Ensure a server exists by name (or URL) and refresh its capabilities.

    Idempotent: if a server with the same name exists, reuse it; otherwise register.
    Then replace tools/resources/prompts with the provided lists.
    """
    servers = await client.list_servers()
    existing = None
    for s in servers or []:
        if s.get("name") == name or s.get("url") == url:
            existing = s
            break

    if existing is None:
        existing = await client.register_server(name=name, url=url, transport_type=transport)

    server_id = existing.get("id")

    tools_caps = [_to_tool_capability(t) for t in (tools or [])]
    resource_caps = [_to_resource_capability(r) for r in (resources or [])]
    prompt_caps = [_to_prompt_capability(p) for p in (prompts or [])]

    await client.refresh_server_capabilities(
        server_id=server_id,
        tools=tools_caps,
        resources=resource_caps,
        prompts=prompt_caps,
    )

    return existing


async def ensure_from_manifest(
    client: MCPHttpClient, manifest: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Ensure all servers and capabilities from a manifest are applied.

    Manifest shape:
    {
      "servers": [
        {
          "name": "calc",
          "url": "http://calculator-mcp:8080",
          "transport": "http",
          "tools": [...],
          "resources": [...],
          "prompts": [...]
        }
      ]
    }
    """
    applied = []
    for srv in manifest.get("servers", []):
        created = await ensure_server_and_capabilities(
            client,
            name=srv["name"],
            url=srv["url"],
            transport=srv.get("transport", "http"),
            tools=srv.get("tools") or [],
            resources=srv.get("resources") or [],
            prompts=srv.get("prompts") or [],
        )
        applied.append(created)
    return applied


def load_manifest(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a JSON object")
    return data
