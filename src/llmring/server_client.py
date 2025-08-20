from __future__ import annotations

from typing import Optional, Iterable, Dict, Any
import os
import httpx

from llmring.lockfile import Lockfile


async def push_aliases(
    lockfile: Lockfile,
    *,
    profile: Optional[str] = None,
    server_url: Optional[str] = None,
    project_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> int:
    """Push lockfile aliases to the server using bulk upsert.

    Args:
        lockfile: Loaded lockfile to read aliases from
        profile: Profile to push (defaults to lockfile default)
        server_url: Base URL to the server (e.g., http://localhost:8000)
        project_key: Project-scoped key for X-Project-Key
        client: Optional preconfigured AsyncClient (e.g., ASGI client in tests)

    Returns:
        Number of aliases pushed/updated
    """
    prof = lockfile.get_profile(profile)
    items: list[dict[str, Any]] = [
        {"alias": b.alias, "model": b.model_ref, "metadata": None}
        for b in prof.bindings
    ]

    headers = {
        "X-Project-Key": project_key or os.environ.get("LLMRING_PROJECT_KEY", "")
    }
    if client is None:
        base = server_url or os.environ.get("LLMRING_SERVER_URL") or ""
        if not base:
            raise RuntimeError("LLMRING_SERVER_URL must be set or a client provided")
        async with httpx.AsyncClient(base_url=base, timeout=10.0) as _client:
            resp = await _client.post(
                "/api/v1/aliases/bulk_upsert",
                params={"profile": prof.name},
                json=items,
                headers=headers,
            )
            resp.raise_for_status()
            return int(resp.json().get("updated", 0))

    # Using provided client (assumed base_url configured by caller)
    resp = await client.post(
        "/api/v1/aliases/bulk_upsert",
        params={"profile": prof.name},
        json=items,
        headers=headers,
    )
    resp.raise_for_status()
    return int(resp.json().get("updated", 0))


async def pull_aliases(
    lockfile: Lockfile,
    *,
    profile: Optional[str] = None,
    merge: bool = False,
    server_url: Optional[str] = None,
    project_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> int:
    """Pull aliases from the server into the lockfile.

    Args:
        lockfile: Loaded lockfile to update
        profile: Profile to pull (defaults to lockfile default)
        merge: If False, replace existing bindings for the profile; otherwise merge
        server_url: Base URL to the server
        project_key: Project-scoped key for X-Project-Key
        client: Optional preconfigured AsyncClient (e.g., ASGI client in tests)

    Returns:
        Number of bindings written to the lockfile for the profile
    """
    prof_name = profile or lockfile.default_profile
    headers = {
        "X-Project-Key": project_key or os.environ.get("LLMRING_PROJECT_KEY", "")
    }

    async def _fetch(_client: httpx.AsyncClient) -> list[dict[str, Any]]:
        resp = await _client.get(
            "/api/v1/aliases/",
            params={"profile": prof_name},
            headers=headers,
        )
        resp.raise_for_status()
        return list(resp.json())

    if client is None:
        base = server_url or os.environ.get("LLMRING_SERVER_URL") or ""
        if not base:
            raise RuntimeError("LLMRING_SERVER_URL must be set or a client provided")
        async with httpx.AsyncClient(base_url=base, timeout=10.0) as _client:
            data = await _fetch(_client)
    else:
        data = await _fetch(client)

    # Update lockfile
    prof = lockfile.get_profile(prof_name)
    if not merge:
        # Clear existing bindings for the profile
        for b in list(prof.bindings):
            prof.remove_binding(b.alias)

    for item in data:
        alias = item.get("alias")
        model = item.get("model")
        if not alias or not model:
            continue
        lockfile.set_binding(alias, model, profile=prof_name)

    # Save back to disk if it was loaded from path; caller can also handle saving
    lockfile.save()
    return len(data)


