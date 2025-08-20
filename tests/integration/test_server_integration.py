from pathlib import Path

import pytest

from llmring.lockfile import Lockfile
from llmring.server_client import pull_aliases, push_aliases


@pytest.mark.asyncio
async def test_aliases_endpoint_seeded(seeded_server, project_headers):
    client = seeded_server

    r = await client.get("/api/v1/aliases/", headers=project_headers)
    assert r.status_code == 200
    data = r.json()
    alias_names = {a["alias"] for a in data}
    assert {"summarizer", "cheap"}.issubset(alias_names)


@pytest.mark.asyncio
async def test_usage_stats_endpoint_seeded(seeded_server, project_headers):
    client = seeded_server

    # Use a wide time window to avoid any timezone/naive datetime issues
    params = {"start_date": "1970-01-01T00:00:00Z", "end_date": "2100-01-01T00:00:00Z"}
    r = await client.get("/api/v1/stats", headers=project_headers, params=params)
    assert r.status_code == 200
    stats = r.json()
    assert "summary" in stats
    # Should have at least one request from seeding
    assert stats["summary"]["total_requests"] >= 1


@pytest.mark.asyncio
async def test_requires_project_key_header(seeded_server):
    client = seeded_server

    r = await client.get("/api/v1/aliases/")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_push_and_pull_aliases_with_client(
    seeded_server, project_headers, tmp_path
):
    client = seeded_server

    # Create local lockfile with a different alias to push
    lf = Lockfile()
    lf.set_binding("new_alias", "openai:gpt-3.5-turbo", profile="default")
    lf_path = tmp_path / "llmring.lock"
    lf.save(lf_path)

    # Push via helper (use provided ASGI client)
    updated = await push_aliases(
        lf,
        profile="default",
        project_key=project_headers["X-Project-Key"],
        client=client,
    )
    assert updated >= 1

    # Pull into a fresh lockfile and ensure aliases are present
    lf2 = Lockfile()
    lf2_path = tmp_path / "llmring2.lock"
    lf2.save(lf2_path)

    pulled = await pull_aliases(
        lf2,
        profile="default",
        merge=False,
        project_key=project_headers["X-Project-Key"],
        client=client,
    )
    assert pulled >= 2  # seeded + new_alias
    aliases = {b.alias: b.model_ref for b in lf2.get_profile("default").bindings}
    assert "summarizer" in aliases and "cheap" in aliases and "new_alias" in aliases
