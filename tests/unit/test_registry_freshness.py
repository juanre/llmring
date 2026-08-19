"""Registry staleness must be visible and assertable.

A registry published before a model existed cannot price that model, and the
call then records no cost at all. Staleness is therefore the upstream cause of
silent zero-cost runs; it should be checkable rather than inferred.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from llmring.exceptions import RegistryStaleError
from llmring.registry import RegistryClient


def _write_registry(tmp_path, provider, updated_at, version=6):
    d = tmp_path / provider
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": provider,
        "version": version,
        "updated_at": updated_at,
        "models": {
            f"{provider}:m": {
                "provider": provider, "model_name": "m", "display_name": "M",
                "dollars_per_million_tokens_input": 1.0,
                "dollars_per_million_tokens_output": 2.0,
            }
        },
    }
    (d / "models.json").write_text(json.dumps(payload))
    return tmp_path


def _client(root, cache_dir):
    return RegistryClient(registry_url=f"file://{root}", cache_dir=cache_dir)


@pytest.mark.asyncio
async def test_source_info_reports_version_and_age(tmp_path):
    root = _write_registry(
        tmp_path / "reg", "openai",
        (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(), version=9,
    )
    info = await _client(root, tmp_path / "c").get_source_info("openai")
    assert info.version == 9
    assert info.age_days == pytest.approx(30, abs=1)
    assert info.provider == "openai"


@pytest.mark.asyncio
async def test_assert_fresh_passes_when_recent(tmp_path):
    root = _write_registry(
        tmp_path / "reg", "openai",
        (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
    )
    info = await _client(root, tmp_path / "c").assert_fresh("openai", max_age_days=30)
    assert info.age_days < 30


@pytest.mark.asyncio
async def test_assert_fresh_raises_when_stale(tmp_path):
    """The real-world case: the registry sat at 7.5 months old."""
    root = _write_registry(
        tmp_path / "reg", "anthropic",
        (datetime.now(timezone.utc) - timedelta(days=228)).isoformat(),
    )
    with pytest.raises(RegistryStaleError) as exc:
        await _client(root, tmp_path / "c").assert_fresh("anthropic", max_age_days=30)
    assert exc.value.age_days == pytest.approx(228, abs=1)
    assert exc.value.provider == "anthropic"
    assert "refresh the registry" in str(exc.value)


@pytest.mark.asyncio
async def test_missing_timestamp_counts_as_stale(tmp_path):
    """Unverifiable age must not silently pass - that would defeat the check."""
    d = tmp_path / "reg" / "google"
    d.mkdir(parents=True)
    (d / "models.json").write_text(json.dumps({"provider": "google", "models": {}}))
    with pytest.raises(RegistryStaleError):
        await _client(tmp_path / "reg", tmp_path / "c").assert_fresh("google", max_age_days=9999)


@pytest.mark.asyncio
async def test_naive_timestamp_is_treated_as_utc(tmp_path):
    """The published registry writes naive ISO timestamps; they must not crash."""
    naive = (datetime.now(timezone.utc) - timedelta(days=5)).replace(tzinfo=None).isoformat()
    root = _write_registry(tmp_path / "reg", "openai", naive)
    info = await _client(root, tmp_path / "c").get_source_info("openai")
    assert info.age_days == pytest.approx(5, abs=1)


# --------------------------------------------------------------------------
# version-awareness: age alone cannot detect a cache that is a VERSION behind
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_age_check_alone_does_not_catch_a_version_behind_cache(tmp_path):
    """The real-world miss: a payload published hours ago is 'fresh' by age
    while already superseded, and is served for up to 24h. This pins the
    limitation so nobody mistakes assert_fresh's age test for a version test."""
    origin = _write_registry(tmp_path / "reg", "openai",
                             datetime.now(timezone.utc).isoformat(), version=9)
    cache = tmp_path / "cache"
    cache.mkdir()
    # A cache holding v8, written moments ago - recent, but a version behind.
    (cache / "openai_current.json").write_text(json.dumps({
        "provider": "openai", "version": 8,
        "updated_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        "models": {},
    }))
    c = _client(origin, cache)
    stale = await c.get_source_info("openai", force_refresh=False)
    assert stale.version == 8 and stale.from_cache is True
    assert stale.age_days < 1          # passes an age test...
    await c.assert_fresh("openai", max_age_days=30, force_refresh=False)  # ...and passes this


@pytest.mark.asyncio
async def test_assert_fresh_defaults_to_checking_the_origin(tmp_path):
    """Default force_refresh=True is what makes the check version-aware."""
    origin = _write_registry(tmp_path / "reg", "openai",
                             datetime.now(timezone.utc).isoformat(), version=9)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "openai_current.json").write_text(json.dumps({
        "provider": "openai", "version": 8,
        "updated_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        "models": {},
    }))
    c = _client(origin, cache)
    info = await c.assert_fresh("openai", max_age_days=30)   # default force_refresh
    assert info.version == 9
    assert info.from_cache is False


@pytest.mark.asyncio
async def test_validating_heals_the_cache(tmp_path):
    """After a forced check, ordinary reads see the current payload too -
    so validation repairs the staleness rather than merely reporting it."""
    origin = _write_registry(tmp_path / "reg", "openai",
                             datetime.now(timezone.utc).isoformat(), version=9)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "openai_current.json").write_text(json.dumps({
        "provider": "openai", "version": 8,
        "updated_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        "models": {},
    }))
    c = _client(origin, cache)
    await c.assert_fresh("openai", max_age_days=30)
    after = await c.get_source_info("openai", force_refresh=False)
    assert after.version == 9


@pytest.mark.asyncio
async def test_invalidate_cache_removes_the_stored_payload(tmp_path):
    origin = _write_registry(tmp_path / "reg", "openai",
                             datetime.now(timezone.utc).isoformat(), version=9)
    cache = tmp_path / "cache"
    c = _client(origin, cache)
    await c.fetch_current_models("openai")
    assert (cache / "openai_current.json").exists()
    c.invalidate_cache("openai")
    assert not (cache / "openai_current.json").exists()


def test_invalidate_cache_is_safe_when_nothing_is_cached(tmp_path):
    _client(tmp_path / "reg", tmp_path / "cache").invalidate_cache()
