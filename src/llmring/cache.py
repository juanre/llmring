"""Caching utilities for LLMRing.

Provides a simple in-memory cache and an optional PostgreSQL-backed async cache.

The DB-backed cache uses the provided LLMDatabase instance to create and
access a small key/value table with TTL. Failures are swallowed so that
the main LLM flow is never interrupted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def compute_cache_key(payload: Dict[str, Any]) -> str:
    """Compute a deterministic cache key from a JSON-serializable payload."""
    try:
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except Exception:
        # Fallback: convert to string
        normalized = str(payload)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class InMemoryCacheBackend:
    """Very simple in-memory cache with TTL support.

    Not safe for multi-process usage. Intended for local/dev use only.
    """

    def __init__(self, max_entries: int = 1000):
        self._store: Dict[str, tuple[float, Dict[str, Any]]] = {}
        self._max_entries = max_entries
        self._lock = asyncio.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        entry = self._store.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if expires_at < now:
            # Expired; delete lazily
            try:
                del self._store[key]
            except KeyError:
                pass
            return None
        return value

    def set(self, key: str, value: Dict[str, Any], ttl_seconds: int) -> None:
        # Evict if too large
        if len(self._store) >= self._max_entries:
            # Remove oldest by expiry (simple heuristic)
            try:
                oldest_key = min(self._store.items(), key=lambda kv: kv[1][0])[0]
                del self._store[oldest_key]
            except Exception:
                self._store.clear()
        self._store[key] = (time.time() + max(1, int(ttl_seconds)), value)


@dataclass
class PgDBAsyncCacheBackend:
    """Async cache stored in PostgreSQL via LLMDatabase.

    Creates a small table in the configured schema on first use:
        CREATE TABLE IF NOT EXISTS {{schema}}.llm_cache (
            cache_key TEXT PRIMARY KEY,
            value JSONB NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

    All operations are best-effort; any error results in a no-op.
    """

    llm_db: Any  # llmring.db.LLMDatabase

    async def _ensure_table(self) -> None:
        try:
            await self.llm_db.initialize()
            query = self.llm_db.db._prepare_query(
                """
                CREATE TABLE IF NOT EXISTS {{schema}}.llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            await self.llm_db.db.execute(query)
        except Exception:
            # Swallow errors; cache is optional
            pass

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            await self._ensure_table()
            query = self.llm_db.db._prepare_query(
                """
                SELECT value FROM {{schema}}.llm_cache
                WHERE cache_key = $1 AND expires_at > NOW()
                """
            )
            row = await self.llm_db.db.fetch_one(query, key)
            if row and row.get("value") is not None:
                value = row["value"]
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except Exception:
                        return None
                return value
            return None
        except Exception:
            return None

    async def set(self, key: str, value: Dict[str, Any], ttl_seconds: int) -> None:
        try:
            await self._ensure_table()
            expires_in = max(1, int(ttl_seconds))
            query = self.llm_db.db._prepare_query(
                """
                INSERT INTO {{schema}}.llm_cache (cache_key, value, expires_at)
                VALUES ($1, $2::jsonb, NOW() + $3 * INTERVAL '1 second')
                ON CONFLICT (cache_key) DO UPDATE SET
                    value = EXCLUDED.value,
                    expires_at = EXCLUDED.expires_at,
                    created_at = NOW()
                """
            )
            await self.llm_db.db.execute(query, key, json.dumps(value), expires_in)
        except Exception:
            # Swallow errors; cache is optional
            pass
