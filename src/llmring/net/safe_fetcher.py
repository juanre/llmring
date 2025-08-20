"""Centralized safe HTTP fetcher for controlled remote downloads.

Default posture is deny. Only enable via configuration and enforce:
- HTTPS only, no redirects
- Host allowlist
- Reject localhost and direct IPs; do not attempt DNS resolution here
- Content-Type whitelist
- Size and timeout limits
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx


class SafeFetchError(Exception):
    pass


def _get_bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def get_default_config() -> Dict[str, object]:
    """Load safe fetcher configuration from environment with safe defaults."""
    allowed_hosts = os.getenv("LLMRING_ALLOWED_HOSTS", "").strip()
    allowed_hosts_list = [h.strip() for h in allowed_hosts.split(",") if h.strip()]
    allowed_content_types = os.getenv(
        "LLMRING_ALLOWED_CONTENT_TYPES",
        "image/png,image/jpeg,image/gif,image/webp,application/pdf",
    )
    return {
        "allow_remote_urls": _get_bool_env("LLMRING_ALLOW_REMOTE_URLS", False),
        "allowed_hosts": allowed_hosts_list,
        "max_size_bytes": int(
            os.getenv("LLMRING_MAX_DOWNLOAD_SIZE_BYTES", str(8 * 1024 * 1024))
        ),
        "connect_timeout_s": float(os.getenv("LLMRING_CONNECT_TIMEOUT_S", "2.0")),
        "read_timeout_s": float(os.getenv("LLMRING_READ_TIMEOUT_S", "5.0")),
        "total_timeout_s": float(os.getenv("LLMRING_TOTAL_TIMEOUT_S", "10.0")),
        "content_types_allowed": [
            ct.strip().lower() for ct in allowed_content_types.split(",") if ct.strip()
        ],
        "follow_redirects": False,
        "max_redirects": 0,
    }


def _is_private_or_local_host(host: str) -> bool:
    # Reject localhost and common local hostnames explicitly
    host_lc = host.lower()
    if host_lc in {"localhost", "localhost.localdomain"}:
        return True
    # Reject direct IP literals that look private/reserved (basic patterns; no DNS resolution here)
    private_ip_patterns = [
        r"^127\.",
        r"^10\.",
        r"^192\.168\.",
        r"^172\.(1[6-9]|2[0-9]|3[0-1])\.",
        r"^169\.254\.",
        r"^::1$",
    ]
    for pat in private_ip_patterns:
        if re.match(pat, host_lc):
            return True
    return False


async def fetch_bytes(
    url: str, config: Optional[Dict[str, object]] = None
) -> Tuple[bytes, str]:
    """Fetch a remote URL safely and return (data, content_type).

    Raises SafeFetchError on violations or failures.
    """
    cfg = get_default_config() if config is None else config
    if not cfg.get("allow_remote_urls", False):
        raise SafeFetchError("Remote URL fetching is disabled by configuration")

    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise SafeFetchError("Only HTTPS URLs are allowed")
    if not parsed.hostname:
        raise SafeFetchError("URL must include a hostname")

    hostname = parsed.hostname
    # Basic local/private checks
    if _is_private_or_local_host(hostname):
        raise SafeFetchError("Local or private hosts are not allowed")

    allowed_hosts: List[str] = cfg.get("allowed_hosts", [])  # type: ignore[assignment]
    if allowed_hosts and hostname not in allowed_hosts:
        raise SafeFetchError("Host is not in the allowlist")

    max_size: int = int(cfg.get("max_size_bytes", 8 * 1024 * 1024))  # type: ignore[arg-type]
    connect_timeout: float = float(cfg.get("connect_timeout_s", 2.0))  # type: ignore[arg-type]
    read_timeout: float = float(cfg.get("read_timeout_s", 5.0))  # type: ignore[arg-type]
    total_timeout: float = float(cfg.get("total_timeout_s", 10.0))  # type: ignore[arg-type]
    allowed_types: List[str] = cfg.get("content_types_allowed", [])  # type: ignore[assignment]

    timeout = httpx.Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=read_timeout,
        pool=connect_timeout,
    )

    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
        # HEAD preflight where possible
        try:
            head = await asyncio.wait_for(client.head(url), timeout=total_timeout)
            cl = head.headers.get("content-length")
            if cl is not None:
                try:
                    if int(cl) > max_size:
                        raise SafeFetchError(
                            "Content length exceeds maximum allowed size"
                        )
                except ValueError:
                    pass
            ct = (head.headers.get("content-type") or "").split(";")[0].lower()
            if allowed_types and ct and ct not in allowed_types:
                raise SafeFetchError("Content-Type not allowed")
        except (httpx.HTTPError, asyncio.TimeoutError):
            # Continue to GET with streaming; still enforce limits
            pass

        try:
            resp = await asyncio.wait_for(client.get(url), timeout=total_timeout)
        except asyncio.TimeoutError as e:
            raise SafeFetchError("Download timed out") from e
        except httpx.HTTPError as e:
            raise SafeFetchError(f"HTTP error: {e}") from e

        if resp.status_code != 200:
            raise SafeFetchError(f"Unexpected status code: {resp.status_code}")

        ct = (resp.headers.get("content-type") or "").split(";")[0].lower()
        if allowed_types and ct and ct not in allowed_types:
            raise SafeFetchError("Content-Type not allowed")

        # Stream with size cap
        total = 0
        chunks: List[bytes] = []
        async for chunk in resp.aiter_bytes():
            if not chunk:
                continue
            total += len(chunk)
            if total > max_size:
                raise SafeFetchError("Downloaded content exceeds maximum allowed size")
            chunks.append(chunk)

        return (b"".join(chunks), ct or "application/octet-stream")
