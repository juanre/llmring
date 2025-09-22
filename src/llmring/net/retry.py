from __future__ import annotations

import asyncio
import os
import random
from typing import Awaitable, Callable, Optional


class RetryError(Exception):
    """Retry wrapper that preserves root-cause messages in __str__."""

    def __init__(self, message: str = None, attempts: int = 0, total_delay: float = 0.0):
        # Use a more informative message by default
        if not message and attempts > 0:
            message = f"Failed after {attempts} retry attempts over {total_delay:.1f}s"
        elif not message:
            message = "Retry failed"
        super().__init__(message)
        self.attempts = attempts
        self.total_delay = total_delay

    def __str__(self) -> str:  # pragma: no cover - trivial accessor
        # Prefer our own message if set
        base = super().__str__()
        if base and base != "unknown error":
            return base
        # Walk the cause/context chain for a meaningful message
        seen = set()

        def _walk(e: BaseException | None) -> str:
            if e is None or id(e) in seen:
                return ""
            seen.add(id(e))
            msg = str(e) or ""
            if msg.strip():
                return msg
            return _walk(getattr(e, "__cause__", None)) or _walk(getattr(e, "__context__", None))

        msg = _walk(self.__cause__) or _walk(self.__context__)
        return msg or "Retry failed"


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def is_retryable_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    retry_terms = [
        "rate limit",
        "quota",
        "timeout",
        "timed out",
        "temporarily",
        "unavailable",
        "service unavailable",
        "connection reset",
        "connection aborted",
        "429",
        "502",
        "503",
        "504",
    ]
    return any(term in msg for term in retry_terms)


async def retry_async(
    op_factory: Callable[[], Awaitable],
    *,
    max_attempts: Optional[int] = None,
    base_delay_s: Optional[float] = None,
    max_delay_s: Optional[float] = None,
    jitter: bool = True,
    is_retryable: Callable[[BaseException], bool] = is_retryable_error,
):
    """Retry an async operation with exponential backoff.

    op_factory should return a fresh awaitable each attempt.
    """
    attempts = max_attempts or _get_int_env("LLMRING_RETRY_ATTEMPTS", 2)
    base = (
        base_delay_s if base_delay_s is not None else _get_float_env("LLMRING_RETRY_BASE_S", 0.25)
    )
    max_delay = (
        max_delay_s if max_delay_s is not None else _get_float_env("LLMRING_RETRY_MAX_S", 2.0)
    )

    last_exc: Optional[BaseException] = None
    total_delay = 0.0
    for attempt in range(1, attempts + 1):
        try:
            return await op_factory()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= attempts or not is_retryable(exc):
                # If not retryable, raise the original exception immediately
                if not is_retryable(exc):
                    raise
                break
            # exponential backoff with jitter
            delay = min(max_delay, base * (2 ** (attempt - 1)))
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            total_delay += delay
            await asyncio.sleep(delay)

    # Preserve the last exception as the cause with better context
    # The RetryError.__str__ will surface the root cause if present.
    raise RetryError(attempts=attempts, total_delay=total_delay) from last_exc
