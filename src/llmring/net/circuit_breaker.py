from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class BreakerState:
    failures: int = 0
    opened_at: float = 0.0
    state: str = "closed"  # closed | open | half_open


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout_s: float = 10.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._states: Dict[str, BreakerState] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        async with self._lock:
            st = self._states.get(key)
            now = time.monotonic()
            if not st:
                return True
            if st.state == "open":
                if now - st.opened_at >= self.reset_timeout_s:
                    st.state = "half_open"
                    return True
                return False
            return True

    async def record_success(self, key: str) -> None:
        async with self._lock:
            st = self._states.get(key)
            if not st:
                return
            st.failures = 0
            st.state = "closed"

    async def record_failure(self, key: str) -> None:
        async with self._lock:
            st = self._states.setdefault(key, BreakerState())
            st.failures += 1
            if st.failures >= self.failure_threshold:
                st.state = "open"
                st.opened_at = time.monotonic()
