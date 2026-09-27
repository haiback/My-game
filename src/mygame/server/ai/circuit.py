"""Circuit breaker for AI service calls.

When Ollama is down or erroring, we stop calling it for a while so every
round doesn't stall on timeouts. While open, callers get None and the
engine falls back to template narration / keyword parsing.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class CircuitBreaker:
    """Closed → Open (after `threshold` consecutive failures) → Half-open.

    Half-open lets a single probe through after `probe_interval` seconds;
    success closes the breaker, failure re-opens it.
    """

    def __init__(self, threshold: int = 3, probe_interval: float = 60.0):
        self.threshold = max(1, threshold)
        self.probe_interval = probe_interval
        self._failure_count = 0
        self._opened_at: float | None = None
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def record_failure(self) -> None:
        self._failure_count += 1
        if not self._open and self._failure_count >= self.threshold:
            self._open = True
            self._opened_at = time.monotonic()

    def record_success(self) -> None:
        self._failure_count = 0
        self._open = False
        self._opened_at = None

    def _should_probe(self) -> bool:
        if not self._open:
            return True
        if self._opened_at is None:
            return False
        return time.monotonic() - self._opened_at >= self.probe_interval

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T | None:
        """Run `fn` if the breaker allows; return None when open.

        A single probe is allowed through once the interval has elapsed.
        The probe result determines whether the breaker closes again.
        """
        if self._open and not self._should_probe():
            return None
        try:
            result = await fn()
        except Exception:
            self.record_failure()
            return None
        self.record_success()
        return result
