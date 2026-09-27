"""Countdown timer display for the decision phase."""

from __future__ import annotations

import time


class CountdownTimer:
    def __init__(self):
        self.deadline_ts: float = 0.0
        self._running: bool = False

    def start(self, deadline_ts: float) -> None:
        self.deadline_ts = deadline_ts
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def remaining(self) -> int:
        if not self._running:
            return 0
        return max(0, int(self.deadline_ts - time.time()))

    @property
    def expired(self) -> bool:
        return self.remaining <= 0

    def format(self) -> str:
        s = self.remaining
        if s <= 0:
            return "0s"
        if s < 60:
            return f"{s}s"
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s"
