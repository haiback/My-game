"""Round state machine — the heartbeat of the game.

Manages the phase transitions:
  LOBBY → NARRATIVE → DECISION → RESOLUTION → (loop or ENDED)

This module is async-aware and designed to be driven by the room_task
orchestrator. It does NOT handle networking — it just manages state
transitions and emits callbacks for the orchestrator to handle I/O.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from mygame.shared.models import (
    Action,
    ActionType,
    GameEvent,
    GamePhase,
    GameState,
    ScenarioDef,
)
from mygame.server.engine.rules import resolve_round
from mygame.server.engine.state import compute_player_view
from mygame.server.engine.winconditions import check_victory, settle_objectives


@dataclass
class RoundCallbacks:
    """Callbacks the orchestrator registers to handle I/O."""
    on_narrative: Callable[[str, dict[str, str]], Awaitable[None]] | None = None
    on_round_start: Callable[[int, dict], Awaitable[None]] | None = None
    on_tick: Callable[[int], Awaitable[None]] | None = None
    on_resolution: Callable[[list[GameEvent]], Awaitable[None]] | None = None
    on_game_over: Callable[[list[str]], Awaitable[None]] | None = None


@dataclass
class RoundState:
    """Tracks per-round mutable state."""
    submitted_actions: dict[str, Action] = field(default_factory=dict)
    deadline_ts: float = 0.0
    round_number: int = 0


class GameLoop:
    def __init__(
        self,
        state: GameState,
        scenario: ScenarioDef,
        callbacks: RoundCallbacks | None = None,
        round_time: int = 45,
        narrative_timeout: int = 60,
    ):
        self.state = state
        self.scenario = scenario
        self.callbacks = callbacks or RoundCallbacks()
        self.round_time = round_time
        self.narrative_timeout = narrative_timeout
        self._round = RoundState()
        self._running = False

    async def run(self) -> None:
        self._running = True
        while self._running:
            self._round.round_number = self.state.round + 1
            self.state.round = self._round.round_number

            self.state.phase = GamePhase.DECISION
            await self._decision_phase()

            if not self._running:
                break

            self.state.phase = GamePhase.RESOLUTION
            await self._resolution_phase()

            winners = check_victory(self.state, self.scenario)
            if winners:
                self.state.winner_ids = winners
                self.state.phase = GamePhase.ENDED
                if self.callbacks.on_game_over:
                    await self.callbacks.on_game_over(winners)
                self._running = False
                break

            if self.state.round >= self.scenario.max_rounds:
                self.state.phase = GamePhase.ENDED
                survivors = [
                    pid for pid, a in self.state.actors.items() if a.alive
                ]
                self.state.winner_ids = survivors
                if self.callbacks.on_game_over:
                    await self.callbacks.on_game_over(survivors)
                self._running = False
                break

    def stop(self) -> None:
        self._running = False

    async def submit_action(self, player_id: str, action: Action) -> bool:
        if self.state.phase != GamePhase.DECISION:
            return False
        if player_id not in self.state.actors:
            return False
        if not self.state.actors[player_id].alive:
            return False
        self._round.submitted_actions[player_id] = action
        return True

    async def _decision_phase(self) -> None:
        self._round.submitted_actions.clear()
        self._round.deadline_ts = time.time() + self.round_time

        for pid in self.state.actors:
            view = compute_player_view(self.state, self.scenario, pid)
            if self.callbacks.on_round_start:
                await self.callbacks.on_round_start(self.state.round, {
                    "player_id": pid,
                    "deadline_ts": self._round.deadline_ts,
                    "view": view.model_dump(),
                })

        remaining = self.round_time
        while remaining > 0:
            tick_interval = 1 if remaining <= 10 else 5
            await asyncio.sleep(min(tick_interval, remaining))
            remaining = self._round.deadline_ts - time.time()
            if self.callbacks.on_tick:
                await self.callbacks.on_tick(max(0, int(remaining)))
            if not self._running:
                return

            if all(
                pid in self._round.submitted_actions
                for pid, a in self.state.actors.items()
                if a.alive
            ):
                break

    async def _resolution_phase(self) -> None:
        actions = dict(self._round.submitted_actions)

        for pid, actor in self.state.actors.items():
            if actor.alive and pid not in actions:
                actions[pid] = Action(player_id=pid, type=ActionType.WAIT)

        events = resolve_round(
            self.state, self.scenario, actions,
            round_seed=self.state.round * 1000 + hash(self.state.id) % 1000,
        )
        events.extend(settle_objectives(self.state, self.scenario))

        self.state.event_log.extend(events)

        if self.callbacks.on_resolution:
            await self.callbacks.on_resolution(events)
