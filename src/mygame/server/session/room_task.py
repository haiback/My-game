"""Room task — orchestrates a single game within a room.

Wires the GameLoop callbacks to WebSocket broadcasts, handles action
submission from players, and builds template-based narrative from events.
When AI narration is available, this layer will delegate to it; for now
it uses deterministic template narration so the game is fully playable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mygame.server.ai.manager import AIService
from mygame.server.engine.loop import GameLoop, RoundCallbacks
from mygame.server.engine.state import (
    PlayerAssignment,
    compute_player_view,
    init_game,
)
from mygame.server.session.lobby import Room
from mygame.shared.models import (
    Action,
    ActionType,
    GameEvent,
    GamePhase,
    GameState,
    ScenarioDef,
)

log = logging.getLogger(__name__)


class RoomTask:
    def __init__(self, room: Room, round_time: int = 45, ai: AIService | None = None):
        self.room = room
        self.round_time = round_time
        self.ai = ai
        self.state: GameState | None = None
        self.loop: GameLoop | None = None
        self._task: asyncio.Task | None = None
        self._connections: dict[str, Any] = {}
        self._narration_history: dict[str, list[str]] = {}

    async def start(self) -> None:
        assignments = [
            PlayerAssignment(player_id=pid, player_name=name, character_id=cid)
            for pid, name, cid in self.room.get_assignments()
        ]

        self.state = init_game(self.room.scenario, assignments)
        self.state.phase = GamePhase.DECISION

        self._connections = {
            pid: slot.connection for pid, slot in self.room.players.items()
        }

        callbacks = RoundCallbacks(
            on_round_start=self._on_round_start,
            on_tick=self._on_tick,
            on_resolution=self._on_resolution,
            on_game_over=self._on_game_over,
        )

        self.loop = GameLoop(
            state=self.state,
            scenario=self.room.scenario,
            callbacks=callbacks,
            round_time=self.round_time,
        )

        await self._broadcast_game_start()

        self._task = asyncio.create_task(self.loop.run())
        log.info("Game started in room %s with %d players", self.room.code, len(assignments))

    async def submit_action(self, player_id: str, action: Action) -> bool:
        if self.loop is None:
            return False
        return await self.loop.submit_action(player_id, action)

    async def stop(self) -> None:
        if self.loop:
            self.loop.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # -- Callbacks from GameLoop --

    async def _on_round_start(self, round_number: int, context: dict) -> None:
        player_id = context.get("player_id", "")
        view_data = context.get("view", {})
        deadline = context.get("deadline_ts", 0.0)

        conn = self._connections.get(player_id)
        if conn and not conn.closed:
            await conn.send("round_start", {
                "round": round_number,
                "phase": "decision",
                "deadline_ts": deadline,
                "your_state": view_data,
            })

    async def _on_tick(self, seconds_left: int) -> None:
        for pid, conn in self._connections.items():
            if conn.closed:
                continue
            await conn.send("tick", {"seconds_left": seconds_left})

    async def _on_resolution(self, events: list[GameEvent]) -> None:
        public_lines = []
        for ev in events:
            if ev.visibility == "public":
                public_lines.append(ev.narrative_seed)

        if public_lines:
            for pid, conn in self._connections.items():
                if conn.closed:
                    continue
                await conn.send("public_log", {
                    "round": self.state.round if self.state else 0,
                    "lines": public_lines,
                })

        if self.ai and self.state:
            await self._narrate_with_ai(events)
        else:
            await self._narrate_template(events)

    async def _narrate_with_ai(self, events: list[GameEvent]) -> None:
        assert self.state is not None
        scenario = self.room.scenario
        round_num = self.state.round

        # Shared world-side summary of public events, produced once per round.
        public_events = [ev for ev in events if ev.visibility == "public"]
        world_summary = await self.ai.narrate_world(round_num, public_events)

        upcoming_threat = self._upcoming_threat(round_num)

        tasks: dict[str, asyncio.Task] = {}
        for pid, actor in self.state.actors.items():
            conn = self._connections.get(pid)
            if conn is None or conn.closed:
                continue
            per_player = [
                ev for ev in events
                if ev.visibility == "public" or ev.private_to == pid
            ]
            if not per_player:
                continue
            tasks[pid] = asyncio.create_task(
                self.ai.narrate_round(
                    actor, round_num, per_player, scenario,
                    world_summary=world_summary,
                    others_summary=self._others_summary(pid),
                    upcoming_threat=upcoming_threat,
                    history=self._narration_history.get(pid),
                )
            )

        if not tasks:
            return

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for pid, result in zip(tasks.keys(), results, strict=True):
            conn = self._connections.get(pid)
            if conn is None or conn.closed:
                continue
            if isinstance(result, str) and result.strip():
                self._narration_history.setdefault(pid, []).append(result)
                await conn.send("narrative", {
                    "round": round_num,
                    "text": result,
                })
            else:
                lines = [
                    ev.narrative_seed for ev in events
                    if ev.visibility == "private" and ev.private_to == pid
                ]
                if lines:
                    await conn.send("narrative", {
                        "round": round_num,
                        "text": "\n".join(lines),
                    })

    def _others_summary(self, viewer_id: str) -> str | None:
        """Fog-of-war-respecting summary of other players for one viewer.

        Only reveals players in the *same location* (the one thing the viewer
        can actually see). Players elsewhere are omitted so narration never
        leaks a hidden position.
        """
        assert self.state is not None
        scenario = self.room.scenario
        viewer_loc = self.state.actors[viewer_id].location_id
        lines: list[str] = []
        for pid, actor in self.state.actors.items():
            if pid == viewer_id:
                continue
            if actor.location_id != viewer_loc:
                continue
            char = next(
                (c for c in scenario.characters if c.id == actor.character_id), None
            )
            name = char.name if char else "?"
            lines.append(f"{actor.player_name}({name}) 与你同处一地, 存活")
        return "\n".join(lines) if lines else None

    def _upcoming_threat(self, round_num: int) -> str | None:
        """Next scheduled random event, framed as a vague premonition (no
        timeline, so the prose stays a hint rather than a confirmed fact)."""
        assert self.state is not None
        scenario = self.room.scenario
        upcoming = []
        for ev in scenario.random_events:
            if ev.round_trigger is None:
                continue
            if ev.round_trigger <= round_num:
                continue
            if self.state.fired_events.get(ev.id, 0) >= ev.max_fires:
                continue
            upcoming.append(ev)
        if not upcoming:
            return None
        nxt = min(upcoming, key=lambda e: e.round_trigger)
        return f"{nxt.name}——{nxt.description}"

    async def _narrate_template(self, events: list[GameEvent]) -> None:
        for ev in events:
            if ev.visibility == "private" and ev.private_to:
                conn = self._connections.get(ev.private_to)
                if conn and not conn.closed:
                    await conn.send("narrative", {
                        "round": self.state.round if self.state else 0,
                        "text": ev.narrative_seed,
                    })

    async def _on_game_over(self, winner_ids: list[str]) -> None:
        winner_names = []
        if self.state:
            for pid in winner_ids:
                actor = self.state.actors.get(pid)
                if actor:
                    winner_names.append(actor.player_name)

        epilogue = self._build_epilogue(winner_ids)

        for pid, conn in self._connections.items():
            if conn.closed:
                continue
            await conn.send("game_over", {
                "winner_ids": winner_ids,
                "epilogue": epilogue,
                "final_log": [e.model_dump() for e in (self.state.event_log if self.state else [])[-20:]],
            })

    # -- Helpers --

    async def _broadcast_game_start(self) -> None:
        if self.state is None:
            return
        for pid, conn in self._connections.items():
            if conn.closed:
                continue
            actor = self.state.actors.get(pid)
            if not actor:
                continue
            char_def = None
            for c in self.room.scenario.characters:
                if c.id == actor.character_id:
                    char_def = c
                    break
            await conn.send("game_start", {
                "scenario_intro": self.room.scenario.intro_text,
                "your_character": {
                    "id": char_def.id if char_def else "",
                    "name": char_def.name if char_def else "",
                    "backstory": char_def.backstory if char_def else "",
                    "secret_objective": char_def.secret_objective if char_def else "",
                },
            })

    def _build_epilogue(self, winner_ids: list[str]) -> str:
        if not self.state:
            return ""
        lines = []
        for pid in winner_ids:
            actor = self.state.actors.get(pid)
            if actor:
                lines.append(f"{actor.player_name} achieved victory.")
        if not lines:
            lines.append("No survivors.")
        return "\n".join(lines)


def parse_action_from_payload(player_id: str, data: dict) -> Action | None:
    if data.get("action"):
        act = data["action"]
        try:
            return Action(
                player_id=player_id,
                type=ActionType(act["type"]),
                params=act.get("params", {}),
                raw_text=data.get("free_text"),
            )
        except (KeyError, ValueError):
            return None

    free_text = data.get("free_text", "")
    if not free_text:
        return None

    return _parse_free_text(player_id, free_text)


def _parse_free_text(player_id: str, text: str) -> Action | None:
    lower = text.lower().strip()

    if lower in ("wait", "w", "do nothing", "nothing"):
        return Action(player_id=player_id, type=ActionType.WAIT, raw_text=text)

    if lower in ("rest", "r", "recover"):
        return Action(player_id=player_id, type=ActionType.REST, raw_text=text)

    if lower in ("search", "s", "look around"):
        return Action(player_id=player_id, type=ActionType.SEARCH, raw_text=text, parse_confidence=0.8)

    for prefix in ("move ", "go ", "walk ", "run ", "head "):
        if lower.startswith(prefix):
            direction = text.strip()[len(prefix):].strip()
            if direction:
                return Action(
                    player_id=player_id,
                    type=ActionType.MOVE,
                    params={"direction": direction},
                    raw_text=text,
                    parse_confidence=0.7,
                )

    for prefix in ("attack ", "hit ", "strike ", "fight "):
        if lower.startswith(prefix):
            target = text.strip()[len(prefix):].strip()
            if target:
                return Action(
                    player_id=player_id,
                    type=ActionType.ATTACK,
                    params={"target": target},
                    raw_text=text,
                    parse_confidence=0.7,
                )

    for prefix in ("use ", "consume ", "drink ", "eat "):
        if lower.startswith(prefix):
            item = text.strip()[len(prefix):].strip()
            if item:
                return Action(
                    player_id=player_id,
                    type=ActionType.USE,
                    params={"item_id": item},
                    raw_text=text,
                    parse_confidence=0.7,
                )

    if lower.startswith("craft "):
        try:
            idx = int(text.strip()[6:].strip())
            return Action(
                player_id=player_id,
                type=ActionType.CRAFT,
                params={"recipe_index": idx},
                raw_text=text,
            )
        except ValueError:
            return None

    return None
