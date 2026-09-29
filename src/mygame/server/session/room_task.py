"""Room task — orchestrates a single game within a room.

Wires the GameLoop callbacks to WebSocket broadcasts, handles action
submission from players, and builds template-based narrative from events.
When AI narration is available, this layer will delegate to it; for now
it uses deterministic template narration so the game is fully playable.
"""

from __future__ import annotations

import asyncio
import logging
import time
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

# Base "dramatic weight" per event kind — axis 1 of the three-axis criticality
# scorer. Ordinary rounds (move/search/rest/equip/talk/travel) score ~0 and fall
# back to template narration so a round never blocks on a 7b request for
# mundane events. Axis 2 (character criticality_triggers) and axis 3 (location
# points) add on top; a faction is narrated when its best event clears the bar.
BASE_WEIGHT = {
    "death": 2.0,
    "attack": 1.0,
    "random_event": 1.0,
    "npc_encounter": 1.0,
    "trap": 0.9,
    "ability_used": 0.8,
    "vision": 0.8,
    "hazard": 0.7,
    "event_damage": 0.7,
    "knowledge_gained": 0.7,
    "status_applied": 0.5,
    "status_damage": 0.3,
    "event_heal": 0.3,
    "stat_bonus": 0.3,
    "event_stamina": 0.2,
    "talk": 0.2,
    "craft": 0.2,
    "danger_damage": 0.1,
    "search_found": 0.1,
}
CRITICAL_THRESHOLD = 0.6


class RoomTask:
    def __init__(self, room: Room, round_time: int = 45, ai: AIService | None = None):
        self.room = room
        self.round_time = round_time
        self.ai = ai
        self.state: GameState | None = None
        self.loop: GameLoop | None = None
        self._task: asyncio.Task | None = None
        self._connections: dict[str, Any] = {}
        self._faction_history: dict[str, list[str]] = {}

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

        if self.ai is not None:
            # Pre-generate location flavor in the background; until it lands,
            # ordinary rounds fall back to plain narrative_seed templates.
            asyncio.create_task(self.ai.library.generate(self.room.scenario))

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
            await self._narrate_factions(events)
        else:
            await self._narrate_template(events)

    def _faction_critical(self, faction_id: str, events: list[GameEvent]) -> bool:
        """Whether a faction's round deserves real-time LLM narration.

        Three-axis score: base weight of event kind + character
        criticality_triggers + location-point criticality. Hard triggers
        (near-end round, a dying member) always narrate.
        """
        assert self.state is not None
        scenario = self.room.scenario

        if self.state.round >= scenario.max_rounds - 3:
            return True

        members = [a for a in self.state.actors.values() if a.faction_id == faction_id]
        if any(a.alive and a.stats.hp <= a.stats.max_hp * 0.3 for a in members):
            return True

        best = 0.0
        for ev in events:
            if not self._faction_sees_event(faction_id, ev):
                continue
            score = BASE_WEIGHT.get(ev.kind, 0.0)
            if ev.kind == "danger_damage" and ev.payload.get("killed"):
                score = max(score, BASE_WEIGHT["death"])
            score += self._trigger_weight(faction_id, ev)
            best = max(best, score)
        return best >= CRITICAL_THRESHOLD

    def _faction_sees_event(self, faction_id: str, ev: GameEvent) -> bool:
        if ev.visibility == "public":
            return True
        target = ev.private_to
        if target is None or self.state is None:
            return False
        actor = self.state.actors.get(target)
        return actor is not None and actor.faction_id == faction_id

    def _trigger_weight(self, faction_id: str, ev: GameEvent) -> float:
        assert self.state is not None
        scenario = self.room.scenario
        total = 0.0
        for actor in self.state.actors.values():
            if actor.faction_id != faction_id:
                continue
            char = next(
                (c for c in scenario.characters if c.id == actor.character_id), None
            )
            if char is None:
                continue
            for trig in char.criticality_triggers:
                if self._trigger_matches(trig, ev):
                    total += trig.weight
        return total

    def _trigger_matches(self, trig, ev: GameEvent) -> bool:
        if trig.kind and trig.kind != ev.kind:
            return False
        if trig.item_id and trig.item_id != ev.payload.get("item_id"):
            return False
        if trig.location_id:
            loc = (
                ev.payload.get("location_id")
                or ev.payload.get("location")
                or ev.payload.get("to")
            )
            if ev.actor_ids and self.state is not None:
                actor = self.state.actors.get(ev.actor_ids[0])
                if actor is not None:
                    loc = loc or actor.location_id
            if loc != trig.location_id:
                return False
        return True

    async def _narrate_factions(self, events: list[GameEvent]) -> None:
        assert self.state is not None
        scenario = self.room.scenario
        round_num = self.state.round
        t0 = time.monotonic()

        grouped: dict[str, list[str]] = {}
        for pid, actor in self.state.actors.items():
            if not actor.alive:
                continue
            conn = self._connections.get(pid)
            if conn is None or conn.closed:
                continue
            grouped.setdefault(actor.faction_id, []).append(pid)

        llm_factions: list[str] = []
        template_pids: list[str] = []
        for faction_id, pids in grouped.items():
            if faction_id in scenario.factions and self._faction_critical(faction_id, events):
                llm_factions.append(faction_id)
            else:
                template_pids.extend(pids)

        kinds = sorted({ev.kind for ev in events})
        log.info("round %d: faction narration start critical=%s kinds=%s", round_num, sorted(llm_factions), kinds)

        tasks: dict[str, asyncio.Task] = {}
        for faction_id in llm_factions:
            faction = scenario.factions[faction_id]
            faction_events = [
                ev for ev in events if self._faction_sees_event(faction_id, ev)
            ]
            members = [self.state.actors[p] for p in grouped[faction_id]]
            tasks[faction_id] = asyncio.create_task(
                self.ai.narrate_faction(
                    faction, members, round_num, faction_events, scenario,
                    history=self._faction_history.get(faction_id),
                )
            )

        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            t_done = time.monotonic()
            log.info(
                "round %d: faction narration done factions=%d total=%.2fs",
                round_num, len(tasks), t_done - t0,
            )
            for faction_id, result in zip(tasks.keys(), results, strict=True):
                if isinstance(result, str) and result.strip():
                    self._faction_history.setdefault(faction_id, []).append(result)
                    for pid in grouped[faction_id]:
                        conn = self._connections.get(pid)
                        if conn is not None and not conn.closed:
                            await conn.send("narrative", {
                                "round": round_num,
                                "text": result,
                            })
                else:
                    template_pids.extend(grouped[faction_id])

        if template_pids:
            await self._narrate_template(events, pids=template_pids)

    async def _narrate_template(
        self,
        events: list[GameEvent],
        pids: list[str] | None = None,
    ) -> None:
        if self.state is None:
            return
        round_num = self.state.round
        t0 = time.monotonic()
        if pids is None:
            pids = [pid for pid, a in self.state.actors.items() if a.alive]
        kinds = sorted({ev.kind for ev in events})
        log.info("round %d: template narration players=%d kinds=%s", round_num, len(pids), kinds)

        moved_to: dict[str, str] = {}
        for ev in events:
            if ev.kind == "move" and ev.actor_ids:
                moved_to[ev.actor_ids[0]] = ev.payload.get("to", "")

        for pid in pids:
            conn = self._connections.get(pid)
            if conn is None or conn.closed:
                continue

            parts: list[str] = []
            flavor = self._location_flavor(moved_to.get(pid, ""))
            if flavor:
                parts.append(flavor)
            for ev in events:
                if ev.visibility == "private" and ev.private_to == pid:
                    parts.append(ev.narrative_seed)

            if parts:
                await conn.send("narrative", {
                    "round": round_num,
                    "text": "\n".join(parts),
                })

        log.info("round %d: template narration done total=%.4fs", round_num, time.monotonic() - t0)

    def _location_flavor(self, loc_id: str) -> str | None:
        if not loc_id or self.ai is None:
            return None
        lib = getattr(self.ai, "library", None)
        if lib is None:
            return None
        return lib.location_text(loc_id)

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
                    "secret_objective": (
                        char_def.personal_objective.description
                        if char_def and char_def.personal_objective else ""
                    ),
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
