"""Run a full 4-player AI-narrated game of 'city_of_ash' and export every
message each player receives to the desktop as a Markdown transcript.

Drives the real RoomTask orchestrator + real Ollama narration in-process
(no interactive terminals), reusing the scripted policies from
tests/test_scripted_strategy.py. Each character pursues its own win
condition; the engine is the source of truth, AI only narrates.

Usage:  python scripts/play_full_game.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from mygame.server.ai.manager import AIService
from mygame.server.engine.state import init_game, PlayerAssignment
from mygame.server.scenario.loader import ScenarioRegistry
from mygame.server.session.lobby import Room
from mygame.server.session.room_task import RoomTask
from mygame.shared.models import Action, ActionType, GamePhase, GameState, ScenarioDef

SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
DESKTOP = Path.home() / "Desktop"
LOG_DIR = PROJECT_ROOT / "logs"

PLAYERS = [
    ("p1", "雷", "soldier"),
    ("p2", "苏", "cultist"),
    ("p3", "柯", "thief"),
    ("p4", "林", "doctor"),
    ("p5", "守", "warden"),
]
CHAR_TO_PID = {char: pid for pid, _, char in PLAYERS}
PID_TO_NAME = {pid: name for pid, name, _ in PLAYERS}
PID_TO_CHAR = {pid: char for pid, _, char in PLAYERS}


class CaptureConn:
    """Fake connection that records every (type, payload) sent to a player."""

    def __init__(self, pid: str):
        self.pid = pid
        self.closed = False
        self.messages: list[tuple[str, dict]] = []

    async def send(self, msg_type: str, payload: dict | None = None):
        self.messages.append((msg_type, dict(payload or {})))


# ---------------------------------------------------------------------------
# Scripted policies (mirror of tests/test_scripted_strategy.py)
# ---------------------------------------------------------------------------

def _bfs_dist(scenario: ScenarioDef, start: str, goal: str) -> int | None:
    q = deque([(start, 0)])
    seen = {start}
    while q:
        loc, d = q.popleft()
        if loc == goal:
            return d
        for nxt in scenario.locations[loc].connections.values():
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    return None


def _first_step(scenario: ScenarioDef, start: str, goal: str) -> str | None:
    prev: dict[str, tuple[str, str] | None] = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for d, nxt in scenario.locations[cur].connections.items():
            if nxt not in prev:
                prev[nxt] = (cur, d)
                q.append(nxt)
    if goal not in prev:
        return None
    path: list[str] = []
    cur = goal
    while prev[cur] is not None:
        cur, d = prev[cur]
        path.append(d)
    path.reverse()
    return path[0] if path else None


def _recipe_index(scenario: ScenarioDef, output_id: str) -> int:
    for i, r in enumerate(scenario.recipes):
        if r.output_item_id == output_id:
            return i
    return -1


def _nearest_drop(state, scenario: ScenarioDef, start: str, item_ids: list[str]) -> str | None:
    q = deque([start])
    seen = {start}
    while q:
        loc = q.popleft()
        rt = state.location_state[loc]
        if any(rt.remaining_loot.get(it, 0) > 0 for it in item_ids):
            return loc
        for nxt in scenario.locations[loc].connections.values():
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return None


def _move_toward(state, scenario: ScenarioDef, pid: str, goal: str) -> Action:
    a = state.actors[pid]
    d = _first_step(scenario, a.location_id, goal)
    return Action(player_id=pid, type=ActionType.MOVE, params={"direction": d}) if d else Action(player_id=pid, type=ActionType.WAIT)


def _heal_or_retreat(state, scenario: ScenarioDef, pid: str, threshold: int) -> Action | None:
    a = state.actors[pid]
    if a.stats.hp >= threshold:
        return None

    heals = [
        (iid, it.stats.get("heal", 0))
        for iid, cnt in a.inventory.items()
        if cnt > 0 and (it := scenario.items[iid]).kind == "consumable" and it.stats.get("heal", 0) > 0
    ]
    if heals:
        best = max(heals, key=lambda x: x[1])[0]
        return Action(player_id=pid, type=ActionType.USE, params={"item_id": best})

    if scenario.locations[a.location_id].danger_level >= 3:
        safest = min(scenario.locations.values(), key=lambda l: l.danger_level)
        if a.location_id != safest.id:
            return _move_toward(state, scenario, pid, safest.id)
    return Action(player_id=pid, type=ActionType.REST)


def _soldier(state, scenario: ScenarioDef, pid: str) -> Action:
    a = state.actors[pid]
    heal = _heal_or_retreat(state, scenario, pid, 40)
    if heal is not None:
        return heal

    others = [o for o in state.actors.values() if o.player_id != pid and o.alive]
    same = [o for o in others if o.location_id == a.location_id]
    if same:
        return Action(player_id=pid, type=ActionType.ATTACK, params={"target": same[0].player_id})
    target = min(others, key=lambda o: _bfs_dist(scenario, a.location_id, o.location_id) or 10**9)
    return _move_toward(state, scenario, pid, target.location_id)


def _cultist(state, scenario: ScenarioDef, pid: str) -> Action:
    a = state.actors[pid]
    heal = _heal_or_retreat(state, scenario, pid, 60)
    if heal is not None:
        return heal

    needed = ["ritual_dagger", "cursed_relic"]
    missing = [it for it in needed if a.inventory.get(it, 0) < 1]

    if missing:
        loc = _nearest_drop(state, scenario, a.location_id, missing)
        if loc is None:
            return Action(player_id=pid, type=ActionType.WAIT)
        if a.location_id == loc:
            return Action(player_id=pid, type=ActionType.SEARCH)
        return _move_toward(state, scenario, pid, loc)

    if a.location_id != "cathedral":
        return _move_toward(state, scenario, pid, "cathedral")
    return Action(player_id=pid, type=ActionType.REST)


def _thief(state, scenario: ScenarioDef, pid: str) -> Action:
    a = state.actors[pid]
    inv = a.inventory

    if inv.get("escape_key", 0) == 0 and inv.get("antique_fragment", 0) >= 2:
        return Action(player_id=pid, type=ActionType.CRAFT, params={"recipe_index": _recipe_index(scenario, "escape_key")})
    if inv.get("signal_flare", 0) == 0 and all(
        inv.get(it, 0) >= 1 for it in ("battery", "transmitter_parts", "metal_scrap")
    ):
        return Action(player_id=pid, type=ActionType.CRAFT, params={"recipe_index": _recipe_index(scenario, "signal_flare")})

    if inv.get("escape_key", 0) >= 1 and inv.get("signal_flare", 0) >= 1:
        if a.location_id != "metro_station":
            return _move_toward(state, scenario, pid, "metro_station")
        return Action(player_id=pid, type=ActionType.WAIT)

    heal = _heal_or_retreat(state, scenario, pid, 45)
    if heal is not None:
        return heal

    needed = {"antique_fragment": 2, "battery": 1, "transmitter_parts": 1, "metal_scrap": 1}
    missing = [it for it, n in needed.items() if inv.get(it, 0) < n]
    loc = _nearest_drop(state, scenario, a.location_id, missing)
    if loc is None:
        return Action(player_id=pid, type=ActionType.WAIT)
    if a.location_id == loc:
        return Action(player_id=pid, type=ActionType.SEARCH)
    return _move_toward(state, scenario, pid, loc)


def _doctor(state, scenario: ScenarioDef, pid: str) -> Action:
    a = state.actors[pid]
    if a.stats.hp < 70:
        if a.ability_cooldowns.get("emergency_triage", 0) == 0 and a.stats.stamina >= 14:
            return Action(player_id=pid, type=ActionType.SPECIAL, params={"ability_id": "emergency_triage"})
        heal = _heal_or_retreat(state, scenario, pid, 70)
        if heal is not None:
            return heal
    if a.location_id != "city_center":
        return _move_toward(state, scenario, pid, "city_center")
    return Action(player_id=pid, type=ActionType.REST)


def _warden(state, scenario: ScenarioDef, pid: str) -> Action:
    # NPC faction: hold position and endure; rest when hurt, otherwise wait.
    a = state.actors[pid]
    if a.stats.hp < 40:
        return Action(player_id=pid, type=ActionType.REST)
    return Action(player_id=pid, type=ActionType.WAIT)


POLICIES = {
    "soldier": _soldier,
    "cultist": _cultist,
    "thief": _thief,
    "doctor": _doctor,
    "warden": _warden,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "mygame.log", encoding="utf-8"),
        ],
    )


def _load_ai() -> AIService:
    cfg_path = PROJECT_ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    ollama = dict(cfg.get("ollama", {}))
    ollama["timeout_seconds"] = 180
    return AIService(ollama)


async def _play() -> tuple[list[CaptureConn], GameState, list[str], int]:
    registry = ScenarioRegistry()
    scenario = registry.load_file(SCENARIOS_DIR / "city_of_ash.yaml")

    conns = {pid: CaptureConn(pid) for pid, _, _ in PLAYERS}
    room = Room("GAME1", scenario, "p1", conns["p1"], "雷")
    for pid, name, _ in PLAYERS[1:]:
        room.add_player(pid, name, conns[pid])
    for pid, _, char in PLAYERS:
        room.select_character(pid, char)
        room.set_ready(pid)

    ai = _load_ai()
    game = RoomTask(room, round_time=5, ai=ai)
    await game.start()

    last_round = 0
    started = time.time()
    try:
        while True:
            while True:
                st = game.state
                if st is None:
                    await asyncio.sleep(0.1)
                    continue
                if st.phase == GamePhase.ENDED:
                    break
                if st.phase == GamePhase.DECISION and st.round > last_round:
                    break
                await asyncio.sleep(0.1)

            st = game.state
            assert st is not None
            if st.phase == GamePhase.ENDED:
                break

            last_round = st.round
            print(f"[round {st.round}] submitting actions... (t={time.time()-started:.0f}s)", flush=True)
            for pid, _, char in PLAYERS:
                actor = st.actors.get(pid)
                if actor and actor.alive:
                    if actor.traveling:
                        action = Action(player_id=pid, type=ActionType.WAIT)
                    else:
                        action = POLICIES[char](st, scenario, pid)
                    await game.submit_action(pid, action)
    finally:
        await game.stop()
        await ai.close()

    st = game.state
    assert st is not None
    return [conns[p] for p, _, _ in PLAYERS], st, st.winner_ids, st.round


# ---------------------------------------------------------------------------
# Transcript export
# ---------------------------------------------------------------------------

def _state_line(state: GameState, pid: str) -> str:
    a = state.actors[pid]
    return f"位置={a.location_id} HP={a.stats.hp}/{a.stats.max_hp} 体力={a.stats.stamina} 武器={a.equipped_weapon or '-'} 护甲={a.equipped_armor or '-'}"


def build_transcript(conns, state, winner_ids, scenario) -> str:
    char_by_id = {c.id: c for c in scenario.characters}
    lines: list[str] = []

    winner_names = [PID_TO_NAME[p] for p in winner_ids]
    lines.append(f"# 终焉之城 · 四人对局实录")
    lines.append("")
    lines.append(f"- 剧本：{scenario.name}（{scenario.id}）")
    lines.append(f"- 结局：第 {state.round} 回合结束")
    lines.append(f"- 胜者：{', '.join(winner_names) if winner_names else '无'}")
    lines.append(f"- 玩家：{', '.join(f'{name}({char_by_id[c].name})' for _, name, c in PLAYERS)}")
    lines.append("")

    for conn in conns:
        pid = conn.pid
        name = PID_TO_NAME[pid]
        char = char_by_id[PID_TO_CHAR[pid]]

        lines.append(f"## 玩家：{name}（{char.name}）")
        lines.append("")

        intro = ""
        objective = ""
        backstory = ""
        rounds: dict[int, dict[str, list[str]]] = {}
        epilogue = ""

        for mtype, payload in conn.messages:
            if mtype == "game_start":
                intro = payload.get("scenario_intro", "")
                yc = payload.get("your_character", {})
                backstory = yc.get("backstory", "")
                objective = yc.get("secret_objective", "")
            elif mtype == "public_log":
                r = payload.get("round", 0)
                rounds.setdefault(r, {"public": [], "narrative": []})["public"].extend(
                    payload.get("lines", [])
                )
            elif mtype == "narrative":
                r = payload.get("round", 0)
                rounds.setdefault(r, {"public": [], "narrative": []})["narrative"].append(
                    payload.get("text", "")
                )
            elif mtype == "game_over":
                epilogue = payload.get("epilogue", "")

        if intro:
            lines.append("### 开局")
            lines.append("")
            lines.append("> " + intro.replace("\n", "\n> "))
            lines.append("")
            lines.append(f"**角色背景**：{backstory}")
            lines.append("")
            lines.append(f"**秘密目标**：{objective}")
            lines.append("")

        lines.append("### 逐回合")
        lines.append("")
        for r in sorted(rounds):
            entry = rounds[r]
            lines.append(f"#### 第 {r} 回合")
            lines.append("")
            if entry["public"]:
                lines.append("**公开动态**（所有玩家可见）：")
                lines.append("")
                for pub in entry["public"]:
                    lines.append(f"- {pub}")
                lines.append("")
            for nar in entry["narrative"]:
                lines.append("**你的叙事**：")
                lines.append("")
                lines.append(nar)
                lines.append("")

        if epilogue:
            lines.append("### 结局")
            lines.append("")
            lines.append(epilogue)
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _narration_stats(conns) -> str:
    out = []
    for conn in conns:
        real = 0
        template = 0
        for mtype, payload in conn.messages:
            if mtype == "narrative":
                if _has_cjk(payload.get("text", "")):
                    real += 1
                else:
                    template += 1
        out.append(f"  {PID_TO_NAME[conn.pid]}: 中文叙事 {real} 回 / 模板 {template} 回")
    return "\n".join(out)


async def main() -> None:
    _setup_logging()
    print("loading scenario & starting game...", flush=True)
    conns, state, winners, _round = await _play()
    registry = ScenarioRegistry()
    scenario = registry.load_file(SCENARIOS_DIR / "city_of_ash.yaml")

    text = build_transcript(conns, state, winners, scenario)
    out = DESKTOP / "mygame_终焉之城_对局实录.md"
    out.write_text(text, encoding="utf-8")
    print(f"\nDone. Transcript written to: {out}", flush=True)
    print(f"  rounds={state.round} winners={winners}", flush=True)
    print("Narration quality:", flush=True)
    print(_narration_stats(conns), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
