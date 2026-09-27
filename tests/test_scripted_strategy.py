"""Scripted-strategy full-game simulation for 'city_of_ash'.

Each character gets a deterministic policy that pursues its own victory
condition; the other three players idle (WAIT). This proves that every one
of the four asymmetric win conditions is actually reachable through play —
not just theoretically — while keeping the run fast and reproducible
(no network, no Ollama, seeded RNG).

Policies:
  soldier  — hunt down and attack every other alive player (eliminate_all)
  cultist  — gather ritual_dagger + cursed_relic, return to the cathedral,
             wait for the fog to awaken (custom_flag)
  thief    — gather components, craft escape_key + signal_flare, reach the
             metro station (escape)
  doctor   — stay safe and heal, survive to round 22 (survive_rounds)
"""

from __future__ import annotations

from collections import deque

import pytest

from mygame.server.engine.rules import resolve_round
from mygame.server.engine.state import init_game, PlayerAssignment
from mygame.server.engine.winconditions import check_victory
from mygame.server.scenario.loader import ScenarioRegistry
from mygame.shared.models import Action, ActionType, ScenarioDef

from pathlib import Path

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

PLAYERS = [
    PlayerAssignment("p1", "雷", "soldier"),
    PlayerAssignment("p2", "苏", "cultist"),
    PlayerAssignment("p3", "柯", "thief"),
    PlayerAssignment("p4", "林", "doctor"),
]

CHAR_TO_PID = {"soldier": "p1", "cultist": "p2", "thief": "p3", "doctor": "p4"}


@pytest.fixture(scope="module")
def scenario():
    return ScenarioRegistry().load_file(SCENARIOS_DIR / "city_of_ash.yaml")


# ---------------------------------------------------------------------------
# Pathfinding helpers
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
        cur, d = prev[cur]  # type: ignore[misc]
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
    """If injured, use a consumable heal first; otherwise retreat from a
    genuinely dangerous location to rest. Returns None when healthy."""
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

    # Only flee a location whose danger actually threatens us; resting in a
    # low-danger spot is net-positive and avoids an oscillating retreat.
    if scenario.locations[a.location_id].danger_level >= 3:
        safest = min(scenario.locations.values(), key=lambda l: l.danger_level)
        if a.location_id != safest.id:
            return _move_toward(state, scenario, pid, safest.id)
    return Action(player_id=pid, type=ActionType.REST)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def _idle(state, scenario: ScenarioDef, pid: str) -> Action:
    return Action(player_id=pid, type=ActionType.WAIT)


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

    # Craft phase
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


POLICIES = {
    "soldier": _soldier,
    "cultist": _cultist,
    "thief": _thief,
    "doctor": _doctor,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _run_scripted_game(scenario: ScenarioDef, hero_id: str, seed: int = 1):
    """Run the engine loop with only `hero_id` actively pursuing its goal."""
    state = init_game(scenario, PLAYERS)
    hero_pid = CHAR_TO_PID[hero_id]

    rnd = 0
    while True:
        rnd += 1
        state.round = rnd
        actions = {}
        for pid in state.actors:
            policy = POLICIES[hero_id] if pid == hero_pid else _idle
            actions[pid] = policy(state, scenario, pid)
        resolve_round(state, scenario, actions, round_seed=seed * 1000 + rnd)

        winners = check_victory(state, scenario)
        if hero_pid in winners:
            return state, winners, rnd
        if rnd >= scenario.max_rounds:
            return state, winners, rnd


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScriptedVictory:
    def test_soldier_eliminates_all(self, scenario):
        state, winners, rnd = _run_scripted_game(scenario, "soldier")
        assert "p1" in winners, f"soldier did not win (round {rnd}, winners={winners})"
        assert not any(o.alive for o in state.actors.values() if o.player_id != "p1")

    def test_cultist_completes_ritual(self, scenario):
        state, winners, rnd = _run_scripted_game(scenario, "cultist")
        assert "p2" in winners, f"cultist did not win (round {rnd}, winners={winners})"
        assert "fog_awakened" in state.global_flags

    def test_thief_escapes(self, scenario):
        state, winners, rnd = _run_scripted_game(scenario, "thief")
        assert "p3" in winners, f"thief did not win (round {rnd}, winners={winners})"
        thief = state.actors["p3"]
        assert thief.location_id == "metro_station"

    def test_doctor_survives(self, scenario):
        state, winners, rnd = _run_scripted_game(scenario, "doctor")
        assert "p4" in winners, f"doctor did not win (round {rnd}, winners={winners})"
        assert state.round >= 22
