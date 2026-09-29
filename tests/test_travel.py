"""Tests for multi-round travel, in-transit cancel, fog of war, and map data.

Uses a tiny hand-built scenario so travel costs are exact and the tests do
not depend on the production scenario files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mygame.server.engine.rules import resolve_round
from mygame.server.engine.state import (
    PlayerAssignment,
    apply_danger_damage,
    attack_actor,
    cancel_travel,
    compute_legal_actions,
    compute_player_view,
    init_game,
)
from mygame.server.engine.winconditions import check_actor_victory
from mygame.server.scenario.loader import ScenarioLoadError, ScenarioRegistry
from mygame.shared.models import (
    Action,
    ActionType,
    CharacterDef,
    Location,
    Objective,
    ScenarioDef,
    Stats,
    VictoryCondition,
)

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


def _mini_scenario() -> ScenarioDef:
    return ScenarioDef(
        id="mini",
        name="mini",
        setting="",
        intro_text="",
        locations={
            "A": Location(
                id="A", name="A", description="",
                connections={"east": "B"}, travel_costs={"east": 3},
                danger_level=5,
            ),
            "B": Location(
                id="B", name="B", description="",
                connections={"west": "A", "north": "C"}, travel_costs={"west": 3},
            ),
            "C": Location(
                id="C", name="C", description="", connections={"south": "B"},
            ),
        },
        items={},
        characters=[
            CharacterDef(
                id="hero", name="Hero", backstory="", base_stats=Stats(),
                start_location="A", secret_objective="",
                victory_condition=VictoryCondition(kind="survive_rounds", params={"rounds": 99}),
            ),
        ],
    )


def _init(n: int = 1):
    scenario = _mini_scenario()
    players = [PlayerAssignment(f"p{i}", f"P{i}", "hero") for i in range(1, n + 1)]
    return init_game(scenario, players), scenario


def _move(pid: str, direction: str) -> Action:
    return Action(player_id=pid, type=ActionType.MOVE, params={"direction": direction})


def _wait(pid: str) -> Action:
    return Action(player_id=pid, type=ActionType.WAIT)


class TestMultiRoundTravel:
    def test_move_with_cost_enters_transit(self):
        state, scenario = _init()
        a = state.actors["p1"]
        state.round = 1
        events = resolve_round(
            state, scenario, {"p1": _move("p1", "east")}, round_seed=1
        )
        assert a.traveling
        assert a.travel_remaining == 2  # 3 -> 2 after the round-end tick
        assert a.location_id == "A"
        assert any(e.kind == "travel_start" for e in events)

    def test_arrive_after_cost_rounds(self):
        state, scenario = _init()
        a = state.actors["p1"]
        for r in (1, 2, 3):
            state.round = r
            action = _move("p1", "east") if r == 1 else _wait("p1")
            resolve_round(state, scenario, {"p1": action}, round_seed=r)
        assert not a.traveling
        assert a.location_id == "B"

    def test_cost_one_is_immediate(self):
        state, scenario = _init()
        a = state.actors["p1"]
        # B -> C has no travel_cost, so default cost 1
        a.location_id = "B"
        state.round = 1
        resolve_round(state, scenario, {"p1": _move("p1", "north")}, round_seed=1)
        assert not a.traveling
        assert a.location_id == "C"

    def test_cancel_turns_back(self):
        state, scenario = _init()
        a = state.actors["p1"]
        state.round = 1
        resolve_round(state, scenario, {"p1": _move("p1", "east")}, round_seed=1)
        assert a.travel_remaining == 2

        state.round = 2
        events = resolve_round(
            state, scenario,
            {"p1": Action(player_id="p1", type=ActionType.CANCEL)}, round_seed=2,
        )
        assert not a.traveling
        assert a.location_id == "A"
        assert any(e.kind == "travel_cancel" for e in events)

    def test_cancel_with_no_progress_stays(self):
        state, scenario = _init()
        a = state.actors["p1"]
        a.travel_from = "A"
        a.travel_to = "B"
        a.travel_direction = "east"
        a.travel_remaining = 3
        cancel_travel(state, scenario, "p1")
        assert not a.traveling
        assert a.location_id == "A"


class TestTravelFogOfWar:
    def test_traveling_actor_invisible_to_others(self):
        state, scenario = _init(2)
        state.round = 1
        resolve_round(
            state, scenario,
            {"p1": _wait("p1"), "p2": _move("p2", "east")}, round_seed=1,
        )
        view = compute_player_view(state, scenario, "p1")
        assert view.visible_actors == []

    def test_traveling_view_shows_route(self):
        state, scenario = _init(1)
        state.round = 1
        resolve_round(state, scenario, {"p1": _move("p1", "east")}, round_seed=1)
        view = compute_player_view(state, scenario, "p1")
        assert view.travel_remaining > 0
        assert view.travel_to == "B"
        assert view.location_name == "途中"

    def test_traveling_actor_not_attackable(self):
        state, scenario = _init(2)
        state.round = 1
        resolve_round(
            state, scenario,
            {"p1": _wait("p1"), "p2": _move("p2", "east")}, round_seed=1,
        )
        assert attack_actor(state, scenario, "p1", "p2") is None

    def test_traveling_skips_danger(self):
        state, scenario = _init(1)
        state.round = 1
        resolve_round(state, scenario, {"p1": _move("p1", "east")}, round_seed=1)
        assert state.actors["p1"].traveling
        assert apply_danger_damage(state, scenario, "p1") is None


class TestTravelVictoryAndActions:
    def test_reach_location_not_while_traveling(self):
        scenario = _mini_scenario()
        scenario.characters[0].personal_objective = Objective(
            description="",
            trigger=VictoryCondition(kind="reach_location", params={"location": "B"}),
            victory=True,
        )
        state = init_game(scenario, [PlayerAssignment("p1", "P1", "hero")])
        a = state.actors["p1"]

        state.round = 1
        resolve_round(state, scenario, {"p1": _move("p1", "east")}, round_seed=1)
        assert a.traveling
        assert not check_actor_victory(state, scenario, "p1")

        for r in (2, 3):
            state.round = r
            resolve_round(state, scenario, {"p1": _wait("p1")}, round_seed=r)
        assert a.location_id == "B"
        assert check_actor_victory(state, scenario, "p1")

    def test_legal_actions_while_traveling(self):
        state, scenario = _init(1)
        a = state.actors["p1"]
        a.travel_from = "A"
        a.travel_to = "B"
        a.travel_direction = "east"
        a.travel_remaining = 2
        kinds = {la.type for la in compute_legal_actions(state, scenario, "p1")}
        assert kinds == {ActionType.WAIT, ActionType.CANCEL}


class TestScenarioTravelData:
    def test_city_of_ash_travel_costs_and_coords(self):
        scenario = ScenarioRegistry().load_file(SCENARIOS_DIR / "city_of_ash.yaml")
        assert scenario.locations["antique_shop"].travel_costs.get("east") == 3
        assert scenario.locations["abandoned_factory"].travel_costs.get("west") == 3
        assert scenario.locations["city_center"].coord == [0, 0]

    def test_loader_rejects_unknown_travel_direction(self, tmp_path):
        raw = {
            "id": "bad",
            "name": "bad",
            "setting": "",
            "intro_text": "",
            "locations": {
                "A": {
                    "id": "A", "name": "A", "description": "",
                    "connections": {"east": "B"},
                    "travel_costs": {"north": 2},
                },
                "B": {
                    "id": "B", "name": "B", "description": "",
                    "connections": {"west": "A"},
                },
            },
            "items": {},
        }
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ScenarioLoadError):
            ScenarioRegistry().load_file(path)
