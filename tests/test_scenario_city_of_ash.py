"""Tests for the 'city_of_ash' scenario (终焉之城).

Validates the scenario loads, its references are sound, the map is fully
connected, and every character's victory condition is reachable from the
data (items droppable/craftable, locations exist, fog flag is wired to
the cultist's custom_flag condition).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mygame.server.engine.state import init_game, PlayerAssignment
from mygame.server.engine.winconditions import check_actor_victory
from mygame.server.scenario.loader import ScenarioRegistry

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"
FILE = SCENARIOS_DIR / "city_of_ash.yaml"


@pytest.fixture(scope="module")
def scenario():
    return ScenarioRegistry().load_file(FILE)


def _reachable(scenario, start: str) -> set[str]:
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for target in scenario.locations[cur].connections.values():
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def _droppable(scenario) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for loc in scenario.locations.values():
        for drop in loc.searchable_items:
            out.setdefault(drop.item_id, []).append(loc.id)
    return out


class TestCityOfAsh:
    def test_loads(self, scenario):
        assert scenario.id == "city_of_ash"
        assert scenario.player_range == (2, 4)
        assert scenario.max_rounds == 26

    def test_structure_counts(self, scenario):
        assert len(scenario.locations) == 14
        assert len(scenario.items) == 26
        assert len(scenario.recipes) == 6
        assert len(scenario.knowledge_sources) == 4
        assert len(scenario.characters) == 4
        assert len(scenario.random_events) == 7

    def test_map_fully_connected(self, scenario):
        all_locs = set(scenario.locations)
        for char in scenario.characters:
            assert _reachable(scenario, char.start_location) == all_locs, (
                f"Character '{char.id}' cannot reach all locations"
            )

    def test_distinct_victory_kinds(self, scenario):
        kinds = {c.id: c.victory_condition.kind for c in scenario.characters}
        assert kinds["soldier"] == "eliminate_all"
        assert kinds["cultist"] == "custom_flag"
        assert kinds["thief"] == "escape"
        assert kinds["doctor"] == "survive_rounds"

    def test_every_item_obtainable(self, scenario):
        droppable = _droppable(scenario)
        craftable = {r.output_item_id for r in scenario.recipes}
        for item_id in scenario.items:
            assert item_id in droppable or item_id in craftable, (
                f"Item '{item_id}' is neither droppable nor craftable"
            )

    def test_recipe_inputs_obtainable(self, scenario):
        droppable = _droppable(scenario)
        craftable = {r.output_item_id for r in scenario.recipes}
        for recipe in scenario.recipes:
            for input_id in recipe.inputs:
                assert input_id in droppable or input_id in craftable, (
                    f"Recipe '{recipe.output_item_id}' input '{input_id}' "
                    "is unobtainable"
                )

    def test_fog_awakened_flag_wired(self, scenario):
        fog = next(e for e in scenario.random_events if e.id == "fog_awakening")
        flag_effects = [
            eff for eff in fog.effects if eff.get("type") == "set_global_flag"
        ]
        assert flag_effects, "fog_awakening must set a global flag"
        assert "fog_awakened" in {eff.get("flag") for eff in flag_effects}


class TestCityOfAshVictory:
    def _state(self, scenario):
        return init_game(scenario, [
            PlayerAssignment("p1", "雷", "soldier"),
            PlayerAssignment("p2", "苏", "cultist"),
            PlayerAssignment("p3", "柯", "thief"),
            PlayerAssignment("p4", "林", "doctor"),
        ])

    def test_cultist_victory(self, scenario):
        state = self._state(scenario)
        state.round = 18
        state.global_flags.add("fog_awakened")
        cultist = state.actors["p2"]
        cultist.inventory["ritual_dagger"] = 1
        cultist.inventory["cursed_relic"] = 1
        cultist.location_id = "cathedral"
        assert check_actor_victory(state, scenario, "p2")

    def test_cultist_requires_flag(self, scenario):
        state = self._state(scenario)
        state.round = 18
        cultist = state.actors["p2"]
        cultist.inventory["ritual_dagger"] = 1
        cultist.inventory["cursed_relic"] = 1
        cultist.location_id = "cathedral"
        assert not check_actor_victory(state, scenario, "p2")

    def test_thief_victory(self, scenario):
        state = self._state(scenario)
        thief = state.actors["p3"]
        thief.inventory["escape_key"] = 1
        thief.inventory["signal_flare"] = 1
        thief.location_id = "metro_station"
        assert check_actor_victory(state, scenario, "p3")

    def test_doctor_victory(self, scenario):
        state = self._state(scenario)
        state.round = 22
        assert check_actor_victory(state, scenario, "p4")
        state.round = 21
        assert not check_actor_victory(state, scenario, "p4")
