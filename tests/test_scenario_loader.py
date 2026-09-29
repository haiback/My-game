"""Tests for scenario loading and validation."""

from pathlib import Path

import pytest

from mygame.server.scenario.loader import ScenarioLoadError, ScenarioRegistry
from mygame.shared.models import (
    CharacterDef,
    Item,
    Location,
    Recipe,
    ScenarioDef,
    VictoryCondition,
)

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


@pytest.fixture
def registry():
    return ScenarioRegistry()


@pytest.fixture
def island_scenario(registry):
    return registry.load_file(SCENARIOS_DIR / "island_of_whispers.yaml")


class TestScenarioLoader:
    def test_load_island_of_whispers(self, island_scenario: ScenarioDef):
        assert island_scenario.id == "island_of_whispers"
        assert island_scenario.player_range == (2, 4)
        assert island_scenario.max_rounds == 30

    def test_locations_count(self, island_scenario):
        assert len(island_scenario.locations) == 14

    def test_items_count(self, island_scenario):
        assert len(island_scenario.items) == 33

    def test_recipes_count(self, island_scenario):
        assert len(island_scenario.recipes) == 6

    def test_characters_count(self, island_scenario):
        assert len(island_scenario.characters) == 4

    def test_random_events_count(self, island_scenario):
        assert len(island_scenario.random_events) == 6

    def test_location_connections_valid(self, island_scenario):
        for loc in island_scenario.locations.values():
            for direction, target_id in loc.connections.items():
                assert target_id in island_scenario.locations, (
                    f"Location '{loc.id}' connection '{direction}' "
                    f"points to unknown '{target_id}'"
                )

    def test_location_drops_valid(self, island_scenario):
        for loc in island_scenario.locations.values():
            for drop in loc.searchable_items:
                assert drop.item_id in island_scenario.items, (
                    f"Location '{loc.id}' drops unknown item '{drop.item_id}'"
                )

    def test_recipe_references_valid(self, island_scenario):
        for recipe in island_scenario.recipes:
            assert recipe.output_item_id in island_scenario.items
            for input_id in recipe.inputs:
                assert input_id in island_scenario.items

    def test_character_start_locations_valid(self, island_scenario):
        for char in island_scenario.characters:
            assert char.start_location in island_scenario.locations

    def test_character_start_inventory_valid(self, island_scenario):
        for char in island_scenario.characters:
            for item_id in char.start_inventory:
                assert item_id in island_scenario.items

    def test_character_victory_conditions(self, island_scenario):
        kinds = {c.id: c.personal_objective.trigger.kind for c in island_scenario.characters}
        assert kinds["ex_soldier"] == "eliminate_all"
        assert kinds["entomologist"] == "custom_flag"
        assert kinds["archaeologist"] == "collect_items"
        assert kinds["mysterious_stranger"] == "custom_flag"


class TestScenarioRegistry:
    def test_load_directory(self, registry):
        loaded = registry.load_directory(SCENARIOS_DIR)
        assert len(loaded) >= 1
        assert "island_of_whispers" in registry.list_ids()

    def test_get_loaded_scenario(self, registry, island_scenario):
        fetched = registry.get("island_of_whispers")
        assert fetched.id == island_scenario.id

    def test_get_unknown_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_get_random(self, registry, island_scenario):
        result = registry.get_random(player_count=3)
        assert result.id == "island_of_whispers"

    def test_get_random_invalid_player_count(self, registry, island_scenario):
        with pytest.raises(ScenarioLoadError):
            registry.get_random(player_count=10)

    def test_load_nonexistent_file(self, registry):
        with pytest.raises(ScenarioLoadError, match="not found"):
            registry.load_file("nonexistent.yaml")

    def test_load_nonexistent_directory(self, registry):
        with pytest.raises(ScenarioLoadError, match="not found"):
            registry.load_directory("nonexistent_dir")


class TestValidationErrors:
    def test_invalid_connection_reference(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("""
id: "bad"
name: "Bad"
setting: "test"
intro_text: "test"
locations:
  room_a:
    id: "room_a"
    name: "Room A"
    description: "A room"
    connections:
      north: "room_nowhere"
items: {}
recipes: []
characters: []
random_events: []
""", encoding="utf-8")
        registry = ScenarioRegistry()
        with pytest.raises(ScenarioLoadError, match="room_nowhere"):
            registry.load_file(bad_yaml)

    def test_invalid_item_reference_in_drop(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("""
id: "bad"
name: "Bad"
setting: "test"
intro_text: "test"
locations:
  room_a:
    id: "room_a"
    name: "Room A"
    description: "A room"
    searchable_items:
      - item_id: "nonexistent_item"
        weight: 1.0
        max_count: 1
items: {}
recipes: []
characters: []
random_events: []
""", encoding="utf-8")
        registry = ScenarioRegistry()
        with pytest.raises(ScenarioLoadError, match="nonexistent_item"):
            registry.load_file(bad_yaml)

    def test_invalid_character_start_location(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("""
id: "bad"
name: "Bad"
setting: "test"
intro_text: "test"
locations:
  room_a:
    id: "room_a"
    name: "Room A"
    description: "A room"
items: {}
recipes: []
characters:
  - id: "hero"
    name: "Hero"
    backstory: "A hero"
    base_stats:
      hp: 100
      max_hp: 100
      stamina: 100
      max_stamina: 100
      attack: 10
      defense: 5
      speed: 5
    start_location: "nowhere"
    secret_objective: "win"
    victory_condition:
      kind: "eliminate_all"
      params: {}
random_events: []
""", encoding="utf-8")
        registry = ScenarioRegistry()
        with pytest.raises(ScenarioLoadError, match="nowhere"):
            registry.load_file(bad_yaml)
