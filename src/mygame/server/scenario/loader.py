"""Scenario loader — reads YAML files and validates against ScenarioDef."""

from __future__ import annotations

import random
from pathlib import Path

import yaml
from pydantic import ValidationError

from mygame.shared.models import ScenarioDef, Stats


class ScenarioLoadError(Exception):
    pass


class ScenarioRegistry:
    """Holds loaded scenarios and provides lookup / random selection."""

    def __init__(self) -> None:
        self._scenarios: dict[str, ScenarioDef] = {}

    def load_file(self, path: str | Path) -> ScenarioDef:
        path = Path(path)
        if not path.exists():
            raise ScenarioLoadError(f"Scenario file not found: {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ScenarioLoadError(f"Invalid scenario format in {path}")

        try:
            scenario = ScenarioDef.model_validate(raw)
        except ValidationError as e:
            raise ScenarioLoadError(f"Validation failed for {path}:\n{e}") from e

        _validate_references(scenario, path)
        self._scenarios[scenario.id] = scenario
        return scenario

    def load_directory(self, directory: str | Path) -> list[ScenarioDef]:
        directory = Path(directory)
        if not directory.is_dir():
            raise ScenarioLoadError(f"Scenario directory not found: {directory}")

        loaded = []
        for yaml_file in sorted(directory.glob("*.yaml")):
            loaded.append(self.load_file(yaml_file))
        return loaded

    def get(self, scenario_id: str) -> ScenarioDef:
        if scenario_id not in self._scenarios:
            raise KeyError(f"Scenario '{scenario_id}' not loaded")
        return self._scenarios[scenario_id]

    def get_random(self, player_count: int | None = None) -> ScenarioDef:
        candidates = list(self._scenarios.values())
        if player_count is not None:
            candidates = [
                s for s in candidates
                if s.player_range[0] <= player_count <= s.player_range[1]
            ]
        if not candidates:
            raise ScenarioLoadError("No scenarios available for the given player count")
        return random.choice(candidates)

    def list_ids(self) -> list[str]:
        return list(self._scenarios.keys())

    def __len__(self) -> int:
        return len(self._scenarios)


def _validate_references(scenario: ScenarioDef, source: Path) -> None:
    """Cross-check IDs referenced across the scenario definition."""
    errors: list[str] = []

    item_ids = set(scenario.items.keys())

    for loc in scenario.locations.values():
        for conn_target in loc.connections.values():
            if conn_target not in scenario.locations:
                errors.append(
                    f"Location '{loc.id}' connects to unknown location '{conn_target}'"
                )
        for drop in loc.searchable_items:
            if drop.item_id not in item_ids:
                errors.append(
                    f"Location '{loc.id}' drops unknown item '{drop.item_id}'"
                )

    for recipe in scenario.recipes:
        if recipe.output_item_id not in item_ids:
            errors.append(
                f"Recipe outputs unknown item '{recipe.output_item_id}'"
            )
        for input_id in recipe.inputs:
            if input_id not in item_ids:
                errors.append(
                    f"Recipe for '{recipe.output_item_id}' requires unknown item '{input_id}'"
                )

    for char in scenario.characters:
        if char.start_location not in scenario.locations:
            errors.append(
                f"Character '{char.id}' starts at unknown location '{char.start_location}'"
            )
        for inv_id in char.start_inventory:
            if inv_id not in item_ids:
                errors.append(
                    f"Character '{char.id}' starts with unknown item '{inv_id}'"
                )
        for ability in char.abilities:
            _validate_effects(
                scenario, errors,
                context=f"Character '{char.id}' ability '{ability.id}'",
                effects=ability.effects,
            )

    knowledge_ids = {k.id for k in scenario.knowledge_sources}
    for src in scenario.knowledge_sources:
        if src.location_id and src.location_id not in scenario.locations:
            errors.append(
                f"Knowledge source '{src.id}' references unknown location '{src.location_id}'"
            )
        if src.item_id and src.item_id not in item_ids:
            errors.append(
                f"Knowledge source '{src.id}' references unknown item '{src.item_id}'"
            )

    event_ids = {e.id for e in scenario.random_events}
    for event in scenario.random_events:
        _validate_effects(
            scenario, errors,
            context=f"Random event '{event.id}'",
            effects=event.effects,
        )

    if errors:
        detail = "\n  ".join(errors)
        raise ScenarioLoadError(f"Reference errors in {source}:\n  {detail}")


def _validate_effects(
    scenario: ScenarioDef,
    errors: list[str],
    context: str,
    effects: list[dict],
) -> None:
    """Cross-check IDs referenced inside effect dicts."""
    knowledge_ids = {k.id for k in scenario.knowledge_sources}
    event_ids = {e.id for e in scenario.random_events}

    for eff in effects:
        etype = eff.get("type", "")
        if etype == "grant_knowledge":
            source_id = eff.get("source", "")
            if source_id and source_id not in knowledge_ids:
                errors.append(
                    f"{context} grants unknown knowledge source '{source_id}'"
                )
        elif etype == "trigger_event":
            event_id = eff.get("event_id", "")
            if event_id and event_id not in event_ids:
                errors.append(
                    f"{context} triggers unknown event '{event_id}'"
                )
            target_loc = eff.get("location_id")
            if target_loc and target_loc not in scenario.locations:
                errors.append(
                    f"{context} targets unknown location '{target_loc}'"
                )
        elif etype == "stat_bonus":
            stat = eff.get("stat", "")
            if stat and stat not in Stats.model_fields:
                errors.append(
                    f"{context} references unknown stat '{stat}'"
                )
