"""Victory condition evaluator.

Each VictoryCondition.kind maps to a pure function (GameState, ScenarioDef) -> list[str].
The engine calls check_victory() after every resolution phase. New condition types
can be added by registering evaluators in the _EVALUATORS dict.
"""

from __future__ import annotations

from typing import Callable

from mygame.shared.models import (
    Actor,
    CharacterDef,
    GameState,
    ScenarioDef,
    VictoryCondition,
)
from mygame.server.engine.state import get_alive_actors

EvaluatorFn = Callable[[GameState, ScenarioDef, Actor, CharacterDef], bool]


def _eval_survive_rounds(
    state: GameState, scenario: ScenarioDef, actor: Actor, char: CharacterDef,
) -> bool:
    target = char.victory_condition.params.get("rounds", scenario.max_rounds)
    return state.round >= target and actor.alive


def _eval_eliminate_all(
    state: GameState, scenario: ScenarioDef, actor: Actor, char: CharacterDef,
) -> bool:
    others_alive = any(
        a.alive and a.player_id != actor.player_id
        for a in state.actors.values()
    )
    return not others_alive


def _eval_collect_items(
    state: GameState, scenario: ScenarioDef, actor: Actor, char: CharacterDef,
) -> bool:
    params = char.victory_condition.params
    required_ids: list[str] = params.get("item_ids", [])
    at_location: str | None = params.get("at_location")

    if at_location and actor.location_id != at_location:
        return False

    for item_id in required_ids:
        count_needed = params.get(f"count_{item_id}", 1)
        if actor.inventory.get(item_id, 0) < count_needed:
            return False
    return True


def _eval_reach_location(
    state: GameState, scenario: ScenarioDef, actor: Actor, char: CharacterDef,
) -> bool:
    target = char.victory_condition.params.get("location")
    return actor.location_id == target and actor.alive


def _eval_escape(
    state: GameState, scenario: ScenarioDef, actor: Actor, char: CharacterDef,
) -> bool:
    escape_items: list[str] = char.victory_condition.params.get("required_items", [])
    escape_loc: str | None = char.victory_condition.params.get("at_location")

    if escape_loc and actor.location_id != escape_loc:
        return False

    for item_id in escape_items:
        if actor.inventory.get(item_id, 0) <= 0:
            return False
    return True


def _eval_custom_flag(
    state: GameState, scenario: ScenarioDef, actor: Actor, char: CharacterDef,
) -> bool:
    params = char.victory_condition.params

    required_knowledge: list[str] = params.get("required_knowledge", [])
    for k in required_knowledge:
        if k not in actor.knowledge:
            return False

    required_items: list[str] = params.get("required_items", [])
    for item_id in required_items:
        if actor.inventory.get(item_id, 0) <= 0:
            return False

    at_location: str | None = params.get("at_location")
    if at_location and actor.location_id != at_location:
        return False

    required_flags: list[str] = params.get("required_flags", [])
    for f in required_flags:
        if f not in state.global_flags:
            return False

    return True


_EVALUATORS: dict[str, EvaluatorFn] = {
    "survive_rounds": _eval_survive_rounds,
    "eliminate_all": _eval_eliminate_all,
    "collect_items": _eval_collect_items,
    "reach_location": _eval_reach_location,
    "escape": _eval_escape,
    "custom_flag": _eval_custom_flag,
}


def register_evaluator(kind: str, fn: EvaluatorFn) -> None:
    _EVALUATORS[kind] = fn


def check_actor_victory(
    state: GameState, scenario: ScenarioDef, player_id: str,
) -> bool:
    actor = state.actors.get(player_id)
    if actor is None or not actor.alive:
        return False

    char_def = None
    for c in scenario.characters:
        if c.id == actor.character_id:
            char_def = c
            break
    if char_def is None:
        return False

    evaluator = _EVALUATORS.get(char_def.victory_condition.kind)
    if evaluator is None:
        return False

    return evaluator(state, scenario, actor, char_def)


def check_victory(state: GameState, scenario: ScenarioDef) -> list[str]:
    winners: list[str] = []
    for player_id in state.actors:
        if check_actor_victory(state, scenario, player_id):
            winners.append(player_id)
    return winners
