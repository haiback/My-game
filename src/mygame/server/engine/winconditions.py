"""Victory / objective evaluator.

An Objective has a `trigger` (VictoryCondition) and an optional `victory` flag.
Objectives are settled after each resolution phase:

  - personal objective: if its trigger is satisfied, apply its `effects`; if
    `victory=True` the owner also wins.
  - faction objective: if its trigger is satisfied, apply its `effects`; if
    `victory=True` all alive faction members also win.

Winning is only one possible outcome — `effects` can instead grant stat
boosts, items, etc. (scenario-defined).
"""

from __future__ import annotations

from typing import Callable

from mygame.shared.models import (
    Actor,
    CharacterDef,
    GameEvent,
    GameState,
    ScenarioDef,
    VictoryCondition,
)
from mygame.server.engine.effects import apply_effects

EvaluatorFn = Callable[[GameState, ScenarioDef, Actor, VictoryCondition], bool]


def _eval_survive_rounds(
    state: GameState, scenario: ScenarioDef, actor: Actor, condition: VictoryCondition,
) -> bool:
    target = condition.params.get("rounds", scenario.max_rounds)
    return state.round >= target and actor.alive


def _eval_eliminate_all(
    state: GameState, scenario: ScenarioDef, actor: Actor, condition: VictoryCondition,
) -> bool:
    others_alive = any(
        a.alive and a.player_id != actor.player_id for a in state.actors.values()
    )
    return not others_alive


def _eval_collect_items(
    state: GameState, scenario: ScenarioDef, actor: Actor, condition: VictoryCondition,
) -> bool:
    params = condition.params
    required_ids: list[str] = params.get("item_ids", [])
    at_location: str | None = params.get("at_location")

    if at_location and (actor.traveling or actor.location_id != at_location):
        return False

    for item_id in required_ids:
        count_needed = params.get(f"count_{item_id}", 1)
        if actor.inventory.get(item_id, 0) < count_needed:
            return False
    return True


def _eval_reach_location(
    state: GameState, scenario: ScenarioDef, actor: Actor, condition: VictoryCondition,
) -> bool:
    target = condition.params.get("location")
    return actor.location_id == target and actor.alive and not actor.traveling


def _eval_escape(
    state: GameState, scenario: ScenarioDef, actor: Actor, condition: VictoryCondition,
) -> bool:
    escape_items: list[str] = condition.params.get("required_items", [])
    escape_loc: str | None = condition.params.get("at_location")

    if escape_loc and (actor.traveling or actor.location_id != escape_loc):
        return False

    for item_id in escape_items:
        if actor.inventory.get(item_id, 0) <= 0:
            return False
    return True


def _eval_custom_flag(
    state: GameState, scenario: ScenarioDef, actor: Actor, condition: VictoryCondition,
) -> bool:
    params = condition.params

    required_knowledge: list[str] = params.get("required_knowledge", [])
    for k in required_knowledge:
        if k not in actor.knowledge:
            return False

    required_items: list[str] = params.get("required_items", [])
    for item_id in required_items:
        if actor.inventory.get(item_id, 0) <= 0:
            return False

    at_location: str | None = params.get("at_location")
    if at_location and (actor.traveling or actor.location_id != at_location):
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


def _get_char_def(scenario: ScenarioDef, actor: Actor) -> CharacterDef | None:
    for c in scenario.characters:
        if c.id == actor.character_id:
            return c
    return None


def check_personal_objective(
    state: GameState, scenario: ScenarioDef, actor: Actor,
) -> bool:
    char = _get_char_def(scenario, actor)
    if char is None or char.personal_objective is None:
        return False
    evaluator = _EVALUATORS.get(char.personal_objective.trigger.kind)
    if evaluator is None:
        return False
    return evaluator(state, scenario, actor, char.personal_objective.trigger)


def check_actor_victory(
    state: GameState, scenario: ScenarioDef, player_id: str,
) -> bool:
    """Whether a single player's personal objective is satisfied as a win."""
    actor = state.actors.get(player_id)
    if actor is None or not actor.alive:
        return False
    char = _get_char_def(scenario, actor)
    if char is None or char.personal_objective is None:
        return False
    if not char.personal_objective.victory:
        return False
    return check_personal_objective(state, scenario, actor)


def check_faction_objective(
    state: GameState, scenario: ScenarioDef, faction_id: str,
) -> bool:
    faction = scenario.factions.get(faction_id)
    if faction is None or faction.objective is None:
        return False

    kind = faction.objective.trigger.kind
    if kind == "eliminate_all":
        others_alive = any(
            a.alive and a.faction_id != faction_id for a in state.actors.values()
        )
        return not others_alive
    if kind == "survive_rounds":
        target = faction.objective.trigger.params.get("rounds", scenario.max_rounds)
        return state.round >= target and any(
            a.alive and a.faction_id == faction_id for a in state.actors.values()
        )
    return False


def settle_objectives(
    state: GameState,
    scenario: ScenarioDef,
    rng=None,
) -> list[GameEvent]:
    """Apply effects for newly-achieved objectives and record them as settled.

    Runs once per resolution phase; each objective settles at most once.
    """
    events: list[GameEvent] = []

    for pid, actor in state.actors.items():
        if not actor.alive:
            continue
        char = _get_char_def(scenario, actor)
        if char is None or char.personal_objective is None:
            continue
        key = f"personal:{char.id}"
        if key in state.settled_objectives:
            continue
        if not check_personal_objective(state, scenario, actor):
            continue
        state.settled_objectives.add(key)
        events.extend(apply_effects(
            state, scenario, char.personal_objective.effects,
            {"source": pid}, rng,
        ))
        events.append(GameEvent(
            round=state.round,
            kind="objective_achieved",
            actor_ids=[pid],
            payload={"objective": key},
            visibility="private",
            private_to=pid,
            narrative_seed=f"{actor.player_name} achieved their personal goal.",
        ))

    for fid, faction in scenario.factions.items():
        if faction.objective is None:
            continue
        key = f"faction:{fid}"
        if key in state.settled_objectives:
            continue
        if not check_faction_objective(state, scenario, fid):
            continue
        state.settled_objectives.add(key)
        events.extend(apply_effects(
            state, scenario, faction.objective.effects, {}, rng,
        ))
        events.append(GameEvent(
            round=state.round,
            kind="objective_achieved",
            actor_ids=[],
            payload={"objective": key},
            visibility="public",
            narrative_seed=f"{faction.name} achieved their goal.",
        ))

    return events


def check_victory(state: GameState, scenario: ScenarioDef) -> list[str]:
    winners: list[str] = []

    for pid, actor in state.actors.items():
        if not actor.alive:
            continue
        char = _get_char_def(scenario, actor)
        if char is None or char.personal_objective is None:
            continue
        if not char.personal_objective.victory:
            continue
        if check_personal_objective(state, scenario, actor):
            winners.append(pid)

    for fid, faction in scenario.factions.items():
        if faction.objective is None or not faction.objective.victory:
            continue
        if not check_faction_objective(state, scenario, fid):
            continue
        for pid, actor in state.actors.items():
            if actor.alive and actor.faction_id == fid and pid not in winners:
                winners.append(pid)

    return winners
