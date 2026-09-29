"""Condition evaluation for ability variants.

A variant's `condition` is a data-driven dict. All keys in a dict are
AND-ed together; `all_of` / `any_of` nest lists of sub-conditions.

Supported predicates (each key maps to an expected value):

  Actor (self):
    self_hp_below: int       -> hp < value
    self_hp_above: int       -> hp > value
    self_hp_pct_below: int   -> hp/max_hp*100 < value
    self_stamina_below: int  -> stamina < value
    self_has_status: str     -> status kind present on self
    self_has_knowledge: str  -> knowledge flag present on self
    self_has_item: str       -> item id present in inventory

  Target (single-target abilities):
    target_hp_pct_below: int -> target hp/max_hp*100 < value
    target_has_status: str   -> status kind present on target

  Location (actor's current location):
    at_location: str         -> location id matches
    at_location_tag: str     -> location has the given tag

  Global:
    global_flag: str         -> flag present in state.global_flags

  Combinators:
    all_of: list[dict]       -> every sub-condition matches
    any_of: list[dict]       -> at least one sub-condition matches

An unknown predicate is simply ignored (matches as true), so adding a new
predicate never breaks an existing scenario.
"""

from __future__ import annotations

from typing import Any

from mygame.shared.models import Actor, GameState, ScenarioDef


def select_variant(
    state: GameState,
    scenario: ScenarioDef,
    actor: Actor,
    context: dict,
    variants: list[Any],
) -> tuple[str, list[dict]]:
    """Pick the first matching variant; return (variant_name, effects).

    Falls back to ("", []) when no variant matches — the caller then uses
    the ability's base effects.
    """
    for v in variants:
        if evaluate_condition(state, scenario, actor, context, v.condition):
            return v.name, list(v.effects)
    return "", []


def evaluate_condition(
    state: GameState,
    scenario: ScenarioDef,
    actor: Actor,
    context: dict,
    condition: dict,
) -> bool:
    for key, value in condition.items():
        if key == "all_of":
            if not all(
                evaluate_condition(state, scenario, actor, context, sub)
                for sub in value
            ):
                return False
            continue
        if key == "any_of":
            if not any(
                evaluate_condition(state, scenario, actor, context, sub)
                for sub in value
            ):
                return False
            continue

        if not _match_predicate(state, scenario, actor, context, key, value):
            return False
    return True


def _match_predicate(
    state: GameState,
    scenario: ScenarioDef,
    actor: Actor,
    context: dict,
    key: str,
    value: Any,
) -> bool:
    target = state.actors.get(context.get("target") or "") if context.get("target") else None
    loc = scenario.locations.get(actor.location_id)

    if key == "self_hp_below":
        return actor.stats.hp < int(value)
    if key == "self_hp_above":
        return actor.stats.hp > int(value)
    if key == "self_hp_pct_below":
        return _hp_pct(actor) < int(value)
    if key == "self_stamina_below":
        return actor.stats.stamina < int(value)
    if key == "self_has_status":
        return any(e.kind == value for e in actor.status_effects)
    if key == "self_has_knowledge":
        return value in actor.knowledge
    if key == "self_has_item":
        return actor.inventory.get(value, 0) > 0

    if key == "target_hp_pct_below":
        return target is not None and _hp_pct(target) < int(value)
    if key == "target_has_status":
        return target is not None and any(e.kind == value for e in target.status_effects)

    if key == "at_location":
        return actor.location_id == value
    if key == "at_location_tag":
        return loc is not None and value in loc.tags

    if key == "global_flag":
        return value in state.global_flags

    return True  # unknown predicate → permissive


def _hp_pct(actor: Actor) -> float:
    if actor.stats.max_hp <= 0:
        return 0.0
    return actor.stats.hp / actor.stats.max_hp * 100.0
