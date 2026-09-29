"""Simultaneous action resolution.

Takes all player actions for a round and resolves them in a deterministic
order that ensures fairness — resolution order depends on action TYPE,
never on submission timestamp.

Resolution order:
  1. TALK / WAIT     (no-ops, generate social events)
  2. USE / REST / SPECIAL (self-targeted / ability use)
  3. MOVE            (all moves apply simultaneously)
  4. SEARCH / CRAFT  (loot drawn with per-round seeded RNG)
  5. ATTACK          (simultaneous damage based on pre-round stats)
  6. Round-end: status effects, danger damage, random events, deaths
"""

from __future__ import annotations

import random

from mygame.shared.models import (
    Action,
    ActionType,
    GameEvent,
    GameState,
    ScenarioDef,
)
from mygame.server.engine.conditions import select_variant
from mygame.server.engine.effects import apply_effects, fire_event
from mygame.server.engine.state import (
    attack_actor,
    cancel_travel,
    craft_item,
    equip_item,
    get_alive_actors,
    move_actor,
    progress_travel,
    rest_actor,
    search_location,
    tick_status_effects,
    trigger_location_points,
    use_item,
    apply_danger_damage,
)

_RESOLUTION_ORDER = [
    {ActionType.TALK, ActionType.WAIT},
    {ActionType.USE, ActionType.REST, ActionType.SPECIAL},
    {ActionType.CANCEL},
    {ActionType.MOVE},
    {ActionType.SEARCH, ActionType.CRAFT},
    {ActionType.ATTACK},
]


def resolve_round(
    state: GameState,
    scenario: ScenarioDef,
    actions: dict[str, Action],
    round_seed: int,
) -> list[GameEvent]:
    events: list[GameEvent] = []
    rng = random.Random(round_seed)

    filled: dict[str, Action] = {}
    for pid in state.actors:
        if pid in actions:
            filled[pid] = actions[pid]
        else:
            filled[pid] = Action(player_id=pid, type=ActionType.WAIT)

    for tier in _RESOLUTION_ORDER:
        tier_events = _resolve_tier(state, scenario, filled, tier, rng)
        events.extend(tier_events)

    end_events = _resolve_round_end(state, scenario, rng)
    events.extend(end_events)

    return events


def _resolve_tier(
    state: GameState,
    scenario: ScenarioDef,
    actions: dict[str, Action],
    tier: set[ActionType],
    rng: random.Random,
) -> list[GameEvent]:
    events: list[GameEvent] = []

    tier_actions = [
        (pid, act) for pid, act in actions.items()
        if act.type in tier and state.actors[pid].alive
    ]

    if ActionType.MOVE in tier:
        move_events = _resolve_moves(state, scenario, tier_actions, rng)
        events.extend(move_events)
        return events

    if ActionType.ATTACK in tier:
        atk_events = _resolve_attacks(state, scenario, tier_actions)
        events.extend(atk_events)
        return events

    for pid, act in tier_actions:
        evt = _resolve_single(state, scenario, pid, act, rng)
        if evt is not None:
            if isinstance(evt, list):
                events.extend(evt)
            else:
                events.append(evt)

    return events


def _resolve_moves(
    state: GameState,
    scenario: ScenarioDef,
    moves: list[tuple[str, Action]],
    rng: random.Random,
) -> list[GameEvent]:
    events: list[GameEvent] = []

    for pid, act in moves:
        direction = act.params.get("direction", "")
        evts = move_actor(state, scenario, pid, direction, rng)
        events.extend(evts)

    return events


def _resolve_attacks(
    state: GameState,
    scenario: ScenarioDef,
    attacks: list[tuple[str, Action]],
) -> list[GameEvent]:
    events: list[GameEvent] = []

    pre_round_hp: dict[str, int] = {}
    for pid, actor in state.actors.items():
        pre_round_hp[pid] = actor.stats.hp

    for pid, act in attacks:
        target_id = act.params.get("target", "")
        if target_id not in state.actors:
            continue
        if not state.actors[target_id].alive:
            continue

        evt = attack_actor(state, scenario, pid, target_id)
        if evt:
            events.append(evt)

    return events


def _resolve_single(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
    action: Action,
    rng: random.Random,
) -> GameEvent | list[GameEvent] | None:
    match action.type:
        case ActionType.WAIT:
            return None
        case ActionType.TALK:
            return GameEvent(
                round=state.round,
                kind="talk",
                actor_ids=[player_id],
                payload={"message": action.params.get("message", "")},
                visibility="public",
                narrative_seed=f"{state.actors[player_id].player_name} said something.",
            )
        case ActionType.REST:
            return rest_actor(state, player_id)
        case ActionType.USE:
            item_id = action.params.get("item_id", "")
            if action.params.get("equip"):
                return equip_item(state, scenario, player_id, item_id)
            return use_item(state, scenario, player_id, item_id)
        case ActionType.SEARCH:
            return search_location(state, scenario, player_id, rng)
        case ActionType.CRAFT:
            recipe_index = action.params.get("recipe_index", 0)
            return craft_item(state, scenario, player_id, recipe_index)
        case ActionType.SPECIAL:
            return _resolve_special(state, scenario, player_id, action, rng)
        case ActionType.CANCEL:
            return cancel_travel(state, scenario, player_id)
        case _:
            return None


def _resolve_special(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
    action: Action,
    rng: random.Random,
) -> list[GameEvent]:
    actor = state.actors[player_id]
    char_def = next(
        (c for c in scenario.characters if c.id == actor.character_id), None
    )
    if char_def is None:
        return []

    ability_id = action.params.get("ability_id", "")
    ability = next((a for a in char_def.abilities if a.id == ability_id), None)
    if ability is None:
        return []
    if actor.ability_cooldowns.get(ability.id, 0) > 0:
        return []
    if actor.stats.stamina < ability.stamina_cost:
        return []

    target_pid = action.params.get("target") or None
    if ability.target == "single":
        if (
            not target_pid
            or target_pid not in state.actors
            or not state.actors[target_pid].alive
        ):
            target_pid = _first_opponent(state, player_id)
        if target_pid is None:
            return []
        if actor.traveling or state.actors[target_pid].traveling:
            return []
        if state.actors[target_pid].location_id != actor.location_id:
            return []

    context = {"source": player_id, "target": target_pid}
    if target_pid:
        context["location"] = state.actors[target_pid].location_id
    elif ability.target == "location":
        context["location"] = action.params.get("location") or actor.location_id

    actor.stats.stamina -= ability.stamina_cost
    actor.ability_cooldowns[ability.id] = ability.cooldown_rounds

    variant_name, variant_effects = select_variant(
        state, scenario, actor, context, ability.variants
    )
    effects = variant_effects or ability.effects
    display_name = f"{ability.name}·{variant_name}" if variant_name else ability.name

    events = [GameEvent(
        round=state.round,
        kind="ability_used",
        actor_ids=[player_id],
        payload={
            "ability_id": ability.id,
            "ability_name": ability.name,
            "variant": variant_name,
            "target": target_pid,
        },
        visibility="public",
        narrative_seed=f"{actor.player_name} used ability: {display_name}.",
    )]
    events.extend(apply_effects(state, scenario, effects, context, rng))
    return events


def _first_opponent(state: GameState, player_id: str) -> str | None:
    actor = state.actors[player_id]
    for pid, other in state.actors.items():
        if (
            pid != player_id
            and other.alive
            and not other.traveling
            and other.location_id == actor.location_id
        ):
            return pid
    return None


def _resolve_round_end(
    state: GameState,
    scenario: ScenarioDef,
    rng: random.Random,
) -> list[GameEvent]:
    events: list[GameEvent] = []

    status_events = tick_status_effects(state)
    events.extend(status_events)

    for actor in get_alive_actors(state):
        evt = apply_danger_damage(state, scenario, actor.player_id)
        if evt:
            events.append(evt)

    events.extend(progress_travel(state, scenario, rng))

    for actor in get_alive_actors(state):
        if not actor.traveling:
            events.extend(
                trigger_location_points(state, scenario, actor.player_id, "stay", rng)
            )

    events.extend(_resolve_random_events(state, scenario, rng))

    for actor in state.actors.values():
        for ability_id in list(actor.ability_cooldowns.keys()):
            if actor.ability_cooldowns[ability_id] > 0:
                actor.ability_cooldowns[ability_id] -= 1

    for actor in list(state.actors.values()):
        if actor.stats.hp <= 0 and actor.alive:
            actor.alive = False
            events.append(GameEvent(
                round=state.round,
                kind="death",
                actor_ids=[actor.player_id],
                payload={"location": actor.location_id},
                visibility="public",
                narrative_seed=f"{actor.player_name} has died.",
            ))

    return events


def _resolve_random_events(
    state: GameState,
    scenario: ScenarioDef,
    rng: random.Random,
) -> list[GameEvent]:
    """Fire one weighted random event per round among those due."""
    due = []
    for ev in scenario.random_events:
        if state.fired_events.get(ev.id, 0) >= ev.max_fires:
            continue
        if ev.round_trigger is None:
            continue
        if state.round < ev.round_trigger:
            continue
        due.append(ev)

    if not due:
        return []

    chosen = rng.choices(due, weights=[ev.weight for ev in due], k=1)[0]
    return fire_event(state, scenario, chosen.id, context={}, rng=rng, force=True)
