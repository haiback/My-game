"""Generic effect application — shared by random events and character abilities.

Effects are data-driven dicts defined in scenario YAML, so scenarios can
express arbitrary mechanics (damage, poison, permanent skill gains,
knowledge grants, chaining other events) without engine changes.

Effect schema (type + optional fields):
  {type: damage,            amount: 8,  target: <target>, location_id: <loc>}
  {type: heal,              amount: 10, target: <target>}
  {type: stamina,           amount: -10, target: <target>}
  {type: status,            status: poison, rounds: 3, magnitude: 3, target: <target>}
  {type: stat_bonus,        stat: attack, amount: 2, target: <target>}  # permanent
  {type: grant_knowledge,   source: <knowledge_source_id>, target: <target>}
  {type: set_global_flag,   flag: <name>}
  {type: trigger_event,     event_id: <event_id>, location_id: <loc|optional>}

Targets: self | single | all | all_others | random | at_location | others_at_location
"""

from __future__ import annotations

import random

from mygame.shared.models import (
    GameEvent,
    GameState,
    ScenarioDef,
    StatusEffect,
)

_MAX_DEPTH = 5


def fire_event(
    state: GameState,
    scenario: ScenarioDef,
    event_id: str,
    context: dict | None = None,
    rng: random.Random | None = None,
    force: bool = False,
    _depth: int = 0,
) -> list[GameEvent]:
    """Execute a scenario random event, honoring round_trigger and max_fires.

    force=True bypasses the round_trigger check (used by ability-triggered
    events); max_fires is always respected.
    """
    if _depth > _MAX_DEPTH:
        return []

    event = next((e for e in scenario.random_events if e.id == event_id), None)
    if event is None:
        return []

    fired = state.fired_events.get(event_id, 0)
    if fired >= event.max_fires:
        return []

    if not force:
        if event.round_trigger is None or state.round < event.round_trigger:
            return []

    state.fired_events[event_id] = fired + 1
    rng = rng or random.Random()

    events = [
        GameEvent(
            round=state.round,
            kind="random_event",
            actor_ids=[],
            payload={"event_id": event.id, "event_name": event.name},
            visibility="public",
            narrative_seed=event.narrative_seed,
        )
    ]
    events.extend(
        apply_effects(
            state, scenario, event.effects,
            context or {}, rng, _depth=_depth + 1,
        )
    )
    return events


def apply_effects(
    state: GameState,
    scenario: ScenarioDef,
    effects: list[dict],
    context: dict | None = None,
    rng: random.Random | None = None,
    _depth: int = 0,
) -> list[GameEvent]:
    if _depth > _MAX_DEPTH:
        return []

    context = dict(context or {})
    rng = rng or random.Random()
    events: list[GameEvent] = []

    for eff in effects:
        etype = eff.get("type", "")

        if etype == "trigger_event":
            merged = dict(context)
            if eff.get("location_id"):
                merged["location"] = eff["location_id"]
            if eff.get("target"):
                merged["target"] = eff["target"]
            events.extend(fire_event(
                state, scenario, str(eff.get("event_id", "")),
                context=merged, rng=rng, force=True, _depth=_depth + 1,
            ))
            continue

        if etype == "set_global_flag":
            flag = str(eff.get("flag", ""))
            if flag:
                state.global_flags.add(flag)
            continue

        targets = _resolve_targets(state, eff, context, rng)
        if not targets:
            continue

        for pid in targets:
            evts = _apply_one(state, scenario, pid, eff)
            events.extend(evts)

    return events


def _apply_one(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
    eff: dict,
) -> list[GameEvent]:
    actor = state.actors[player_id]
    if not actor.alive:
        return []

    etype = eff.get("type", "")
    amount = int(eff.get("amount", 0))

    if etype == "damage":
        actor.stats.hp = max(0, actor.stats.hp - amount)
        events = [GameEvent(
            round=state.round,
            kind="event_damage",
            actor_ids=[player_id],
            payload={"amount": amount, "hp": actor.stats.hp},
            visibility="private",
            private_to=player_id,
            narrative_seed=f"{actor.player_name} took {amount} damage.",
        )]
        if actor.stats.hp <= 0 and actor.alive:
            actor.alive = False
            events.append(GameEvent(
                round=state.round,
                kind="death",
                actor_ids=[player_id],
                payload={"cause": "event"},
                visibility="public",
                narrative_seed=f"{actor.player_name} has died.",
            ))
        return events

    if etype == "heal":
        if actor.alive:
            actor.stats.hp = min(actor.stats.max_hp, actor.stats.hp + amount)
        return [GameEvent(
            round=state.round,
            kind="event_heal",
            actor_ids=[player_id],
            payload={"amount": amount, "hp": actor.stats.hp},
            visibility="private",
            private_to=player_id,
            narrative_seed=f"{actor.player_name} recovered {amount} HP.",
        )]

    if etype == "stamina":
        actor.stats.stamina = max(
            0, min(actor.stats.max_stamina, actor.stats.stamina + amount)
        )
        return [GameEvent(
            round=state.round,
            kind="event_stamina",
            actor_ids=[player_id],
            payload={"amount": amount, "stamina": actor.stats.stamina},
            visibility="private",
            private_to=player_id,
            narrative_seed=f"{actor.player_name}'s stamina changed by {amount}.",
        )]

    if etype == "status":
        status_kind = str(eff.get("status", "poison"))
        rounds = int(eff.get("rounds", 3))
        magnitude = int(eff.get("magnitude", 0))
        actor.status_effects = [
            e for e in actor.status_effects if e.kind != status_kind
        ]
        actor.status_effects.append(StatusEffect(
            kind=status_kind, rounds_left=rounds, magnitude=magnitude,
        ))
        return [GameEvent(
            round=state.round,
            kind="status_applied",
            actor_ids=[player_id],
            payload={"status": status_kind, "rounds": rounds},
            visibility="private",
            private_to=player_id,
            narrative_seed=f"{actor.player_name} was afflicted by {status_kind}.",
        )]

    if etype == "stat_bonus":
        stat = str(eff.get("stat", ""))
        if stat and hasattr(actor.stats, stat):
            current = getattr(actor.stats, stat)
            setattr(actor.stats, stat, current + amount)
        return [GameEvent(
            round=state.round,
            kind="stat_bonus",
            actor_ids=[player_id],
            payload={"stat": stat, "amount": amount},
            visibility="private",
            private_to=player_id,
            narrative_seed=f"{actor.player_name}'s {stat} increased by {amount}.",
        )]

    if etype == "grant_knowledge":
        from mygame.server.engine.state import grant_knowledge

        source_id = str(eff.get("source", ""))
        evt = grant_knowledge(state, scenario, player_id, source_id)
        return [evt] if evt else []

    return []


def _resolve_targets(
    state: GameState,
    eff: dict,
    context: dict,
    rng: random.Random,
) -> list[str]:
    target = eff.get("target", "self")
    alive = [pid for pid, a in state.actors.items() if a.alive]

    if target == "self":
        src = context.get("source")
        return [src] if src in state.actors else []
    if target == "single":
        tg = context.get("target")
        return [tg] if tg in state.actors and state.actors[tg].alive else []
    if target == "all":
        return alive
    if target == "all_others":
        src = context.get("source")
        return [p for p in alive if p != src]
    if target == "random":
        return [rng.choice(alive)] if alive else []
    if target == "at_location":
        loc = eff.get("location_id") or context.get("location")
        if not loc:
            return []
        return [
            pid for pid in alive
            if state.actors[pid].location_id == loc
            and not state.actors[pid].traveling
        ]
    if target == "others_at_location":
        loc = eff.get("location_id") or context.get("location")
        if not loc:
            return []
        src = context.get("source")
        return [
            pid for pid in alive
            if pid != src and state.actors[pid].location_id == loc
            and not state.actors[pid].traveling
        ]
    return []
