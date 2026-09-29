"""Game state management — initialization, mutations, and player views.

This module is the single authority on how game state changes. Every
mutation returns a GameEvent for the event log. The game loop calls
into these functions; it never mutates models directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from mygame.shared.models import (
    Action,
    ActionType,
    Actor,
    CharacterDef,
    GameEvent,
    GamePhase,
    GameState,
    Item,
    LegalAction,
    Location,
    LocationRuntime,
    MapLocation,
    PlayerView,
    ScenarioDef,
    Stats,
    StatusEffect,
    VisibleActor,
)
from mygame.server.engine.effects import apply_effects


# ---------------------------------------------------------------------------
# Player assignment (input to init_game)
# ---------------------------------------------------------------------------

@dataclass
class PlayerAssignment:
    player_id: str
    player_name: str
    character_id: str


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_game(
    scenario: ScenarioDef,
    players: list[PlayerAssignment],
) -> GameState:
    char_map = {c.id: c for c in scenario.characters}

    actors: dict[str, Actor] = {}
    for p in players:
        char = char_map[p.character_id]
        actors[p.player_id] = Actor(
            character_id=char.id,
            player_id=p.player_id,
            player_name=p.player_name,
            faction_id=char.faction_id,
            stats=char.base_stats.model_copy(),
            location_id=char.start_location,
            inventory=dict(char.start_inventory),
        )

    location_state: dict[str, LocationRuntime] = {}
    for loc in scenario.locations.values():
        remaining: dict[str, int] = {}
        for drop in loc.searchable_items:
            remaining[drop.item_id] = drop.max_count
        location_state[loc.id] = LocationRuntime(remaining_loot=remaining)

    return GameState(
        scenario_id=scenario.id,
        round=0,
        phase=GamePhase.LOBBY,
        actors=actors,
        location_state=location_state,
    )


# ---------------------------------------------------------------------------
# State mutations — each returns a GameEvent (or None if no-op)
# ---------------------------------------------------------------------------

def move_actor(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
    direction: str,
    rng: random.Random | None = None,
) -> list[GameEvent]:
    actor = state.actors[player_id]
    if actor.traveling:
        return []
    loc = scenario.locations[actor.location_id]

    if direction not in loc.connections:
        return []

    old_loc = actor.location_id
    new_loc = loc.connections[direction]
    travel_cost = _travel_cost(scenario, loc, direction)

    stamina_cost = _move_stamina_cost(scenario, loc)
    if stamina_cost > 0:
        actor.stats.stamina = max(0, actor.stats.stamina - stamina_cost)

    if travel_cost > 1:
        # Enter transit. `location_id` keeps the origin so lookups stay valid,
        # but the actor is logically "on the road" and `traveling` is truth.
        actor.travel_from = old_loc
        actor.travel_to = new_loc
        actor.travel_direction = direction
        actor.travel_remaining = travel_cost
        return [GameEvent(
            round=state.round,
            kind="travel_start",
            actor_ids=[player_id],
            payload={"from": old_loc, "to": new_loc, "direction": direction, "rounds": travel_cost},
            visibility="public",
            narrative_seed=(
                f"{actor.player_name} set out from {loc.name} toward "
                f"{scenario.locations[new_loc].name} ({travel_cost} rounds)."
            ),
        )]

    return _arrive_at(state, scenario, actor, old_loc, new_loc, direction, rng)


def cancel_travel(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
) -> list[GameEvent]:
    """Turn a traveling actor around and head back the way they came.

    Backtracking takes as many rounds as were already traveled, so the round
    spent turning around is never a free teleport home.
    """
    actor = state.actors[player_id]
    if not actor.traveling:
        return []

    origin = actor.travel_from or actor.location_id
    destination = actor.travel_to or actor.location_id
    total = _travel_cost_between(scenario, origin, destination)
    traveled = max(0, total - actor.travel_remaining)

    if traveled <= 0:
        # Just set out — turning around means never leaving.
        actor.travel_from = None
        actor.travel_to = None
        actor.travel_direction = None
        actor.travel_remaining = 0
        return [GameEvent(
            round=state.round,
            kind="travel_cancel",
            actor_ids=[player_id],
            payload={"returned_to": origin},
            visibility="public",
            narrative_seed=f"{actor.player_name} turned back to {scenario.locations[origin].name}.",
        )]

    actor.travel_from = destination
    actor.travel_to = origin
    actor.travel_direction = _reverse_direction(scenario, destination, origin)
    actor.travel_remaining = traveled
    return [GameEvent(
        round=state.round,
        kind="travel_cancel",
        actor_ids=[player_id],
        payload={"returning_to": origin, "rounds": traveled},
        visibility="public",
        narrative_seed=(
            f"{actor.player_name} turned around, heading back to "
            f"{scenario.locations[origin].name}."
        ),
    )]


def trigger_location_points(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
    trigger: str,
    rng: random.Random | None = None,
) -> list[GameEvent]:
    """Fire a location's `points` (traps / NPC encounters / ...) matching the
    given trigger, honoring chance and one-shot (repeatable=False) semantics."""
    actor = state.actors.get(player_id)
    if actor is None or not actor.alive:
        return []
    loc = scenario.locations.get(actor.location_id)
    if loc is None:
        return []
    rng = rng or random.Random()
    loc_rt = state.location_state[actor.location_id]
    events: list[GameEvent] = []
    for point in loc.points:
        if point.trigger != trigger:
            continue
        if not point.repeatable and point.id in loc_rt.flags:
            continue
        if rng.random() > point.chance:
            continue
        loc_rt.flags.add(point.id)
        events.append(GameEvent(
            round=state.round,
            kind=point.kind,
            actor_ids=[player_id],
            payload={"point_id": point.id, "location_id": actor.location_id},
            visibility="private",
            private_to=player_id,
            narrative_seed=point.narrative_seed or f"{actor.player_name} encountered {point.name}.",
        ))
        events.extend(apply_effects(
            state, scenario, point.effects,
            context={"source": player_id, "location": actor.location_id},
            rng=rng,
        ))
    return events


def progress_travel(
    state: GameState,
    scenario: ScenarioDef,
    rng: random.Random | None = None,
) -> list[GameEvent]:
    """Advance every traveling actor one round; arrive when the count hits 0."""
    events: list[GameEvent] = []
    for actor in state.actors.values():
        if not actor.alive or not actor.traveling:
            continue
        actor.travel_remaining -= 1
        if actor.travel_remaining <= 0:
            old_loc = actor.travel_from or actor.location_id
            new_loc = actor.travel_to or actor.location_id
            direction = actor.travel_direction or ""
            events.extend(_arrive_at(state, scenario, actor, old_loc, new_loc, direction, rng))
    return events


def _arrive_at(
    state: GameState,
    scenario: ScenarioDef,
    actor: Actor,
    old_loc: str,
    new_loc: str,
    direction: str,
    rng: random.Random | None = None,
) -> list[GameEvent]:
    first_visit = actor.player_id not in state.location_state[new_loc].visited_by

    actor.location_id = new_loc
    actor.travel_from = None
    actor.travel_to = None
    actor.travel_direction = None
    actor.travel_remaining = 0
    state.location_state[new_loc].visited_by.add(actor.player_id)

    events = [GameEvent(
        round=state.round,
        kind="move",
        actor_ids=[actor.player_id],
        payload={"from": old_loc, "to": new_loc, "direction": direction},
        visibility="public",
        narrative_seed=f"{actor.player_name} arrived at {scenario.locations[new_loc].name}.",
    )]

    if first_visit:
        events.extend(_grant_arrival_knowledge(state, scenario, actor.player_id, new_loc))

    events.extend(trigger_location_points(state, scenario, actor.player_id, "arrive", rng))

    return events


def _travel_cost(scenario: ScenarioDef, loc: Location, direction: str) -> int:
    return loc.travel_costs.get(direction, 1)


def _travel_cost_between(scenario: ScenarioDef, origin: str, destination: str) -> int:
    loc = scenario.locations.get(origin)
    if loc is None:
        return 1
    for direction, target in loc.connections.items():
        if target == destination:
            return loc.travel_costs.get(direction, 1)
    return 1


def _reverse_direction(scenario: ScenarioDef, origin: str, destination: str) -> str:
    loc = scenario.locations.get(origin)
    if loc is None:
        return ""
    for direction, target in loc.connections.items():
        if target == destination:
            return direction
    return ""


def search_location(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
    rng: random.Random,
) -> list[GameEvent]:
    actor = state.actors[player_id]
    loc = scenario.locations[actor.location_id]
    loc_rt = state.location_state[actor.location_id]
    events: list[GameEvent] = []

    events.extend(trigger_location_points(state, scenario, player_id, "search", rng))

    # Knowledge is learnable even after loot is depleted — research continues.
    for src in scenario.knowledge_sources:
        if src.acquisition != "search":
            continue
        if src.location_id != actor.location_id:
            continue
        if rng.random() <= src.weight:
            evt = grant_knowledge(state, scenario, player_id, src.id)
            if evt:
                events.append(evt)

    available = [
        d for d in loc.searchable_items
        if loc_rt.remaining_loot.get(d.item_id, 0) > 0
    ]
    if not available:
        events.append(GameEvent(
            round=state.round,
            kind="search_empty",
            actor_ids=[player_id],
            payload={"location": actor.location_id},
            visibility="private",
            private_to=player_id,
            narrative_seed=f"{actor.player_name} searched {loc.name} but found nothing.",
        ))
        return events

    total_weight = sum(d.weight for d in available)
    picks = max(1, rng.randint(1, 3))

    for _ in range(picks):
        if not available:
            break
        roll = rng.uniform(0, total_weight)
        cumulative = 0.0
        chosen = available[-1]
        for d in available:
            cumulative += d.weight
            if roll <= cumulative:
                chosen = d
                break

        if loc_rt.remaining_loot.get(chosen.item_id, 0) > 0:
            loc_rt.remaining_loot[chosen.item_id] -= 1
            actor.inventory[chosen.item_id] = actor.inventory.get(chosen.item_id, 0) + 1
            item = scenario.items[chosen.item_id]
            events.append(GameEvent(
                round=state.round,
                kind="search_found",
                actor_ids=[player_id],
                payload={"item_id": chosen.item_id, "item_name": item.name},
                visibility="private",
                private_to=player_id,
                narrative_seed=f"{actor.player_name} found {item.name} at {loc.name}.",
            ))

            if loc_rt.remaining_loot[chosen.item_id] <= 0:
                available = [d for d in available if d.item_id != chosen.item_id]
                total_weight = sum(d.weight for d in available) if available else 0

    stamina_cost = 5
    actor.stats.stamina = max(0, actor.stats.stamina - stamina_cost)

    return events


def use_item(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
    item_id: str,
) -> GameEvent | None:
    actor = state.actors[player_id]
    if actor.inventory.get(item_id, 0) <= 0:
        return None

    item = scenario.items.get(item_id)
    if item is None or item.kind != "consumable":
        return None

    heal = item.stats.get("heal", 0)
    if heal > 0:
        actor.stats.hp = min(actor.stats.max_hp, actor.stats.hp + heal)

    stamina_restore = item.stats.get("stamina", 0)
    if stamina_restore > 0:
        actor.stats.stamina = min(actor.stats.max_stamina, actor.stats.stamina + stamina_restore)

    if item.stats.get("cure_poison"):
        actor.status_effects = [e for e in actor.status_effects if e.kind != "poison"]

    actor.inventory[item_id] -= 1
    if actor.inventory[item_id] <= 0:
        del actor.inventory[item_id]

    return GameEvent(
        round=state.round,
        kind="use_item",
        actor_ids=[player_id],
        payload={"item_id": item_id, "item_name": item.name, "heal": heal},
        visibility="private",
        private_to=player_id,
        narrative_seed=f"{actor.player_name} used {item.name}.",
    )


def craft_item(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
    recipe_index: int,
) -> list[GameEvent]:
    actor = state.actors[player_id]
    if recipe_index < 0 or recipe_index >= len(scenario.recipes):
        return []

    recipe = scenario.recipes[recipe_index]

    for input_id, needed in recipe.inputs.items():
        if actor.inventory.get(input_id, 0) < needed:
            return []

    if recipe.requires_location_tag:
        loc = scenario.locations[actor.location_id]
        if recipe.requires_location_tag not in loc.tags:
            return []

    for input_id, needed in recipe.inputs.items():
        actor.inventory[input_id] -= needed
        if actor.inventory[input_id] <= 0:
            del actor.inventory[input_id]

    actor.inventory[recipe.output_item_id] = actor.inventory.get(recipe.output_item_id, 0) + 1

    out_item = scenario.items[recipe.output_item_id]
    events = [GameEvent(
        round=state.round,
        kind="craft",
        actor_ids=[player_id],
        payload={"output": recipe.output_item_id, "output_name": out_item.name},
        visibility="private",
        private_to=player_id,
        narrative_seed=f"{actor.player_name} crafted {out_item.name}.",
    )]

    for src in scenario.knowledge_sources:
        if src.acquisition == "craft" and src.item_id == recipe.output_item_id:
            evt = grant_knowledge(state, scenario, player_id, src.id)
            if evt:
                events.append(evt)

    return events


def attack_actor(
    state: GameState,
    scenario: ScenarioDef,
    attacker_id: str,
    target_id: str,
) -> GameEvent | None:
    attacker = state.actors[attacker_id]
    target = state.actors[target_id]

    if attacker.traveling or target.traveling:
        return None
    if attacker.location_id != target.location_id:
        return None
    if not target.alive:
        return None

    atk_power = attacker.stats.attack
    if attacker.equipped_weapon:
        weapon = scenario.items.get(attacker.equipped_weapon)
        if weapon:
            atk_power += weapon.stats.get("attack", 0)

    def_power = target.stats.defense
    if target.equipped_armor:
        armor = scenario.items.get(target.equipped_armor)
        if armor:
            def_power += armor.stats.get("defense", 0)

    base_damage = max(1, atk_power - def_power // 2)
    variance = max(1, base_damage // 4)
    damage = base_damage + random.randint(-variance, variance)
    damage = max(1, damage)

    target.stats.hp -= damage
    killed = False
    if target.stats.hp <= 0:
        target.stats.hp = 0
        target.alive = False
        killed = True

    weapon_name = attacker.equipped_weapon or "bare hands"
    return GameEvent(
        round=state.round,
        kind="attack",
        actor_ids=[attacker_id, target_id],
        payload={
            "attacker": attacker_id,
            "target": target_id,
            "damage": damage,
            "weapon": weapon_name,
            "killed": killed,
            "target_hp": target.stats.hp,
        },
        visibility="public",
        narrative_seed=(
            f"{attacker.player_name} attacked {target.player_name} for {damage} damage"
            f"{' and killed them' if killed else ''}."
        ),
    )


def rest_actor(
    state: GameState,
    player_id: str,
) -> GameEvent:
    actor = state.actors[player_id]
    stamina_restore = 15
    hp_restore = 5
    actor.stats.stamina = min(actor.stats.max_stamina, actor.stats.stamina + stamina_restore)
    actor.stats.hp = min(actor.stats.max_hp, actor.stats.hp + hp_restore)

    return GameEvent(
        round=state.round,
        kind="rest",
        actor_ids=[player_id],
        payload={"stamina_restored": stamina_restore, "hp_restored": hp_restore},
        visibility="private",
        private_to=player_id,
        narrative_seed=f"{actor.player_name} rested for a moment.",
    )


def equip_item(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
    item_id: str,
) -> GameEvent | None:
    actor = state.actors[player_id]
    if actor.inventory.get(item_id, 0) <= 0:
        return None

    item = scenario.items.get(item_id)
    if item is None:
        return None

    if item.kind == "weapon":
        if actor.equipped_weapon:
            actor.inventory[actor.equipped_weapon] = actor.inventory.get(actor.equipped_weapon, 0) + 1
        actor.equipped_weapon = item_id
        actor.inventory[item_id] -= 1
        if actor.inventory[item_id] <= 0:
            del actor.inventory[item_id]
    elif item.kind == "armor":
        if actor.equipped_armor:
            actor.inventory[actor.equipped_armor] = actor.inventory.get(actor.equipped_armor, 0) + 1
        actor.equipped_armor = item_id
        actor.inventory[item_id] -= 1
        if actor.inventory[item_id] <= 0:
            del actor.inventory[item_id]
    else:
        return None

    return GameEvent(
        round=state.round,
        kind="equip",
        actor_ids=[player_id],
        payload={"item_id": item_id, "item_name": item.name, "slot": item.kind},
        visibility="private",
        private_to=player_id,
        narrative_seed=f"{actor.player_name} equipped {item.name}.",
    )


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------

def grant_knowledge(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
    source_id: str,
) -> GameEvent | None:
    source = next(
        (s for s in scenario.knowledge_sources if s.id == source_id), None
    )
    if source is None:
        return None

    actor = state.actors[player_id]
    if source_id in actor.learned_sources:
        return None

    actor.learned_sources.add(source_id)
    for flag in source.grants_knowledge:
        actor.knowledge.add(flag)

    for stat, amount in source.stat_bonus.items():
        if hasattr(actor.stats, stat):
            current = getattr(actor.stats, stat)
            setattr(actor.stats, stat, current + amount)

    return GameEvent(
        round=state.round,
        kind="knowledge_gained",
        actor_ids=[player_id],
        payload={
            "source_id": source.id,
            "source_name": source.name,
            "knowledge": source.grants_knowledge,
            "stat_bonus": source.stat_bonus,
        },
        visibility="private",
        private_to=player_id,
        narrative_seed=f"{actor.player_name} learned: {source.name}.",
    )


def _grant_arrival_knowledge(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
    location_id: str,
) -> list[GameEvent]:
    events: list[GameEvent] = []
    for src in scenario.knowledge_sources:
        if src.acquisition == "arrive" and src.location_id == location_id:
            evt = grant_knowledge(state, scenario, player_id, src.id)
            if evt:
                events.append(evt)
    return events


# ---------------------------------------------------------------------------
# Status effects & round-end processing
# ---------------------------------------------------------------------------

def tick_status_effects(state: GameState) -> list[GameEvent]:
    events: list[GameEvent] = []
    for pid, actor in state.actors.items():
        if not actor.alive:
            continue
        surviving: list[StatusEffect] = []
        for eff in actor.status_effects:
            if eff.kind == "poison":
                dmg = eff.magnitude or 3
                actor.stats.hp -= dmg
                if actor.stats.hp <= 0:
                    actor.stats.hp = 0
                    actor.alive = False
                events.append(GameEvent(
                    round=state.round,
                    kind="status_damage",
                    actor_ids=[pid],
                    payload={"effect": "poison", "damage": dmg, "hp": actor.stats.hp},
                    visibility="private",
                    private_to=pid,
                    narrative_seed=f"{actor.player_name} took {dmg} poison damage.",
                ))
            eff.rounds_left -= 1
            if eff.rounds_left > 0:
                surviving.append(eff)
        actor.status_effects = surviving
    return events


def apply_danger_damage(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
) -> GameEvent | None:
    actor = state.actors[player_id]
    if actor.traveling:
        return None
    loc = scenario.locations[actor.location_id]
    if loc.danger_level <= 0:
        return None

    dmg = loc.danger_level
    actor.stats.hp -= dmg
    killed = False
    if actor.stats.hp <= 0:
        actor.stats.hp = 0
        actor.alive = False
        killed = True

    return GameEvent(
        round=state.round,
        kind="danger_damage",
        actor_ids=[player_id],
        payload={"location": loc.id, "danger": loc.danger_level, "damage": dmg, "killed": killed},
        visibility="private",
        private_to=player_id,
        narrative_seed=f"{actor.player_name} suffered {dmg} danger damage at {loc.name}.",
    )


# ---------------------------------------------------------------------------
# Player view (fog of war)
# ---------------------------------------------------------------------------

def compute_player_view(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
) -> PlayerView:
    actor = state.actors[player_id]
    char_def = _get_char_def(scenario, actor)

    visible_actors: list[VisibleActor] = []
    if not actor.traveling:
        for pid, other in state.actors.items():
            if pid == player_id or not other.alive or other.traveling:
                continue
            same_faction = bool(actor.faction_id) and other.faction_id == actor.faction_id
            if same_faction or other.location_id == actor.location_id:
                visible_actors.append(VisibleActor(
                    player_id=pid,
                    player_name=other.player_name,
                    character_name=_get_char_def(scenario, other).name,
                    location_id=other.location_id,
                    stats_summary={"hp": other.stats.hp, "max_hp": other.stats.max_hp},
                ))

    legal = compute_legal_actions(state, scenario, player_id)

    if actor.traveling:
        origin = scenario.locations[actor.travel_from or actor.location_id].name
        destination = scenario.locations[actor.travel_to or actor.location_id].name
        loc_name = "途中"
        loc_desc = f"正在从 {origin} 前往 {destination}（还剩 {actor.travel_remaining} 回合）"
        connections: dict[str, str] = {}
    else:
        loc = scenario.locations[actor.location_id]
        loc_name = loc.name
        loc_desc = loc.description
        connections = dict(loc.connections)

    faction = scenario.factions.get(actor.faction_id)
    secret_objective = ""
    if char_def is not None and char_def.personal_objective is not None:
        secret_objective = char_def.personal_objective.description

    return PlayerView(
        round=state.round,
        phase=state.phase,
        location_id=actor.location_id,
        location_name=loc_name,
        location_description=loc_desc,
        connections=connections,
        stats=actor.stats.model_copy(),
        inventory=dict(actor.inventory),
        equipped_weapon=actor.equipped_weapon,
        equipped_armor=actor.equipped_armor,
        status_effects=list(actor.status_effects),
        visible_actors=visible_actors,
        legal_actions=legal,
        secret_objective=secret_objective,
        faction_id=actor.faction_id,
        faction_name=faction.name if faction else "",
        faction_objective=faction.objective.description if faction and faction.objective else "",
        alive=actor.alive,
        map=_build_map(scenario),
        travel_from=actor.travel_from,
        travel_to=actor.travel_to,
        travel_remaining=actor.travel_remaining,
    )


def compute_legal_actions(
    state: GameState,
    scenario: ScenarioDef,
    player_id: str,
) -> list[LegalAction]:
    actor = state.actors[player_id]
    if not actor.alive:
        return []

    if actor.traveling:
        return [
            LegalAction(
                type=ActionType.WAIT,
                label="继续赶路 (Continue traveling)",
                params_schema={},
            ),
            LegalAction(
                type=ActionType.CANCEL,
                label="折返 (Turn back)",
                params_schema={},
            ),
        ]

    actions: list[LegalAction] = []
    loc = scenario.locations[actor.location_id]
    loc_rt = state.location_state[actor.location_id]

    for direction, target_id in loc.connections.items():
        target_name = scenario.locations[target_id].name
        cost = _travel_cost(scenario, loc, direction)
        cost_label = f" ({cost}回合)" if cost > 1 else ""
        actions.append(LegalAction(
            type=ActionType.MOVE,
            label=f"Move {direction} → {target_name}{cost_label}",
            params_schema={"direction": direction},
        ))

    has_loot = any(
        loc_rt.remaining_loot.get(d.item_id, 0) > 0
        for d in loc.searchable_items
    )
    if has_loot:
        actions.append(LegalAction(
            type=ActionType.SEARCH,
            label=f"Search {loc.name}",
            params_schema={},
        ))

    for item_id, count in actor.inventory.items():
        if count <= 0:
            continue
        item = scenario.items.get(item_id)
        if item is None:
            continue
        if item.kind == "consumable":
            actions.append(LegalAction(
                type=ActionType.USE,
                label=f"Use {item.name}",
                params_schema={"item_id": item_id},
            ))
        elif item.kind in ("weapon", "armor"):
            actions.append(LegalAction(
                type=ActionType.USE,
                label=f"Equip {item.name}",
                params_schema={"item_id": item_id, "equip": True},
            ))

    for idx, recipe in enumerate(scenario.recipes):
        if not _can_craft(actor, recipe, loc):
            continue
        out_item = scenario.items.get(recipe.output_item_id)
        out_name = out_item.name if out_item else recipe.output_item_id
        actions.append(LegalAction(
            type=ActionType.CRAFT,
            label=f"Craft {out_name}",
            params_schema={"recipe_index": idx},
        ))

    for pid, other in state.actors.items():
        if pid == player_id or not other.alive or other.traveling:
            continue
        if other.location_id == actor.location_id:
            actions.append(LegalAction(
                type=ActionType.ATTACK,
                label=f"Attack {other.player_name}",
                params_schema={"target": pid},
            ))

    char_def = _get_char_def(scenario, actor)
    if char_def:
        opponents_here = any(
            pid != player_id and other.alive and not other.traveling
            and other.location_id == actor.location_id
            for pid, other in state.actors.items()
        )
        for ability in char_def.abilities:
            cooldown = actor.ability_cooldowns.get(ability.id, 0)
            if cooldown > 0:
                continue
            if actor.stats.stamina < ability.stamina_cost:
                continue
            if ability.target == "single" and not opponents_here:
                continue
            actions.append(LegalAction(
                type=ActionType.SPECIAL,
                label=f"Ability: {ability.name} ({ability.stamina_cost} stamina)",
                params_schema={"ability_id": ability.id, "target": None},
            ))

    actions.append(LegalAction(
        type=ActionType.REST,
        label="Rest (recover stamina + small heal)",
        params_schema={},
    ))

    actions.append(LegalAction(
        type=ActionType.WAIT,
        label="Wait / Do nothing",
        params_schema={},
    ))

    return actions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_char_def(scenario: ScenarioDef, actor: Actor) -> CharacterDef | None:
    for c in scenario.characters:
        if c.id == actor.character_id:
            return c
    return None


def _build_map(scenario: ScenarioDef) -> list[MapLocation]:
    out: list[MapLocation] = []
    for loc in scenario.locations.values():
        out.append(MapLocation(
            id=loc.id,
            name=loc.name,
            coord=list(loc.coord) if loc.coord else [],
            danger_level=loc.danger_level,
            connections=dict(loc.connections),
            travel_costs=dict(loc.travel_costs),
        ))
    return out


def _move_stamina_cost(scenario: ScenarioDef, from_loc: Location) -> int:
    base = 3
    danger = from_loc.danger_level
    return base + danger


def _can_craft(actor: Actor, recipe, loc: Location) -> bool:
    for input_id, needed in recipe.inputs.items():
        if actor.inventory.get(input_id, 0) < needed:
            return False
    if recipe.requires_location_tag:
        if recipe.requires_location_tag not in loc.tags:
            return False
    return True


def get_actors_at_location(state: GameState, location_id: str) -> list[Actor]:
    return [
        a for a in state.actors.values()
        if a.location_id == location_id and a.alive and not a.traveling
    ]


def get_alive_actors(state: GameState) -> list[Actor]:
    return [a for a in state.actors.values() if a.alive]
