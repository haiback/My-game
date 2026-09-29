"""Domain models for the multiplayer text adventure duel game.

All models use Pydantic v2. Static definitions (Scenario, Character, etc.)
are loaded from YAML. Runtime state (GameState, Actor, etc.) is managed
by the game engine.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Static definitions (loaded from scenario YAML)
# ---------------------------------------------------------------------------

class Stats(BaseModel):
    hp: int = 100
    max_hp: int = 100
    stamina: int = 100
    max_stamina: int = 100
    attack: int = 10
    defense: int = 5
    speed: int = 5


class Item(BaseModel):
    id: str
    name: str
    kind: Literal["material", "weapon", "armor", "consumable", "key"]
    stats: dict[str, int] = Field(default_factory=dict)
    stackable: bool = False
    description: str = ""


class ItemDrop(BaseModel):
    item_id: str
    weight: float = 1.0
    max_count: int = 1


class LocationPoint(BaseModel):
    """A scripted special at a location: trap, NPC encounter, vision, hazard.

    `trigger` selects which engine hook fires it (arrive / search / stay).
    `effects` reuse the generic effect schema. `criticality` feeds the
    three-axis criticality scorer so these moments get LLM narration.
    """

    id: str
    name: str = ""
    kind: Literal["trap", "npc_encounter", "vision", "hazard"] = "trap"
    trigger: Literal["arrive", "search", "stay"] = "arrive"
    chance: float = 1.0
    repeatable: bool = False
    effects: list[dict] = Field(default_factory=list)
    narrative_seed: str = ""
    criticality: float = 0.8


class Location(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    connections: dict[str, str] = Field(default_factory=dict)
    travel_costs: dict[str, int] = Field(default_factory=dict)
    coord: list[int] | None = None
    searchable_items: list[ItemDrop] = Field(default_factory=list)
    danger_level: int = 0
    points: list[LocationPoint] = Field(default_factory=list)


class Recipe(BaseModel):
    output_item_id: str
    inputs: dict[str, int]
    requires_location_tag: str | None = None


class KnowledgeSource(BaseModel):
    """A piece of learnable knowledge defined by the scenario.

    Acquisition paths (all data-driven, scenario-specific):
      - search:  learned by searching a location (weight = chance per search)
      - arrive:  learned on first entering a location
      - craft:   learned when crafting a specific item
      - event:   granted by scenario events / character abilities

    stat_bonus allows knowledge to also confer permanent skill gains
    (e.g. a cyberpunk brain implant granting +attack on install).
    """

    id: str
    name: str
    description: str = ""
    acquisition: Literal["search", "arrive", "craft", "event"] = "search"
    location_id: str | None = None
    item_id: str | None = None
    weight: float = 1.0
    grants_knowledge: list[str] = Field(default_factory=list)
    stat_bonus: dict[str, int] = Field(default_factory=dict)


class AbilityVariant(BaseModel):
    """A conditional alternate form of an ability.

    When an ability is used, the first variant whose `condition` matches
    replaces the base `effects`. Conditions are data-driven dicts evaluated
    against the actor/target/location/global state (see engine.conditions).
    """

    name: str = ""
    condition: dict = Field(default_factory=dict)
    effects: list[dict] = Field(default_factory=list)


class AbilityDef(BaseModel):
    """A character-specific active ability.

    Effects use the same generic effect schema as random events, so a
    scenario can express anything from a combat skill to an NPC-role
    ability like "dispatch monsters" (a trigger_event effect).
    """

    id: str
    name: str
    description: str = ""
    stamina_cost: int = 0
    cooldown_rounds: int = 0
    target: Literal["none", "self", "single", "location"] = "none"
    effects: list[dict] = Field(default_factory=list)
    variants: list[AbilityVariant] = Field(default_factory=list)


class VictoryCondition(BaseModel):
    kind: Literal[
        "survive_rounds",
        "eliminate_all",
        "collect_items",
        "reach_location",
        "escape",
        "custom_flag",
    ]
    params: dict = Field(default_factory=dict)


class Objective(BaseModel):
    """A data-driven goal: trigger condition → outcome.

    `trigger` reuses VictoryCondition kinds (items/location/flags/rounds/
    eliminate_all). `effects` are generic effects (stat_bonus / grant_item /
    equip / trigger_event ...). `victory=True` means achieving it also wins —
    winning is only one possible outcome (others: stat boost, new weapon, ...).
    """

    id: str = ""
    name: str = ""
    description: str = ""
    trigger: VictoryCondition
    effects: list[dict] = Field(default_factory=list)
    victory: bool = False


class FactionDef(BaseModel):
    id: str
    name: str
    is_npc: bool = False
    objective: Objective | None = None


class CriticalityTrigger(BaseModel):
    """Data-driven rule that makes an event more 'critical' for this character's
    faction (identity/goal relevance). Matches on kind/item/location; `weight`
    is added to that faction's round criticality."""

    kind: str | None = None
    item_id: str | None = None
    location_id: str | None = None
    weight: float = 0.8


class CharacterDef(BaseModel):
    id: str
    name: str
    backstory: str
    narration_style: str = ""
    focus: str = ""
    base_stats: Stats
    start_location: str
    start_inventory: dict[str, int] = Field(default_factory=dict)
    faction_id: str = ""
    personal_objective: Objective | None = None
    criticality_triggers: list[CriticalityTrigger] = Field(default_factory=list)
    abilities: list[AbilityDef] = Field(default_factory=list)


class RandomEvent(BaseModel):
    id: str
    name: str
    description: str
    round_trigger: int | None = None
    weight: float = 1.0
    effects: list[dict] = Field(default_factory=list)
    narrative_seed: str = ""
    repeatable: bool = False
    max_fires: int = 1


class ScenarioDef(BaseModel):
    id: str
    name: str
    setting: str
    intro_text: str
    player_range: tuple[int, int] = (2, 4)
    round_time_seconds: int = 45
    max_rounds: int = 30
    locations: dict[str, Location] = Field(default_factory=dict)
    items: dict[str, Item] = Field(default_factory=dict)
    recipes: list[Recipe] = Field(default_factory=list)
    knowledge_sources: list[KnowledgeSource] = Field(default_factory=list)
    characters: list[CharacterDef] = Field(default_factory=list)
    factions: dict[str, FactionDef] = Field(default_factory=dict)
    random_events: list[RandomEvent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Runtime models (managed by game engine)
# ---------------------------------------------------------------------------

class StatusEffect(BaseModel):
    kind: str
    rounds_left: int
    magnitude: int = 0


class Actor(BaseModel):
    character_id: str
    player_id: str
    player_name: str
    faction_id: str = ""
    stats: Stats
    location_id: str
    inventory: dict[str, int] = Field(default_factory=dict)
    equipped_weapon: str | None = None
    equipped_armor: str | None = None
    status_effects: list[StatusEffect] = Field(default_factory=list)
    alive: bool = True
    knowledge: set[str] = Field(default_factory=set)
    learned_sources: set[str] = Field(default_factory=set)
    ability_cooldowns: dict[str, int] = Field(default_factory=dict)
    objective_progress: dict = Field(default_factory=dict)
    travel_from: str | None = None
    travel_to: str | None = None
    travel_direction: str | None = None
    travel_remaining: int = 0

    @property
    def traveling(self) -> bool:
        return self.travel_remaining > 0


class LocationRuntime(BaseModel):
    remaining_loot: dict[str, int] = Field(default_factory=dict)
    visited_by: set[str] = Field(default_factory=set)
    flags: set[str] = Field(default_factory=set)


class GameEvent(BaseModel):
    round: int
    kind: str
    actor_ids: list[str] = Field(default_factory=list)
    payload: dict = Field(default_factory=dict)
    visibility: Literal["public", "private"] = "public"
    private_to: str | None = None
    narrative_seed: str = ""


class GamePhase(StrEnum):
    LOBBY = "lobby"
    NARRATIVE = "narrative"
    DECISION = "decision"
    RESOLUTION = "resolution"
    ENDED = "ended"


class GameState(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    scenario_id: str
    round: int = 0
    phase: GamePhase = GamePhase.LOBBY
    actors: dict[str, Actor] = Field(default_factory=dict)
    location_state: dict[str, LocationRuntime] = Field(default_factory=dict)
    global_flags: set[str] = Field(default_factory=set)
    fired_events: dict[str, int] = Field(default_factory=dict)
    event_log: list[GameEvent] = Field(default_factory=list)
    settled_objectives: set[str] = Field(default_factory=set)
    winner_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class ActionType(StrEnum):
    MOVE = "move"
    SEARCH = "search"
    USE = "use"
    CRAFT = "craft"
    ATTACK = "attack"
    REST = "rest"
    TALK = "talk"
    WAIT = "wait"
    SPECIAL = "special"
    CANCEL = "cancel"


class Action(BaseModel):
    player_id: str
    type: ActionType
    params: dict = Field(default_factory=dict)
    raw_text: str | None = None
    parse_confidence: float = 1.0


# ---------------------------------------------------------------------------
# Player view (sent to client each round)
# ---------------------------------------------------------------------------

class VisibleActor(BaseModel):
    player_id: str
    player_name: str
    character_name: str
    location_id: str
    stats_summary: dict


class LegalAction(BaseModel):
    type: ActionType
    label: str
    params_schema: dict = Field(default_factory=dict)


class MapLocation(BaseModel):
    """Public map layout entry sent to clients for rendering."""

    id: str
    name: str
    coord: list[int] = Field(default_factory=list)
    danger_level: int = 0
    connections: dict[str, str] = Field(default_factory=dict)
    travel_costs: dict[str, int] = Field(default_factory=dict)


class PlayerView(BaseModel):
    round: int
    phase: GamePhase
    location_id: str
    location_name: str
    location_description: str
    connections: dict[str, str]
    stats: Stats
    inventory: dict[str, int]
    equipped_weapon: str | None = None
    equipped_armor: str | None = None
    status_effects: list[StatusEffect] = Field(default_factory=list)
    visible_actors: list[VisibleActor] = Field(default_factory=list)
    legal_actions: list[LegalAction] = Field(default_factory=list)
    secret_objective: str = ""
    faction_id: str = ""
    faction_name: str = ""
    faction_objective: str = ""
    alive: bool = True
    map: list[MapLocation] = Field(default_factory=list)
    travel_from: str | None = None
    travel_to: str | None = None
    travel_remaining: int = 0
