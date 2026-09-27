"""Tests for knowledge acquisition, random events, and character abilities."""

import random
from pathlib import Path

import pytest

from mygame.server.engine.effects import fire_event
from mygame.server.engine.rules import resolve_round
from mygame.server.engine.state import (
    PlayerAssignment,
    compute_legal_actions,
    craft_item,
    grant_knowledge,
    init_game,
    move_actor,
    search_location,
)
from mygame.server.engine.winconditions import check_victory
from mygame.server.scenario.loader import ScenarioLoadError, ScenarioRegistry
from mygame.shared.models import (
    AbilityDef,
    Action,
    ActionType,
    CharacterDef,
    Item,
    KnowledgeSource,
    Location,
    RandomEvent,
    Recipe,
    ScenarioDef,
    Stats,
    VictoryCondition,
)

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


@pytest.fixture
def island_scenario():
    return ScenarioRegistry().load_file(SCENARIOS_DIR / "island_of_whispers.yaml")


# ---------------------------------------------------------------------------
# Mini scenario builder for deterministic tests
# ---------------------------------------------------------------------------

def make_char(
    cid: str,
    start_location: str = "a",
    abilities: list[AbilityDef] | None = None,
) -> CharacterDef:
    return CharacterDef(
        id=cid,
        name=cid.upper(),
        backstory="",
        base_stats=Stats(),
        start_location=start_location,
        secret_objective="",
        victory_condition=VictoryCondition(kind="survive_rounds", params={"rounds": 99}),
        abilities=abilities or [],
    )


def make_scenario(
    characters: list[CharacterDef] | None = None,
    knowledge_sources: list[KnowledgeSource] | None = None,
    random_events: list[RandomEvent] | None = None,
    locations: dict[str, Location] | None = None,
    items: dict[str, Item] | None = None,
    recipes: list[Recipe] | None = None,
) -> ScenarioDef:
    return ScenarioDef(
        id="mini",
        name="Mini",
        setting="",
        intro_text="",
        locations=locations or {
            "a": Location(id="a", name="A", description="", connections={"north": "b"}),
            "b": Location(id="b", name="B", description="", connections={"south": "a"}),
        },
        items=items or {
            "herb": Item(id="herb", name="Herb", kind="consumable", stats={"heal": 10}),
            "cloth": Item(id="cloth", name="Cloth", kind="material"),
            "bandage": Item(id="bandage", name="Bandage", kind="consumable", stats={"heal": 25}),
        },
        recipes=recipes or [Recipe(output_item_id="bandage", inputs={"cloth": 2})],
        knowledge_sources=knowledge_sources or [],
        characters=characters or [make_char("c1"), make_char("c2", start_location="b")],
        random_events=random_events or [],
    )


def init_two_players(scenario: ScenarioDef, location: str = "a") -> tuple:
    players = [
        PlayerAssignment(player_id="p1", player_name="P1", character_id="c1"),
        PlayerAssignment(player_id="p2", player_name="P2", character_id="c2"),
    ]
    state = init_game(scenario, players)
    state.actors["p1"].location_id = location
    state.actors["p2"].location_id = location
    return state, scenario


# ---------------------------------------------------------------------------
# Knowledge acquisition
# ---------------------------------------------------------------------------

class TestKnowledgeAcquisition:
    def test_grant_knowledge_adds_flags_and_stat_bonus(self):
        scenario = make_scenario(knowledge_sources=[
            KnowledgeSource(
                id="chip", name="Neural Chip", acquisition="event",
                grants_knowledge=["skill_x"], stat_bonus={"attack": 5},
            ),
        ])
        state, _ = init_two_players(scenario)

        evt = grant_knowledge(state, scenario, "p1", "chip")
        assert evt is not None
        assert evt.kind == "knowledge_gained"
        actor = state.actors["p1"]
        assert "skill_x" in actor.knowledge
        assert actor.stats.attack == 15  # 10 base + 5

        # second grant is a no-op (no double stat bonus)
        assert grant_knowledge(state, scenario, "p1", "chip") is None
        assert actor.stats.attack == 15

    def test_search_knowledge_acquisition(self):
        scenario = make_scenario(knowledge_sources=[
            KnowledgeSource(
                id="insect_a", name="Insect A", acquisition="search",
                location_id="a", weight=1.0, grants_knowledge=["insect_a"],
            ),
        ])
        state, _ = init_two_players(scenario)

        events = search_location(state, scenario, "p1", random.Random(42))
        assert any(e.kind == "knowledge_gained" for e in events)
        assert "insect_a" in state.actors["p1"].knowledge

    def test_search_knowledge_respects_weight_zero(self):
        scenario = make_scenario(knowledge_sources=[
            KnowledgeSource(
                id="insect_a", name="Insect A", acquisition="search",
                location_id="a", weight=0.0, grants_knowledge=["insect_a"],
            ),
        ])
        state, _ = init_two_players(scenario)

        for _ in range(5):
            search_location(state, scenario, "p1", random.Random(7))
        assert "insect_a" not in state.actors["p1"].knowledge

    def test_craft_knowledge_acquisition(self):
        scenario = make_scenario(knowledge_sources=[
            KnowledgeSource(
                id="first_aid", name="First Aid", acquisition="craft",
                item_id="bandage", grants_knowledge=["first_aid"],
            ),
        ])
        state, _ = init_two_players(scenario)
        state.actors["p1"].inventory["cloth"] = 2

        events = craft_item(state, scenario, "p1", 0)
        assert any(e.kind == "craft" for e in events)
        assert any(e.kind == "knowledge_gained" for e in events)
        assert "first_aid" in state.actors["p1"].knowledge

    def test_arrival_knowledge_on_first_visit_only(self):
        scenario = make_scenario(knowledge_sources=[
            KnowledgeSource(
                id="ruins", name="Ruins", acquisition="arrive",
                location_id="b", grants_knowledge=["ruins_lore"],
            ),
        ])
        state, _ = init_two_players(scenario)

        events = move_actor(state, scenario, "p1", "north")
        assert any(e.kind == "knowledge_gained" for e in events)
        assert "ruins_lore" in state.actors["p1"].knowledge

        # leave and return: no second grant
        move_actor(state, scenario, "p1", "south")
        events = move_actor(state, scenario, "p1", "north")
        assert not any(e.kind == "knowledge_gained" for e in events)

    def test_entomologist_victory_path(self, island_scenario):
        players = [
            PlayerAssignment(player_id="p1", player_name="Lin", character_id="entomologist"),
        ]
        state = init_game(island_scenario, players)
        assert check_victory(state, island_scenario) == []

        for src_id in ["insect_alpha", "insect_beta", "insect_gamma"]:
            grant_knowledge(state, island_scenario, "p1", src_id)

        assert check_victory(state, island_scenario) == ["p1"]


# ---------------------------------------------------------------------------
# Random events
# ---------------------------------------------------------------------------

class TestRandomEvents:
    def make_event_scenario(self) -> ScenarioDef:
        return make_scenario(random_events=[
            RandomEvent(
                id="quake", name="Quake", description="",
                round_trigger=1, weight=1.0,
                effects=[{"type": "damage", "target": "all", "amount": 5}],
                narrative_seed="The ground shakes.",
            ),
        ])

    def test_event_fires_at_trigger_round(self):
        scenario = self.make_event_scenario()
        state, _ = init_two_players(scenario)
        state.round = 1

        hp_before = {pid: a.stats.hp for pid, a in state.actors.items()}
        events = resolve_round(
            state, scenario,
            {"p1": Action(player_id="p1", type=ActionType.WAIT),
             "p2": Action(player_id="p2", type=ActionType.WAIT)},
            round_seed=1,
        )

        assert any(e.kind == "random_event" and e.payload["event_id"] == "quake" for e in events)
        for pid, hp in hp_before.items():
            assert state.actors[pid].stats.hp == hp - 5
        assert state.fired_events["quake"] == 1

    def test_event_not_fired_before_trigger_round(self):
        scenario = self.make_event_scenario()
        state, _ = init_two_players(scenario)
        state.round = 0

        events = resolve_round(
            state, scenario,
            {"p1": Action(player_id="p1", type=ActionType.WAIT),
             "p2": Action(player_id="p2", type=ActionType.WAIT)},
            round_seed=1,
        )
        assert not any(e.kind == "random_event" for e in events)

    def test_event_respects_max_fires(self):
        scenario = self.make_event_scenario()
        state, _ = init_two_players(scenario)
        state.round = 1

        resolve_round(state, scenario, {
            "p1": Action(player_id="p1", type=ActionType.WAIT),
            "p2": Action(player_id="p2", type=ActionType.WAIT),
        }, round_seed=1)

        state.round = 2
        events = resolve_round(state, scenario, {
            "p1": Action(player_id="p1", type=ActionType.WAIT),
            "p2": Action(player_id="p2", type=ActionType.WAIT),
        }, round_seed=2)
        assert not any(e.kind == "random_event" for e in events)
        assert state.fired_events["quake"] == 1

    def test_fire_event_forced_bypasses_round_trigger(self):
        scenario = make_scenario(random_events=[
            RandomEvent(
                id="ambush", name="Ambush", description="",
                round_trigger=None, max_fires=5,
                effects=[{"type": "damage", "target": "single", "amount": 9}],
                narrative_seed="Ambush!",
            ),
        ])
        state, _ = init_two_players(scenario)

        events = fire_event(
            state, scenario, "ambush",
            context={"source": "p1", "target": "p2", "location": "a"},
            rng=random.Random(1), force=True,
        )
        assert any(e.kind == "random_event" for e in events)
        assert state.actors["p2"].stats.hp == 91


# ---------------------------------------------------------------------------
# Abilities
# ---------------------------------------------------------------------------

class TestAbilities:
    def strike_scenario(self) -> ScenarioDef:
        return make_scenario(characters=[
            make_char("c1", abilities=[
                AbilityDef(
                    id="strike", name="Strike", stamina_cost=15,
                    cooldown_rounds=3, target="single",
                    effects=[{"type": "damage", "target": "single", "amount": 12}],
                ),
            ]),
            make_char("c2"),
        ])

    def test_ability_damage_and_cost(self):
        scenario = self.strike_scenario()
        state, _ = init_two_players(scenario)

        events = resolve_round(state, scenario, {
            "p1": Action(player_id="p1", type=ActionType.SPECIAL,
                         params={"ability_id": "strike", "target": "p2"}),
            "p2": Action(player_id="p2", type=ActionType.WAIT),
        }, round_seed=1)

        assert any(e.kind == "ability_used" for e in events)
        assert state.actors["p2"].stats.hp == 88
        assert state.actors["p1"].stats.stamina == 85
        assert state.actors["p1"].ability_cooldowns["strike"] == 2  # decremented at round end

    def test_ability_on_cooldown_does_nothing(self):
        scenario = self.strike_scenario()
        state, _ = init_two_players(scenario)

        state.actors["p1"].ability_cooldowns["strike"] = 1
        events = resolve_round(state, scenario, {
            "p1": Action(player_id="p1", type=ActionType.SPECIAL,
                         params={"ability_id": "strike", "target": "p2"}),
            "p2": Action(player_id="p2", type=ActionType.WAIT),
        }, round_seed=1)

        assert not any(e.kind == "ability_used" for e in events)
        assert state.actors["p2"].stats.hp == 100
        assert state.actors["p1"].stats.stamina == 100

    def test_ability_requires_insufficient_stamina_fails(self):
        scenario = self.strike_scenario()
        state, _ = init_two_players(scenario)
        state.actors["p1"].stats.stamina = 10

        events = resolve_round(state, scenario, {
            "p1": Action(player_id="p1", type=ActionType.SPECIAL,
                         params={"ability_id": "strike", "target": "p2"}),
            "p2": Action(player_id="p2", type=ActionType.WAIT),
        }, round_seed=1)
        assert not any(e.kind == "ability_used" for e in events)
        assert state.actors["p2"].stats.hp == 100

    def test_ability_requires_same_location(self):
        scenario = self.strike_scenario()
        state, _ = init_two_players(scenario)
        state.actors["p2"].location_id = "b"

        events = resolve_round(state, scenario, {
            "p1": Action(player_id="p1", type=ActionType.SPECIAL,
                         params={"ability_id": "strike", "target": "p2"}),
            "p2": Action(player_id="p2", type=ActionType.WAIT),
        }, round_seed=1)
        assert not any(e.kind == "ability_used" for e in events)
        assert state.actors["p2"].stats.hp == 100

    def test_trigger_event_ability(self):
        scenario = make_scenario(
            characters=[
                make_char("c1", abilities=[
                    AbilityDef(
                        id="summon", name="Summon", stamina_cost=20,
                        cooldown_rounds=5, target="single",
                        effects=[{"type": "trigger_event", "event_id": "swarm"}],
                    ),
                ]),
                make_char("c2"),
            ],
            random_events=[
                RandomEvent(
                    id="swarm", name="Swarm", description="",
                    round_trigger=None, max_fires=5,
                    effects=[{"type": "damage", "target": "others_at_location", "amount": 8}],
                    narrative_seed="A swarm descends!",
                ),
            ],
        )
        state, _ = init_two_players(scenario)

        events = resolve_round(state, scenario, {
            "p1": Action(player_id="p1", type=ActionType.SPECIAL,
                         params={"ability_id": "summon", "target": "p2"}),
            "p2": Action(player_id="p2", type=ActionType.WAIT),
        }, round_seed=1)

        assert any(e.kind == "ability_used" for e in events)
        assert any(
            e.kind == "random_event" and e.payload["event_id"] == "swarm"
            for e in events
        )
        assert state.fired_events["swarm"] == 1
        assert state.actors["p2"].stats.hp == 92
        assert state.actors["p1"].stats.hp == 100  # summoner spared

    def test_legal_actions_include_ability(self):
        scenario = self.strike_scenario()
        state, _ = init_two_players(scenario)

        legal = compute_legal_actions(state, scenario, "p1")
        specials = [a for a in legal if a.type == ActionType.SPECIAL]
        assert len(specials) == 1
        assert specials[0].params_schema["ability_id"] == "strike"

    def test_legal_actions_exclude_ability_on_cooldown(self):
        scenario = self.strike_scenario()
        state, _ = init_two_players(scenario)
        state.actors["p1"].ability_cooldowns["strike"] = 2

        legal = compute_legal_actions(state, scenario, "p1")
        assert not any(a.type == ActionType.SPECIAL for a in legal)


# ---------------------------------------------------------------------------
# Loader validation for new fields
# ---------------------------------------------------------------------------

class TestNewFieldValidation:
    def test_invalid_knowledge_source_location(self, tmp_path):
        scenario = make_scenario(knowledge_sources=[
            KnowledgeSource(
                id="bad", name="Bad", acquisition="search",
                location_id="nowhere", grants_knowledge=["x"],
            ),
        ])
        yaml_path = tmp_path / "bad_knowledge.yaml"
        import yaml
        yaml_path.write_text(
            yaml.safe_dump(scenario.model_dump(), allow_unicode=True),
            encoding="utf-8",
        )
        with pytest.raises(ScenarioLoadError, match="unknown location"):
            ScenarioRegistry().load_file(yaml_path)

    def test_invalid_trigger_event_reference(self, tmp_path):
        scenario = make_scenario(characters=[
            make_char("c1", abilities=[
                AbilityDef(
                    id="summon", name="Summon", target="none",
                    effects=[{"type": "trigger_event", "event_id": "nonexistent"}],
                ),
            ]),
            make_char("c2"),
        ])
        yaml_path = tmp_path / "bad_event.yaml"
        import yaml
        yaml_path.write_text(
            yaml.safe_dump(scenario.model_dump(), allow_unicode=True),
            encoding="utf-8",
        )
        with pytest.raises(ScenarioLoadError, match="unknown event"):
            ScenarioRegistry().load_file(yaml_path)
