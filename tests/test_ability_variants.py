"""Tests for condition-triggered ability variants."""

from __future__ import annotations

import pytest

from mygame.server.engine.conditions import evaluate_condition, select_variant
from mygame.server.engine.rules import resolve_round
from mygame.server.engine.state import PlayerAssignment, init_game
from mygame.server.scenario.loader import ScenarioLoadError, ScenarioRegistry
from mygame.shared.models import (
    AbilityDef,
    AbilityVariant,
    Action,
    ActionType,
    CharacterDef,
    Item,
    Location,
    ScenarioDef,
    Stats,
    VictoryCondition,
)

from pathlib import Path


def _make_scenario() -> ScenarioDef:
    return ScenarioDef(
        id="cond",
        name="Cond",
        setting="",
        intro_text="",
        locations={
            "a": Location(id="a", name="A", description="", tags=["ritual"], connections={"north": "b"}),
            "b": Location(id="b", name="B", description="", connections={"south": "a"}),
        },
        items={
            "potion": Item(id="potion", name="Potion", kind="consumable", stats={"heal": 10}),
        },
        characters=[
            CharacterDef(
                id="c1",
                name="Hero",
                backstory="",
                base_stats=Stats(hp=100, max_hp=100, stamina=100, max_stamina=100, attack=10, defense=5, speed=5),
                start_location="a",
                secret_objective="",
                victory_condition=VictoryCondition(kind="survive_rounds"),
                abilities=[
                    AbilityDef(
                        id="finisher",
                        name="Finisher",
                        stamina_cost=10,
                        cooldown_rounds=2,
                        target="single",
                        effects=[{"type": "damage", "target": "single", "amount": 10}],
                        variants=[
                            AbilityVariant(
                                name="execute",
                                condition={"target_hp_pct_below": 50},
                                effects=[{"type": "damage", "target": "single", "amount": 30}],
                            ),
                        ],
                    ),
                    AbilityDef(
                        id="ritual_burst",
                        name="RitualBurst",
                        stamina_cost=10,
                        target="self",
                        effects=[{"type": "heal", "target": "self", "amount": 5}],
                        variants=[
                            AbilityVariant(
                                name="sanctified",
                                condition={"at_location_tag": "ritual"},
                                effects=[{"type": "heal", "target": "self", "amount": 20}],
                            ),
                        ],
                    ),
                ],
            ),
            CharacterDef(
                id="c2",
                name="Foe",
                backstory="",
                base_stats=Stats(hp=80, max_hp=80, stamina=100, max_stamina=100, attack=5, defense=5, speed=5),
                start_location="a",
                secret_objective="",
                victory_condition=VictoryCondition(kind="survive_rounds"),
            ),
        ],
    )


def _state(scenario=None):
    s = scenario or _make_scenario()
    return init_game(s, [
        PlayerAssignment("p1", "Hero", "c1"),
        PlayerAssignment("p2", "Foe", "c2"),
    ])


class TestConditionEvaluation:
    def test_self_hp_pct_below(self):
        st = _state()
        actor = st.actors["p1"]
        actor.stats.hp = 30
        assert evaluate_condition(st, _make_scenario(), actor, {}, {"self_hp_pct_below": 50})
        assert not evaluate_condition(st, _make_scenario(), actor, {}, {"self_hp_pct_below": 20})

    def test_target_hp_pct_below(self):
        st = _state()
        actor = st.actors["p1"]
        ctx = {"target": "p2"}
        # p2 at 80/80 = 100%, so "below 50" is false; "below 120" is true
        assert not evaluate_condition(st, _make_scenario(), actor, ctx, {"target_hp_pct_below": 50})
        assert evaluate_condition(st, _make_scenario(), actor, ctx, {"target_hp_pct_below": 120})
        # missing target → false
        assert not evaluate_condition(st, _make_scenario(), actor, {}, {"target_hp_pct_below": 50})

    def test_at_location_tag(self):
        st = _state()
        actor = st.actors["p1"]
        actor.location_id = "a"  # has "ritual" tag
        assert evaluate_condition(st, _make_scenario(), actor, {}, {"at_location_tag": "ritual"})
        actor.location_id = "b"
        assert not evaluate_condition(st, _make_scenario(), actor, {}, {"at_location_tag": "ritual"})

    def test_self_has_status_and_item(self):
        st = _state()
        actor = st.actors["p1"]
        from mygame.shared.models import StatusEffect
        actor.status_effects = [StatusEffect(kind="poison", rounds_left=2, magnitude=3)]
        actor.inventory["potion"] = 1
        assert evaluate_condition(st, _make_scenario(), actor, {}, {"self_has_status": "poison"})
        assert evaluate_condition(st, _make_scenario(), actor, {}, {"self_has_item": "potion"})
        assert not evaluate_condition(st, _make_scenario(), actor, {}, {"self_has_status": "burn"})

    def test_global_flag(self):
        st = _state()
        st.global_flags.add("awakened")
        actor = st.actors["p1"]
        assert evaluate_condition(st, _make_scenario(), actor, {}, {"global_flag": "awakened"})
        assert not evaluate_condition(st, _make_scenario(), actor, {}, {"global_flag": "missing"})

    def test_all_of_any_of(self):
        st = _state()
        actor = st.actors["p1"]
        actor.stats.hp = 20
        cond = {"all_of": [{"self_hp_pct_below": 50}, {"any_of": [{"self_hp_above": 10}, {"self_hp_above": 200}]}]}
        assert evaluate_condition(st, _make_scenario(), actor, {}, cond)

    def test_unknown_key_is_permissive(self):
        st = _state()
        actor = st.actors["p1"]
        assert evaluate_condition(st, _make_scenario(), actor, {}, {"totally_unknown": 1})


class TestSelectVariant:
    def test_first_matching_variant_wins(self):
        st = _state()
        actor = st.actors["p1"]
        ctx = {"source": "p1", "target": "p2"}
        variants = [
            AbilityVariant(name="v1", condition={"self_hp_pct_below": 50}, effects=[{"type": "heal", "target": "self", "amount": 1}]),
            AbilityVariant(name="v2", condition={"self_hp_pct_below": 90}, effects=[{"type": "heal", "target": "self", "amount": 2}]),
        ]
        # actor at 100% hp: neither matches
        name, eff = select_variant(st, _make_scenario(), actor, ctx, variants)
        assert name == "" and eff == []

    def test_no_match_returns_empty(self):
        st = _state()
        actor = st.actors["p1"]
        ctx = {"source": "p1", "target": "p2"}
        name, eff = select_variant(st, _make_scenario(), actor, ctx, [])
        assert name == "" and eff == []


class TestVariantResolutionInRound:
    def test_finisher_executes_when_target_low(self):
        s = _make_scenario()
        st = _state(s)
        st.actors["p2"].stats.hp = 20  # 20/80 = 25% < 50%
        st.round = 1
        for pid in st.actors:
            st.actors[pid].location_id = "a"
        actions = {
            "p1": Action(player_id="p1", type=ActionType.SPECIAL, params={"ability_id": "finisher", "target": "p2"}),
            "p2": Action(player_id="p2", type=ActionType.WAIT),
        }
        evts = resolve_round(st, s, actions, round_seed=1)
        used = next(e for e in evts if e.kind == "ability_used")
        assert used.payload["variant"] == "execute"
        # 30 damage > 20 hp → killed
        assert st.actors["p2"].alive is False

    def test_finisher_base_when_target_healthy(self):
        s = _make_scenario()
        st = _state(s)
        st.actors["p2"].stats.hp = 80  # 100%
        st.round = 1
        for pid in st.actors:
            st.actors[pid].location_id = "a"
        actions = {
            "p1": Action(player_id="p1", type=ActionType.SPECIAL, params={"ability_id": "finisher", "target": "p2"}),
            "p2": Action(player_id="p2", type=ActionType.WAIT),
        }
        evts = resolve_round(st, s, actions, round_seed=1)
        used = next(e for e in evts if e.kind == "ability_used")
        assert used.payload["variant"] == ""
        # 10 base damage, 80-10=70, still alive
        assert st.actors["p2"].alive is True
        assert st.actors["p2"].stats.hp == 70

    def test_ritual_burst_uses_location_tag_variant(self):
        s = _make_scenario()
        st = _state(s)
        st.round = 1
        st.actors["p1"].location_id = "a"  # ritual tag
        st.actors["p1"].stats.hp = 50
        actions = {
            "p1": Action(player_id="p1", type=ActionType.SPECIAL, params={"ability_id": "ritual_burst"}),
            "p2": Action(player_id="p2", type=ActionType.WAIT),
        }
        evts = resolve_round(st, s, actions, round_seed=1)
        used = next(e for e in evts if e.kind == "ability_used")
        assert used.payload["variant"] == "sanctified"
        assert st.actors["p1"].stats.hp == 70  # 50 + 20


class TestLoaderValidatesVariants:
    def test_unknown_condition_key_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("""
id: "bad"
name: "Bad"
setting: ""
intro_text: ""
locations: {a: {id: "a", name: "A", description: ""}}
items: {}
characters:
  - id: "hero"
    name: "Hero"
    backstory: ""
    base_stats: {hp: 100, max_hp: 100, stamina: 100, max_stamina: 100, attack: 10, defense: 5, speed: 5}
    start_location: "a"
    secret_objective: ""
    victory_condition: {kind: "survive_rounds"}
    abilities:
      - id: "s"
        name: "S"
        target: "self"
        effects: []
        variants:
          - name: "v"
            condition: {self_hp_beloow: 50}
            effects: []
random_events: []
""", encoding="utf-8")
        with pytest.raises(ScenarioLoadError, match="unknown condition key"):
            ScenarioRegistry().load_file(bad)

    def test_unknown_effect_in_variant_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("""
id: "bad"
name: "Bad"
setting: ""
intro_text: ""
locations: {a: {id: "a", name: "A", description: ""}}
items: {}
characters:
  - id: "hero"
    name: "Hero"
    backstory: ""
    base_stats: {hp: 100, max_hp: 100, stamina: 100, max_stamina: 100, attack: 10, defense: 5, speed: 5}
    start_location: "a"
    secret_objective: ""
    victory_condition: {kind: "survive_rounds"}
    abilities:
      - id: "s"
        name: "S"
        target: "self"
        effects: []
        variants:
          - name: "v"
            condition: {self_hp_pct_below: 50}
            effects:
              - {type: trigger_event, event_id: nonexistent}
random_events: []
""", encoding="utf-8")
        with pytest.raises(ScenarioLoadError, match="nonexistent"):
            ScenarioRegistry().load_file(bad)

    def test_scenario_loads_with_variants(self):
        s = ScenarioRegistry().load_file(Path(__file__).parent.parent / "scenarios" / "city_of_ash.yaml")
        soldier = next(c for c in s.characters if c.id == "soldier")
        assert soldier.abilities[0].variants[0].name == "濒死反击"
