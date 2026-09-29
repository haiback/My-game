"""Tests for the narration flow: three-axis criticality detection and the
pre-generated NarrativeLibrary."""

from __future__ import annotations

from mygame.server.ai.library import NarrativeLibrary
from mygame.server.engine.state import PlayerAssignment, init_game
from mygame.server.session.lobby import Room
from mygame.server.session.room_task import RoomTask
from mygame.shared.models import (
    CharacterDef,
    CriticalityTrigger,
    FactionDef,
    GameEvent,
    Location,
    ScenarioDef,
    Stats,
)


def _scenario(max_rounds: int = 30, triggers=None) -> ScenarioDef:
    return ScenarioDef(
        id="mini",
        name="Mini",
        setting="",
        intro_text="",
        max_rounds=max_rounds,
        locations={
            "a": Location(id="a", name="A", description=""),
            "b": Location(id="b", name="B", description="", connections={"south": "a"}),
        },
        characters=[
            CharacterDef(
                id="c1",
                name="Warrior",
                backstory="",
                base_stats=Stats(),
                start_location="a",
                faction_id="f",
                criticality_triggers=triggers or [],
            ),
        ],
        factions={"f": FactionDef(id="f", name="F")},
    )


class _FakeConn:
    closed = False


def _room_task(max_rounds: int = 30, triggers=None) -> RoomTask:
    scenario = _scenario(max_rounds, triggers)
    room = Room("T", scenario, "p1", _FakeConn(), "Alice")
    game = RoomTask(room, ai=None)
    game.state = init_game(scenario, [PlayerAssignment("p1", "P1", "c1")])
    return game


def _ev(kind: str) -> GameEvent:
    return GameEvent(round=1, kind=kind, actor_ids=["p1"], narrative_seed="x")


class TestCriticalRound:
    def test_attack_is_critical(self):
        game = _room_task()
        assert game._faction_critical("f", [_ev("attack")]) is True

    def test_ordinary_round_is_not_critical(self):
        game = _room_task()
        assert game._faction_critical("f", [_ev("move"), _ev("search_found")]) is False

    def test_danger_damage_tick_is_not_critical(self):
        game = _room_task()
        assert game._faction_critical("f", [_ev("danger_damage")]) is False

    def test_danger_damage_kill_is_critical(self):
        game = _room_task()
        ev = GameEvent(
            round=1, kind="danger_damage", actor_ids=["p1"],
            payload={"killed": True}, narrative_seed="x",
        )
        assert game._faction_critical("f", [ev]) is True

    def test_trap_is_critical(self):
        game = _room_task()
        assert game._faction_critical("f", [_ev("trap")]) is True

    def test_npc_encounter_is_critical(self):
        game = _room_task()
        assert game._faction_critical("f", [_ev("npc_encounter")]) is True

    def test_near_max_rounds_is_critical(self):
        game = _room_task(max_rounds=5)
        game.state.round = 3
        assert game._faction_critical("f", [_ev("move")]) is True

    def test_dying_member_is_critical(self):
        game = _room_task()
        game.state.actors["p1"].stats.hp = 10
        assert game._faction_critical("f", [_ev("move")]) is True

    def test_character_trigger_makes_critical(self):
        game = _room_task(triggers=[
            CriticalityTrigger(kind="search_found", item_id="ritual_dagger", weight=0.9),
        ])
        ev = GameEvent(
            round=1, kind="search_found", actor_ids=["p1"],
            payload={"item_id": "ritual_dagger"}, narrative_seed="x",
        )
        assert game._faction_critical("f", [ev]) is True

    def test_unrelated_trigger_does_not_make_critical(self):
        game = _room_task(triggers=[
            CriticalityTrigger(kind="search_found", item_id="ritual_dagger", weight=0.9),
        ])
        ev = GameEvent(
            round=1, kind="search_found", actor_ids=["p1"],
            payload={"item_id": "cloth_scrap"}, narrative_seed="x",
        )
        assert game._faction_critical("f", [ev]) is False


class TestNarrativeLibrary:
    def _lib(self) -> NarrativeLibrary:
        return NarrativeLibrary(client=object(), model="m")

    def test_parse_extracts_json(self):
        raw = '{"a": "破败的广场。", "b": "阴森的教堂。"}'
        assert self._lib()._parse(raw, _scenario()) == {
            "a": "破败的广场。",
            "b": "阴森的教堂。",
        }

    def test_parse_ignores_unknown_and_nonstring(self):
        raw = '{"a": "广场。", "b": 123, "zzz": "多余"}'
        assert self._lib()._parse(raw, _scenario()) == {"a": "广场。"}

    def test_parse_invalid_json_returns_empty(self):
        assert self._lib()._parse("not json", _scenario()) == {}

    async def test_generate_failure_leaves_empty(self):
        class Boom:
            async def generate(self, *args, **kwargs):
                raise RuntimeError("boom")

        lib = NarrativeLibrary(client=Boom(), model="m")
        await lib.generate(_scenario())
        assert lib.location_flavor == {}
