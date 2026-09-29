"""Tests for the faction model: victory (faction + personal coexist), intel
sharing (fog-of-war at faction granularity), and location-point triggering."""

from __future__ import annotations

from mygame.server.engine.rules import resolve_round
from mygame.server.engine.state import PlayerAssignment, compute_player_view, init_game
from mygame.server.engine.winconditions import check_victory
from mygame.shared.models import (
    Action,
    ActionType,
    CharacterDef,
    FactionDef,
    Location,
    LocationPoint,
    Objective,
    ScenarioDef,
    Stats,
    VictoryCondition,
)


def _survive_objective() -> Objective:
    return Objective(
        trigger=VictoryCondition(kind="survive_rounds", params={"rounds": 99}),
        victory=True,
    )


def _scenario(with_points: bool = False) -> ScenarioDef:
    loc_a = Location(id="a", name="A", description="", connections={"north": "b"})
    loc_b = Location(id="b", name="B", description="", connections={"south": "a"})
    if with_points:
        loc_b.points = [
            LocationPoint(
                id="trap", name="陷阱", kind="trap", trigger="stay",
                chance=1.0, effects=[{"type": "damage", "target": "self", "amount": 5}],
            ),
            LocationPoint(
                id="watcher", name="守望者", kind="npc_encounter", trigger="arrive",
                chance=1.0, effects=[],
            ),
        ]
    return ScenarioDef(
        id="mini", name="mini", setting="", intro_text="", max_rounds=30,
        locations={"a": loc_a, "b": loc_b},
        items={},
        characters=[
            CharacterDef(id="c1", name="A1", backstory="", base_stats=Stats(),
                         start_location="a", faction_id="f1",
                         personal_objective=_survive_objective()),
            CharacterDef(id="c2", name="A2", backstory="", base_stats=Stats(),
                         start_location="a", faction_id="f1"),
            CharacterDef(id="c3", name="B1", backstory="", base_stats=Stats(),
                         start_location="a", faction_id="f2",
                         personal_objective=_survive_objective()),
        ],
        factions={
            "f1": FactionDef(id="f1", name="F1", objective=Objective(
                trigger=VictoryCondition(kind="eliminate_all"), victory=True,
            )),
            "f2": FactionDef(id="f2", name="F2", objective=Objective(
                trigger=VictoryCondition(kind="eliminate_all"), victory=True,
            )),
        },
    )


def _init(with_points: bool = False):
    sc = _scenario(with_points)
    st = init_game(sc, [
        PlayerAssignment("p1", "P1", "c1"),
        PlayerAssignment("p2", "P2", "c2"),
        PlayerAssignment("p3", "P3", "c3"),
    ])
    return st, sc


class TestFactionVictory:
    def test_faction_wins_when_enemies_eliminated(self):
        st, sc = _init()
        st.actors["p3"].alive = False  # f2 eliminated -> f1 wins
        winners = check_victory(st, sc)
        assert "p1" in winners and "p2" in winners
        assert "p3" not in winners

    def test_personal_victory_coexists(self):
        st, sc = _init()
        st.round = 99
        winners = check_victory(st, sc)
        assert "p1" in winners and "p3" in winners  # personal survive_rounds(99)


class TestIntelSharing:
    def test_faction_member_visible_across_distance(self):
        st, sc = _init()
        st.actors["p2"].location_id = "b"  # faction member far away
        st.actors["p3"].location_id = "b"  # enemy far away
        view = compute_player_view(st, sc, "p1")  # p1 at "a"
        visible = {v.player_id for v in view.visible_actors}
        assert "p2" in visible
        assert "p3" not in visible


class TestLocationPoints:
    def test_stay_trap_fires(self):
        st, sc = _init(with_points=True)
        st.actors["p1"].location_id = "b"
        st.round = 1
        events = resolve_round(
            st, sc, {"p1": Action(player_id="p1", type=ActionType.WAIT)},
            round_seed=1,
        )
        assert any(e.kind == "trap" for e in events)

    def test_arrive_npc_encounter_fires(self):
        st, sc = _init(with_points=True)
        st.round = 1
        events = resolve_round(
            st, sc, {"p1": Action(player_id="p1", type=ActionType.MOVE, params={"direction": "north"})},
            round_seed=1,
        )
        assert any(e.kind == "npc_encounter" for e in events)
