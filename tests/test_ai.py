"""Tests for the AI module: circuit breaker, Ollama client, parser, narrator."""

from __future__ import annotations

import httpx
import pytest

from mygame.server.ai.circuit import CircuitBreaker
from mygame.server.ai.manager import AIService
from mygame.server.ai.narrator import Narrator
from mygame.server.ai.ollama_client import OllamaClient, OllamaError
from mygame.server.ai.parser import IntentParser, ParseResult
from mygame.server.engine.state import PlayerAssignment, compute_player_view, init_game
from mygame.shared.models import (
    CharacterDef,
    Item,
    Location,
    Recipe,
    ScenarioDef,
    Stats,
    VictoryCondition,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeClient:
    """Stands in for OllamaClient in parser/narrator tests."""

    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []
        self.models: list[str] = []

    async def generate(self, model, prompt, system=None, temperature=0.1):
        self.models.append(model)
        self.prompts.append(prompt)
        return self.response


class FailingParser:
    def __init__(self):
        self.calls = 0

    async def parse(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("boom")


class FailingNarrator:
    def __init__(self):
        self.calls = 0

    async def narrate(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("boom")

    async def narrate_world(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Mini scenario / view builder
# ---------------------------------------------------------------------------

def make_scenario() -> ScenarioDef:
    return ScenarioDef(
        id="mini",
        name="Mini",
        setting="",
        intro_text="",
        locations={
            "a": Location(id="a", name="A", description="", connections={"north": "b"}),
            "b": Location(id="b", name="B", description="", connections={"south": "a"}),
        },
        items={
            "herb": Item(id="herb", name="Herb", kind="consumable", stats={"heal": 10}),
            "cloth": Item(id="cloth", name="Cloth", kind="material"),
            "bandage": Item(id="bandage", name="Bandage", kind="consumable", stats={"heal": 25}),
        },
        recipes=[Recipe(output_item_id="bandage", inputs={"cloth": 2})],
        characters=[
            CharacterDef(
                id="c1", name="Warrior", backstory="", base_stats=Stats(),
                start_location="a", secret_objective="", victory_condition=VictoryCondition(kind="survive_rounds"),
            ),
            CharacterDef(
                id="c2", name="Doctor", backstory="", base_stats=Stats(),
                start_location="b", secret_objective="", victory_condition=VictoryCondition(kind="survive_rounds"),
            ),
        ],
    )


def make_view(inventory: dict[str, int] | None = None) -> tuple:
    scenario = make_scenario()
    players = [
        PlayerAssignment(player_id="p1", player_name="P1", character_id="c1"),
        PlayerAssignment(player_id="p2", player_name="P2", character_id="c2"),
    ]
    state = init_game(scenario, players)
    state.actors["p1"].location_id = "a"
    state.actors["p2"].location_id = "a"
    if inventory:
        state.actors["p1"].inventory = dict(inventory)
    view = compute_player_view(state, scenario, "p1")
    return view, scenario


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3, probe_interval=60)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open
        cb.record_failure()
        assert cb.is_open

    def test_record_success_resets(self):
        cb = CircuitBreaker(threshold=2, probe_interval=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        cb.record_success()
        assert not cb.is_open
        cb.record_failure()
        assert not cb.is_open

    async def test_call_returns_none_when_open(self):
        cb = CircuitBreaker(threshold=1, probe_interval=60)
        calls = []

        async def fn():
            calls.append(1)
            raise RuntimeError()

        assert await cb.call(fn) is None
        assert cb.is_open
        # open with no probe interval elapsed -> fn not invoked
        assert await cb.call(fn) is None
        assert len(calls) == 1

    async def test_half_open_probe_closes_on_success(self):
        cb = CircuitBreaker(threshold=1, probe_interval=0)

        async def fail():
            raise RuntimeError()

        assert await cb.call(fail) is None
        assert cb.is_open

        async def succeed():
            return "ok"

        assert await cb.call(succeed) == "ok"
        assert not cb.is_open


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestOllamaClient:
    async def test_success_returns_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "hello world"})

        client = OllamaClient(transport=_transport(handler), max_retries=0)
        text = await client.generate("m", "prompt")
        assert text == "hello world"
        await client.close()

    async def test_non_200_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={})

        client = OllamaClient(transport=_transport(handler), max_retries=0)
        with pytest.raises(OllamaError):
            await client.generate("m", "prompt")
        await client.close()

    async def test_empty_response_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": ""})

        client = OllamaClient(transport=_transport(handler), max_retries=0)
        with pytest.raises(OllamaError):
            await client.generate("m", "prompt")
        await client.close()

    async def test_retry_then_success(self):
        state = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(500, json={})
            return httpx.Response(200, json={"response": "recovered"})

        client = OllamaClient(transport=_transport(handler), max_retries=1)
        text = await client.generate("m", "prompt")
        assert text == "recovered"
        assert state["n"] == 2
        await client.close()


# ---------------------------------------------------------------------------
# Intent parser
# ---------------------------------------------------------------------------

class TestIntentParser:
    def _parser(self, response: str) -> IntentParser:
        return IntentParser(client=FakeClient(response), model="m")

    async def test_move_valid(self):
        view, scenario = make_view()
        parser = self._parser('{"kind": "action", "type": "move", "params": {"direction": "north"}}')
        result = await parser.parse("p1", "go north", view, scenario)
        assert result.action is not None
        assert result.action.type.value == "move"
        assert result.action.params == {"direction": "north"}

    async def test_move_invalid_direction(self):
        view, scenario = make_view()
        parser = self._parser('{"kind": "action", "type": "move", "params": {"direction": "west"}}')
        result = await parser.parse("p1", "go west", view, scenario)
        assert result.action is None and not result.ambiguous

    async def test_attack_by_player_name(self):
        view, scenario = make_view()
        parser = self._parser('{"kind": "action", "type": "attack", "params": {"target": "P2"}}')
        result = await parser.parse("p1", "attack P2", view, scenario)
        assert result.action is not None
        assert result.action.params == {"target": "p2"}

    async def test_attack_by_character_name(self):
        view, scenario = make_view()
        parser = self._parser('{"kind": "action", "type": "attack", "params": {"target": "Doctor"}}')
        result = await parser.parse("p1", "attack the doctor", view, scenario)
        assert result.action is not None
        assert result.action.params == {"target": "p2"}

    async def test_use_item_name_resolves_to_id(self):
        view, scenario = make_view(inventory={"herb": 2})
        parser = self._parser('{"kind": "action", "type": "use", "params": {"item_id": "Herb"}}')
        result = await parser.parse("p1", "use the herb", view, scenario)
        assert result.action is not None
        assert result.action.params == {"item_id": "herb"}

    async def test_use_item_not_in_inventory(self):
        view, scenario = make_view(inventory={"herb": 2})
        parser = self._parser('{"kind": "action", "type": "use", "params": {"item_id": "bandage"}}')
        result = await parser.parse("p1", "use bandage", view, scenario)
        assert result.action is None

    async def test_craft_valid_index(self):
        view, scenario = make_view(inventory={"cloth": 2})
        parser = self._parser('{"kind": "action", "type": "craft", "params": {"recipe_index": 0}}')
        result = await parser.parse("p1", "craft bandage", view, scenario)
        assert result.action is not None
        assert result.action.params == {"recipe_index": 0}

    async def test_unknown_type_rejected(self):
        view, scenario = make_view()
        parser = self._parser('{"kind": "action", "type": "fly", "params": {}}')
        result = await parser.parse("p1", "fly", view, scenario)
        assert result.action is None and not result.ambiguous

    async def test_ambiguous_two_options(self):
        view, scenario = make_view()
        payload = (
            '{"kind": "ambiguous", "options": ['
            '{"label": "Attack P2", "type": "attack", "params": {"target": "P2"}},'
            '{"label": "Attack Doctor", "type": "attack", "params": {"target": "Doctor"}}'
            ']}'
        )
        parser = self._parser(payload)
        result = await parser.parse("p1", "attack them", view, scenario)
        assert result.ambiguous
        assert len(result.options) == 2
        assert result.options[0].action.params["target"] == "p2"

    async def test_ambiguous_single_valid_becomes_action(self):
        view, scenario = make_view()
        payload = (
            '{"kind": "ambiguous", "options": ['
            '{"label": "Attack P2", "type": "attack", "params": {"target": "P2"}},'
            '{"label": "Fly away", "type": "fly", "params": {}}'
            ']}'
        )
        parser = self._parser(payload)
        result = await parser.parse("p1", "attack", view, scenario)
        assert not result.ambiguous
        assert result.action is not None
        assert result.action.params["target"] == "p2"

    async def test_none_kind(self):
        view, scenario = make_view()
        parser = self._parser('{"kind": "none"}')
        result = await parser.parse("p1", "do something impossible", view, scenario)
        assert result.action is None and not result.ambiguous

    async def test_markdown_fence_json(self):
        view, scenario = make_view()
        parser = self._parser('```json\n{"kind": "action", "type": "move", "params": {"direction": "north"}}\n```')
        result = await parser.parse("p1", "go north", view, scenario)
        assert result.action is not None
        assert result.action.type.value == "move"

    async def test_garbage_response(self):
        view, scenario = make_view()
        parser = self._parser("not json at all")
        result = await parser.parse("p1", "whatever", view, scenario)
        assert result.action is None and not result.ambiguous


# ---------------------------------------------------------------------------
# Narrator
# ---------------------------------------------------------------------------

class TestNarrator:
    def _narrator(self, response: str) -> Narrator:
        return Narrator(client=FakeClient(response), model="m")

    async def test_returns_text(self):
        narrator = self._narrator("你搜索了沙滩，发现一卷绷带。")
        scenario = make_scenario()
        players = [
            PlayerAssignment(player_id="p1", player_name="P1", character_id="c1"),
        ]
        state = init_game(scenario, players)
        actor = state.actors["p1"]

        from mygame.shared.models import GameEvent
        events = [GameEvent(round=1, kind="search_found", actor_ids=["p1"], narrative_seed="P1 found Herb at A.")]
        text = await narrator.narrate(actor, 1, events, scenario)
        assert text == "你搜索了沙滩，发现一卷绷带。"

    async def test_prompt_contains_facts(self):
        narrator = self._narrator("ok")
        scenario = make_scenario()
        players = [PlayerAssignment(player_id="p1", player_name="P1", character_id="c1")]
        state = init_game(scenario, players)
        actor = state.actors["p1"]

        from mygame.shared.models import GameEvent
        events = [GameEvent(round=2, kind="attack", actor_ids=["p1", "p2"], narrative_seed="P1 attacked P2.")]
        await narrator.narrate(actor, 2, events, scenario)
        prompt = narrator.client.prompts[0]
        assert "P1" in prompt
        assert "P1 attacked P2" in prompt

    async def test_empty_response_raises(self):
        narrator = self._narrator("   ")
        scenario = make_scenario()
        players = [PlayerAssignment(player_id="p1", player_name="P1", character_id="c1")]
        state = init_game(scenario, players)
        actor = state.actors["p1"]

        from mygame.shared.models import GameEvent
        events = [GameEvent(round=1, kind="wait", actor_ids=["p1"], narrative_seed="P1 waited.")]
        with pytest.raises(ValueError):
            await narrator.narrate(actor, 1, events, scenario)

    async def test_prompt_includes_world_summary_and_others(self):
        narrator = self._narrator("ok")
        scenario = make_scenario()
        players = [PlayerAssignment(player_id="p1", player_name="P1", character_id="c1")]
        state = init_game(scenario, players)
        actor = state.actors["p1"]

        from mygame.shared.models import GameEvent
        events = [GameEvent(round=3, kind="move", actor_ids=["p2"], narrative_seed="P2 moved.")]
        await narrator.narrate(
            actor, 3, events, scenario,
            world_summary="有人穿过迷雾。",
            others_summary="P2(Doctor) 与你同处一地, 存活",
            upcoming_threat="黑雾蔓延——雾气逼近中环。",
            history=["上一回合你在废墟中搜寻。"],
        )
        prompt = narrator.client.prompts[0]
        assert "有人穿过迷雾" in prompt
        assert "P2(Doctor) 与你同处一地" in prompt
        assert "黑雾蔓延" in prompt
        assert "上一回合你在废墟中搜寻" in prompt

    async def test_prompt_includes_identity_focus(self):
        scenario = make_scenario()
        scenario.characters[0].focus = "你更在意威胁与战术。"
        narrator = self._narrator("ok")
        players = [PlayerAssignment(player_id="p1", player_name="P1", character_id="c1")]
        state = init_game(scenario, players)
        actor = state.actors["p1"]

        from mygame.shared.models import GameEvent
        events = [GameEvent(round=1, kind="wait", actor_ids=["p1"], narrative_seed="P1 waited.")]
        await narrator.narrate(actor, 1, events, scenario)
        assert "你更在意威胁与战术" in narrator.client.prompts[0]

    async def test_narrate_world_empty_events_returns_empty(self):
        narrator = self._narrator("不应被调用")
        assert await narrator.narrate_world(1, []) == ""

    async def test_narrate_world_builds_public_prompt(self):
        narrator = self._narrator("黑雾越过城郊。")
        from mygame.shared.models import GameEvent
        events = [GameEvent(round=4, kind="random_event", actor_ids=[], narrative_seed="雾墙越过城郊。")]
        text = await narrator.narrate_world(4, events)
        assert text == "黑雾越过城郊。"
        assert "雾墙越过城郊" in narrator.client.prompts[0]


# ---------------------------------------------------------------------------
# AIService (breaker wiring)
# ---------------------------------------------------------------------------

class TestAIService:
    def _service(self, threshold: int = 2) -> AIService:
        service = AIService({
            "circuit_breaker_threshold": threshold,
            "probe_interval_seconds": 60,
            "parser_model": "p",
            "narrator_model": "n",
        })
        return service

    async def test_parse_action_opens_breaker_and_degrades(self):
        service = self._service(threshold=2)
        failing = FailingParser()
        service.parser = failing

        view, scenario = make_view()
        r1 = await service.parse_action("p1", "hi", view, scenario)
        assert isinstance(r1, ParseResult) and r1.action is None
        r2 = await service.parse_action("p1", "hi", view, scenario)
        assert r2.action is None
        # breaker now open; third call must not reach the parser
        r3 = await service.parse_action("p1", "hi", view, scenario)
        assert r3.action is None
        assert failing.calls == 2

    async def test_narrate_round_degrades_to_none(self):
        service = self._service(threshold=2)
        failing = FailingNarrator()
        service.narrator = failing

        scenario = make_scenario()
        players = [PlayerAssignment(player_id="p1", player_name="P1", character_id="c1")]
        state = init_game(scenario, players)
        actor = state.actors["p1"]

        from mygame.shared.models import GameEvent
        events = [GameEvent(round=1, kind="wait", actor_ids=["p1"], narrative_seed="P1 waited.")]

        assert await service.narrate_round(actor, 1, events, scenario) is None
        assert await service.narrate_round(actor, 1, events, scenario) is None
        assert await service.narrate_round(actor, 1, events, scenario) is None
        assert failing.calls == 2

    async def test_narrate_world_degrades_to_none(self):
        service = self._service(threshold=2)
        failing = FailingNarrator()
        service.narrator = failing

        from mygame.shared.models import GameEvent
        events = [GameEvent(round=1, kind="move", actor_ids=["p1"], narrative_seed="P1 moved.")]

        assert await service.narrate_world(1, events) is None
        assert await service.narrate_world(1, events) is None
        assert await service.narrate_world(1, events) is None
        assert failing.calls == 2

    async def test_narrate_world_skips_when_no_public_events(self):
        service = self._service(threshold=2)
        failing = FailingNarrator()
        service.narrator = failing
        assert await service.narrate_world(1, []) is None
        assert failing.calls == 0
