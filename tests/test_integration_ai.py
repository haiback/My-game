"""Integration regression tests for the Ollama AI module.

These hit the real local Ollama and (for the full-flow test) spin up the
real server, so they are skipped automatically when Ollama is not running
or the required models are missing.

Run explicitly with:  pytest -m integration
Skip them with:       pytest -m "not integration"
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import websockets
import yaml

from mygame.server.ai.manager import AIService
from mygame.server.engine.state import compute_player_view
from mygame.server.scenario.loader import ScenarioRegistry
from mygame.server.session.lobby import Room
from mygame.server.session.room_task import RoomTask
from mygame.shared.models import Action, ActionType, GamePhase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"

OLLAMA_URL = "http://localhost:11434"
REQUIRED_MODELS = {"qwen2.5:3b", "qwen2.5:7b"}

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

def ollama_ready() -> bool:
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        resp.raise_for_status()
        models = {m.get("name", "") for m in resp.json().get("models", [])}
        return REQUIRED_MODELS <= models
    except Exception:
        return False


@pytest.fixture(scope="module")
def ai_available():
    if not ollama_ready():
        pytest.skip("Ollama not running or required models missing")
    return True


def _ai_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    ollama = dict(cfg.get("ollama", {}))
    ollama.setdefault("timeout_seconds", 180)  # first model load can be slow
    return ollama


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeConn:
    def __init__(self):
        self.closed = False
        self.sent: list[tuple[str, dict]] = []

    async def send(self, msg_type: str, payload: dict | None = None):
        self.sent.append((msg_type, payload or {}))


async def _start_game(round_time: int = 45):
    scenario = ScenarioRegistry().load_file(SCENARIOS_DIR / "island_of_whispers.yaml")
    ai = AIService(_ai_config())
    c1, c2 = FakeConn(), FakeConn()
    room = Room("TEST1", scenario, "p1", c1, "Alice")
    room.add_player("p2", "Bob", c2)
    room.select_character("p1", "ex_soldier")
    room.select_character("p2", "entomologist")
    room.set_ready("p1")
    room.set_ready("p2")
    game = RoomTask(room, round_time=round_time, ai=ai)
    await game.start()
    return game, ai, scenario, c1, c2


async def _wait_decision(game: RoomTask) -> None:
    for _ in range(50):
        if game.state and game.state.phase == GamePhase.DECISION:
            return
        await asyncio.sleep(0.1)


async def _wait_round(game: RoomTask, target: int, timeout: float = 180.0) -> None:
    for _ in range(int(timeout / 0.2)):
        await asyncio.sleep(0.2)
        if game.state and game.state.round >= target:
            return
    raise AssertionError(f"round did not reach {target}")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# Test 1: direct RoomTask — free-text parse + AI narration
# ---------------------------------------------------------------------------

async def test_ai_parse_and_narrate(ai_available):
    game, ai, scenario, c1, c2 = await _start_game()
    try:
        await _wait_decision(game)

        view = compute_player_view(game.state, scenario, "p1")
        ok1 = await game.submit_action("p1", Action(player_id="p1", type=ActionType.SEARCH))
        ok2 = await game.submit_action("p2", Action(player_id="p2", type=ActionType.SEARCH))
        assert ok1 and ok2

        result = await ai.parse_action("p1", "搜索一下这片沙滩看看有什么", view, scenario)
        assert result.action is not None
        assert result.action.type == ActionType.SEARCH

        await _wait_round(game, 2)

        narratives = [p.get("text", "") for t, p in c1.sent if t == "narrative"]
        assert narratives, "expected at least one narrative message"
        assert any(_has_cjk(t) for t in narratives), "AI narration should be Chinese prose"
    finally:
        await game.stop()
        await ai.close()


# ---------------------------------------------------------------------------
# Test 2: full flow — real server + two WebSocket clients
# ---------------------------------------------------------------------------

async def test_full_game_flow(ai_available, tmp_path):
    port = _free_port()
    env = dict(os.environ)
    env["MYGAME_PORT"] = str(port)

    log_path = tmp_path / "server.log"
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "mygame.server.main"],
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
        )

    uri = f"ws://127.0.0.1:{port}/ws"
    try:
        for _ in range(40):
            try:
                probe = await websockets.connect(uri)
                await probe.close()
                break
            except Exception:
                await asyncio.sleep(0.5)
        else:
            raise AssertionError("server did not come up")

        bucket_a: list = []
        bucket_b: list = []
        ws_a = await websockets.connect(uri)
        ws_b = await websockets.connect(uri)
        reader_a = asyncio.create_task(_read_into(ws_a, bucket_a))
        reader_b = asyncio.create_task(_read_into(ws_b, bucket_b))

        async def send(ws, msg_type, payload=None):
            message = {"type": msg_type, "payload": payload or {}}
            await ws.send(json.dumps(message))

        await send(ws_a, "hello", {"player_name": "Alice"})
        await send(ws_b, "hello", {"player_name": "Bob"})

        await send(ws_a, "join_room", {"create": True})
        rs = await _wait_for(bucket_a, "room_state", pred=lambda m: bool(m["payload"].get("room_code")))
        code = rs["payload"]["room_code"]
        assert code

        await send(ws_b, "join_room", {"room_code": code})
        await _wait_for(bucket_b, "room_state", pred=lambda m: bool(m["payload"].get("room_code")))

        await send(ws_a, "select_character", {"character_id": "ex_soldier"})
        await send(ws_b, "select_character", {"character_id": "entomologist"})
        await asyncio.sleep(0.3)
        await send(ws_a, "ready", {})
        await send(ws_b, "ready", {})

        gs_a = await _wait_for(bucket_a, "game_start", timeout=30)
        gs_b = await _wait_for(bucket_b, "game_start", timeout=30)
        assert gs_a and gs_b, f"game_start missing (A={bool(gs_a)}, B={bool(gs_b)})"
        assert gs_a["payload"].get("your_character", {}).get("name")

        await _wait_for(bucket_a, "round_start", timeout=30)
        await _wait_for(bucket_b, "round_start", timeout=30)

        bucket_a.clear()
        bucket_b.clear()
        await send(ws_a, "action_submit", {"free_text": "搜索一下这片区域看看有什么"})
        await send(ws_b, "action_submit", {"free_text": "搜索一下周围找点物资"})

        await _wait_for(bucket_a, "round_start", pred=lambda m: m["payload"].get("round", 0) >= 2, timeout=180)
        await _wait_for(bucket_b, "round_start", pred=lambda m: m["payload"].get("round", 0) >= 2, timeout=180)

        for label, bucket in (("A", bucket_a), ("B", bucket_b)):
            narratives = [m["payload"].get("text", "") for m in bucket if m["type"] == "narrative"]
            assert narratives, f"player {label} got no narrative"
            assert any(_has_cjk(t) for t in narratives), f"player {label} narration should be Chinese prose"

        await ws_a.close()
        await ws_b.close()
        reader_a.cancel()
        reader_b.cancel()
    finally:
        proc.terminate()
        try:
            await asyncio.sleep(0.5)
            proc.kill()
        except Exception:
            pass


async def _read_into(ws, bucket: list) -> None:
    try:
        async for raw in ws:
            try:
                bucket.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
    except websockets.ConnectionClosed:
        pass


async def _wait_for(bucket: list, want: str, pred=None, timeout: float = 60.0):
    import time

    end = time.time() + timeout
    while time.time() < end:
        for m in bucket:
            if m.get("type") == want and (pred is None or pred(m)):
                return m
        await asyncio.sleep(0.1)
    return None
