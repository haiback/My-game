"""FastAPI server entry point.

Loads config, initializes scenario registry and lobby, and exposes the
WebSocket endpoint that drives the full client lifecycle:
  hello → join_room → select_character → ready → game loop → game_over
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from mygame.server.ai.manager import AIService
from mygame.server.ai.parser import ClarifyOption
from mygame.server.engine.state import compute_player_view
from mygame.server.net.connection import Connection
from mygame.server.scenario.loader import ScenarioRegistry
from mygame.server.session.lobby import Lobby
from mygame.server.session.room_task import RoomTask, parse_action_from_payload

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
SCENARIOS_DIR = BASE_DIR / "scenarios"


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def create_app() -> FastAPI:
    config = load_config()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    registry = ScenarioRegistry()
    scenarios_path = config.get("scenarios_dir", str(SCENARIOS_DIR))
    if os.path.isdir(scenarios_path):
        registry.load_directory(scenarios_path)
    elif os.path.isdir(str(SCENARIOS_DIR)):
        registry.load_directory(str(SCENARIOS_DIR))

    lobby = Lobby(registry)
    active_games: dict[str, RoomTask] = {}
    ai = AIService(config.get("ollama"))
    pending_clarify: dict[str, tuple[int, list[ClarifyOption]]] = {}

    game_cfg = config.get("game", {})
    round_time = game_cfg.get("round_time_seconds", 45)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await ai.close()

    app = FastAPI(title="MyGame Server", lifespan=lifespan)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        conn = Connection(ws)
        await conn.accept()

        player_id = ""
        room_code = ""

        try:
            while not conn.closed:
                result = await conn.recv()
                if result is None:
                    break

                msg_type, payload = result

                if msg_type == "hello":
                    player_id = uuid.uuid4().hex[:12]
                    player_name = payload.get("player_name", "Anonymous")
                    conn.player_id = player_id
                    conn.player_name = player_name
                    await conn.send("room_state", {
                        "room_code": "",
                        "players": [{"player_id": player_id, "player_name": player_name}],
                        "scenario_id": None,
                        "scenario_name": None,
                        "characters_available": [],
                    })

                elif msg_type == "join_room":
                    if not player_id:
                        await conn.send("error", {"code": "no_hello", "message": "Send hello first"})
                        continue

                    create = payload.get("create", False)
                    target_code = payload.get("room_code")
                    scenario_id = payload.get("scenario_id")

                    if create or not target_code:
                        room = lobby.create_room(scenario_id, player_id, conn, conn.player_name)
                    else:
                        room = lobby.join_room(target_code, player_id, conn.player_name, conn)
                        if room is None:
                            await conn.send("error", {"code": "room_full", "message": "Room not found or full"})
                            continue

                    room_code = room.code
                    conn.player_id = player_id
                    await conn.send("room_state", room.room_state_payload())

                    await _broadcast_room_state(room)

                elif msg_type == "select_character":
                    room = lobby.get_room(room_code)
                    if not room:
                        await conn.send("error", {"code": "no_room", "message": "Not in a room"})
                        continue
                    char_id = payload.get("character_id", "")
                    ok = room.select_character(player_id, char_id)
                    if not ok:
                        await conn.send("error", {"code": "bad_character", "message": "Character unavailable or taken"})
                        continue
                    await _broadcast_room_state(room)

                elif msg_type == "ready":
                    room = lobby.get_room(room_code)
                    if not room:
                        continue
                    ok = room.set_ready(player_id)
                    if not ok:
                        await conn.send("error", {"code": "not_ready", "message": "Select a character first"})
                        continue
                    await _broadcast_room_state(room)

                    if room.all_ready:
                        game = RoomTask(room, round_time=round_time, ai=ai)
                        active_games[room_code] = game
                        await game.start()

                elif msg_type == "action_submit":
                    game = active_games.get(room_code)
                    if not game or not game.state:
                        await conn.send("error", {"code": "no_game", "message": "No active game"})
                        continue

                    action, clarify_options = await _parse_player_action(
                        game, player_id, payload
                    )
                    if clarify_options is not None:
                        pending_clarify[player_id] = (game.state.round, clarify_options)
                        await conn.send("clarify", {
                            "round": game.state.round,
                            "options": [
                                {"index": i, "label": opt.label}
                                for i, opt in enumerate(clarify_options)
                            ],
                        })
                        continue
                    if action is None:
                        await conn.send("action_rejected", {"reason": "Could not parse action"})
                        continue

                    pending_clarify.pop(player_id, None)
                    ok = await game.submit_action(player_id, action)
                    if not ok:
                        await conn.send("action_rejected", {"reason": "Not in decision phase or player inactive"})

                elif msg_type == "clarify_choice":
                    game = active_games.get(room_code)
                    if not game or not game.state:
                        continue
                    pending = pending_clarify.pop(player_id, None)
                    if pending is None:
                        await conn.send("error", {"code": "no_clarify", "message": "No pending clarification"})
                        continue
                    pending_round, options = pending
                    if pending_round != game.state.round:
                        await conn.send("action_rejected", {"reason": "Clarification expired"})
                        continue
                    idx = payload.get("index")
                    if not isinstance(idx, int) or idx < 0 or idx >= len(options):
                        await conn.send("action_rejected", {"reason": "Invalid clarification index"})
                        continue
                    ok = await game.submit_action(player_id, options[idx].action)
                    if not ok:
                        await conn.send("action_rejected", {"reason": "Not in decision phase or player inactive"})

                elif msg_type == "ping":
                    await conn.send("pong", {})

                elif msg_type == "narrative_ack":
                    pass

        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("Unhandled error in ws loop for player %s", player_id)
        finally:
            pending_clarify.pop(player_id, None)
            if room_code and player_id:
                room = lobby.get_room(room_code)
                if room:
                    room.remove_player(player_id)
                    await _broadcast_room_state(room)
                    if room.player_count == 0:
                        game = active_games.pop(room_code, None)
                        if game:
                            await game.stop()
                        lobby.remove_room(room_code)

    async def _parse_player_action(
        game: RoomTask,
        player_id: str,
        payload: dict,
    ) -> tuple[Any, list[ClarifyOption] | None]:
        """Resolve an action_submit payload to an Action.

        Structured action dicts pass through untouched. Free text goes to
        the AI intent parser first (the AI GM's main job); on empty result
        or Ollama outage the deterministic keyword parser is the fallback.
        Returns (action, clarify_options); clarify_options is non-None when
        the AI needs the player to disambiguate.
        """
        if payload.get("action"):
            return parse_action_from_payload(player_id, payload), None

        free_text = str(payload.get("free_text", "")).strip()
        if not free_text:
            return None, None

        assert game.state is not None
        actor = game.state.actors.get(player_id)
        if actor is not None and actor.alive:
            view = compute_player_view(game.state, game.room.scenario, player_id)
            result = await ai.parse_action(player_id, free_text, view, game.room.scenario)
            if result.ambiguous:
                return None, result.options
            if result.action is not None:
                result.action.raw_text = free_text
                return result.action, None

        return parse_action_from_payload(player_id, {"free_text": free_text}), None

    async def _broadcast_room_state(room) -> None:
        data = room.room_state_payload()
        for pid, slot in room.players.items():
            if not slot.connection.closed:
                await slot.connection.send("room_state", data)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    config = load_config()
    server_cfg = config.get("server", {})
    # env overrides let integration tests bind a free port
    host = os.environ.get("MYGAME_HOST", server_cfg.get("host", "127.0.0.1"))
    port = int(os.environ.get("MYGAME_PORT", server_cfg.get("port", 8765)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
