"""Room / lobby management.

A Room holds the pre-game state: connected players, selected scenario,
character picks, and ready flags. When all players are ready, the room
transitions to a running game via room_task.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field

from mygame.server.net.connection import Connection
from mygame.server.scenario.loader import ScenarioRegistry
from mygame.shared.models import ScenarioDef

log = logging.getLogger(__name__)

ROOM_CODE_LEN = 6
MAX_PLAYERS_PER_ROOM = 4


@dataclass
class PlayerSlot:
    connection: Connection
    player_name: str
    character_id: str | None = None
    ready: bool = False


class Room:
    def __init__(
        self,
        code: str,
        scenario: ScenarioDef,
        host_player_id: str,
        host_connection: Connection,
        host_name: str,
    ):
        self.code = code
        self.scenario = scenario
        self.host_player_id = host_player_id
        self.created_at: float = time.time()
        self._slots: dict[str, PlayerSlot] = {}
        self._slots[host_player_id] = PlayerSlot(
            connection=host_connection,
            player_name=host_name,
        )

    @property
    def players(self) -> dict[str, PlayerSlot]:
        return dict(self._slots)

    @property
    def player_count(self) -> int:
        return len(self._slots)

    @property
    def all_ready(self) -> bool:
        if not self._slots:
            return False
        return all(s.ready and s.character_id for s in self._slots.values())

    @property
    def all_have_character(self) -> bool:
        return all(s.character_id for s in self._slots.values())

    def can_join(self) -> bool:
        return self.player_count < MAX_PLAYERS_PER_ROOM

    def add_player(self, player_id: str, name: str, conn: Connection) -> bool:
        if not self.can_join():
            return False
        self._slots[player_id] = PlayerSlot(
            connection=conn,
            player_name=name,
        )
        return True

    def remove_player(self, player_id: str) -> None:
        self._slots.pop(player_id, None)

    def select_character(self, player_id: str, character_id: str) -> bool:
        slot = self._slots.get(player_id)
        if slot is None:
            return False
        valid_ids = {c.id for c in self.scenario.characters}
        if character_id not in valid_ids:
            return False
        taken = {
            s.character_id for pid, s in self._slots.items()
            if pid != player_id and s.character_id
        }
        if character_id in taken:
            return False
        slot.character_id = character_id
        slot.ready = False
        return True

    def set_ready(self, player_id: str) -> bool:
        slot = self._slots.get(player_id)
        if slot is None or not slot.character_id:
            return False
        slot.ready = True
        return True

    def get_assignments(self) -> list[tuple[str, str, str]]:
        return [
            (pid, slot.player_name, slot.character_id)
            for pid, slot in self._slots.items()
            if slot.character_id
        ]

    def room_state_payload(self) -> dict:
        players = []
        for pid, slot in self._slots.items():
            players.append({
                "player_id": pid,
                "player_name": slot.player_name,
                "character_id": slot.character_id,
                "ready": slot.ready,
                "is_host": pid == self.host_player_id,
            })
        chars = [
            {
                "id": c.id,
                "name": c.name,
                "backstory": c.backstory,
                "taken": any(s.character_id == c.id for s in self._slots.values()),
            }
            for c in self.scenario.characters
        ]
        return {
            "room_code": self.code,
            "players": players,
            "scenario_id": self.scenario.id,
            "scenario_name": self.scenario.name,
            "characters_available": chars,
        }


class Lobby:
    def __init__(self, scenario_registry: ScenarioRegistry):
        self._registry = scenario_registry
        self._rooms: dict[str, Room] = {}

    def create_room(
        self,
        scenario_id: str | None,
        host_player_id: str,
        host_connection: Connection,
        host_name: str,
    ) -> Room:
        if scenario_id:
            scenario = self._registry.get(scenario_id)
        else:
            scenario = self._registry.get_random()

        code = self._generate_code()
        room = Room(
            code=code,
            scenario=scenario,
            host_player_id=host_player_id,
            host_connection=host_connection,
            host_name=host_name,
        )
        self._rooms[code] = room
        log.info("Room %s created by %s with scenario %s", code, host_name, scenario.id)
        return room

    def join_room(self, code: str, player_id: str, name: str, conn: Connection) -> Room | None:
        room = self._rooms.get(code.upper())
        if room is None:
            return None
        if not room.can_join():
            return None
        room.add_player(player_id, name, conn)
        return room

    def get_room(self, code: str) -> Room | None:
        return self._rooms.get(code.upper())

    def remove_room(self, code: str) -> None:
        self._rooms.pop(code.upper(), None)

    def list_rooms(self) -> list[str]:
        return list(self._rooms.keys())

    def _generate_code(self) -> str:
        for _ in range(100):
            code = secrets.token_urlsafe(ROOM_CODE_LEN)[:ROOM_CODE_LEN].upper()
            if code not in self._rooms:
                return code
        return secrets.token_hex(ROOM_CODE_LEN // 2).upper()
