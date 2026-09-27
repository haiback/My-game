"""WebSocket message protocol — single source of truth for wire format.

Both server and client import from this module. Every message is a JSON
object with a ``type`` field for dispatch. Messages carry a monotonic
``seq`` for idempotent handling after reconnect.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from mygame.shared.models import GamePhase, PlayerView


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

class Envelope(BaseModel):
    type: str
    seq: int = 0
    ts: float = Field(default_factory=time.time)
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Client → Server
# ---------------------------------------------------------------------------

class HelloPayload(BaseModel):
    player_name: str
    token: str | None = None


class JoinRoomPayload(BaseModel):
    room_code: str | None = None
    create: bool = False
    scenario_id: str | None = None


class SelectCharacterPayload(BaseModel):
    character_id: str


class ReadyPayload(BaseModel):
    pass


class ActionSubmitPayload(BaseModel):
    free_text: str | None = None
    action: dict[str, Any] | None = None


class ClarifyChoicePayload(BaseModel):
    choice_index: int


class NarrativeAckPayload(BaseModel):
    pass


class PingPayload(BaseModel):
    pass


CLIENT_PAYLOADS: dict[str, type[BaseModel]] = {
    "hello": HelloPayload,
    "join_room": JoinRoomPayload,
    "select_character": SelectCharacterPayload,
    "ready": ReadyPayload,
    "action_submit": ActionSubmitPayload,
    "clarify_choice": ClarifyChoicePayload,
    "narrative_ack": NarrativeAckPayload,
    "ping": PingPayload,
}


# ---------------------------------------------------------------------------
# Server → Client
# ---------------------------------------------------------------------------

class RoomStatePayload(BaseModel):
    room_code: str
    players: list[dict[str, Any]]
    scenario_id: str | None = None
    scenario_name: str | None = None
    characters_available: list[dict[str, Any]] = Field(default_factory=list)


class GameStartPayload(BaseModel):
    scenario_intro: str
    your_character: dict[str, Any]


class RoundStartPayload(BaseModel):
    round: int
    phase: str = "decision"
    deadline_ts: float
    your_state: dict[str, Any]


class NarrativePayload(BaseModel):
    round: int
    text: str


class PublicLogPayload(BaseModel):
    round: int
    lines: list[str]


class ClarifyPayload(BaseModel):
    options: list[dict[str, Any]]


class ActionRejectedPayload(BaseModel):
    reason: str


class TickPayload(BaseModel):
    seconds_left: int


class GameOverPayload(BaseModel):
    winner_ids: list[str]
    epilogue: str
    final_log: list[dict[str, Any]] = Field(default_factory=list)


class ErrorPayload(BaseModel):
    code: str
    message: str


class PongPayload(BaseModel):
    pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_envelope(msg_type: str, payload: BaseModel | dict, seq: int = 0) -> Envelope:
    data = payload.model_dump() if isinstance(payload, BaseModel) else payload
    return Envelope(type=msg_type, seq=seq, payload=data)
