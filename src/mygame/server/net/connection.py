"""WebSocket connection wrapper.

Handles JSON serialization, protocol validation, sequence numbering,
and graceful disconnect. Each connection gets a unique player_id assigned
on hello handshake.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from mygame.shared.protocol import (
    CLIENT_PAYLOADS,
    Envelope,
    make_envelope,
)

log = logging.getLogger(__name__)


class ConnectionClosed(Exception):
    pass


class Connection:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.player_id: str = ""
        self.player_name: str = ""
        self._seq_out: int = 0
        self._seq_in: int = 0
        self._closed: bool = False

    async def accept(self) -> None:
        await self.ws.accept()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.ws.close(code=code, reason=reason)
        except Exception:
            pass

    async def send(self, msg_type: str, payload: dict[str, Any] | Any = {}) -> None:
        if self._closed:
            return
        self._seq_out += 1
        env = make_envelope(msg_type, payload, seq=self._seq_out)
        try:
            await self.ws.send_text(env.model_dump_json())
        except Exception:
            self._closed = True
            raise ConnectionClosed()

    async def send_envelope(self, env: Envelope) -> None:
        if self._closed:
            return
        self._seq_out += 1
        env.seq = self._seq_out
        try:
            await self.ws.send_text(env.model_dump_json())
        except Exception:
            self._closed = True
            raise ConnectionClosed()

    async def recv(self) -> tuple[str, dict[str, Any]] | None:
        if self._closed:
            return None
        try:
            raw = await self.ws.receive_text()
        except WebSocketDisconnect:
            self._closed = True
            return None
        except Exception:
            self._closed = True
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await self.send("error", {"code": "bad_json", "message": "Invalid JSON"})
            return None

        msg_type = data.get("type", "")
        if msg_type not in CLIENT_PAYLOADS:
            await self.send("error", {"code": "unknown_type", "message": f"Unknown type: {msg_type}"})
            return None

        payload_cls = CLIENT_PAYLOADS[msg_type]
        try:
            payload_cls.model_validate(data.get("payload", {}))
        except Exception as exc:
            await self.send("error", {"code": "bad_payload", "message": str(exc)})
            return None

        self._seq_in += 1
        return msg_type, data.get("payload", {})

    @property
    def closed(self) -> bool:
        return self._closed
