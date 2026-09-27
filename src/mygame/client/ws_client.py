"""WebSocket client for connecting to the game server.

Handles connection lifecycle, message dispatch, and reconnection.
Runs a receive loop in the background and dispatches messages to
registered handlers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable

import websockets
from websockets.asyncio.client import ClientConnection

log = logging.getLogger(__name__)

MessageHandler = Callable[[str, dict[str, Any]], Awaitable[None]]


class WSClient:
    def __init__(self, uri: str):
        self.uri = uri
        self._ws: ClientConnection | None = None
        self._handlers: dict[str, MessageHandler] = {}
        self._recv_task: asyncio.Task | None = None
        self._connected = asyncio.Event()
        self._closed = False

    def on(self, msg_type: str, handler: MessageHandler) -> None:
        self._handlers[msg_type] = handler

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.uri)
        self._closed = False
        self._connected.set()
        self._recv_task = asyncio.create_task(self._recv_loop())
        log.info("Connected to %s", self.uri)

    async def disconnect(self) -> None:
        self._closed = True
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
        self._connected.clear()

    async def send(self, msg_type: str, payload: dict[str, Any] | None = None) -> None:
        if not self._ws or self._closed:
            raise RuntimeError("Not connected")
        msg = {
            "type": msg_type,
            "payload": payload or {},
        }
        await self._ws.send(json.dumps(msg))

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                if self._closed:
                    break
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")
                payload = data.get("payload", {})

                handler = self._handlers.get(msg_type)
                if handler:
                    try:
                        await handler(msg_type, payload)
                    except Exception:
                        log.exception("Handler error for %s", msg_type)
                else:
                    log.debug("No handler for message type: %s", msg_type)
        except websockets.ConnectionClosed:
            log.info("Connection closed by server")
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("WebSocket receive error")
        finally:
            self._connected.clear()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()
