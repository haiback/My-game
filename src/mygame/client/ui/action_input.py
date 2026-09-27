"""Action input handling — menu selection and free-text commands."""

from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console

console = Console()


class ActionInput:
    def __init__(self):
        self._current_actions: list[dict] = []
        self._pending_input: asyncio.Future | None = None

    def set_actions(self, actions: list[dict]) -> None:
        self._current_actions = actions

    def get_action_by_number(self, num: int) -> dict | None:
        if 1 <= num <= len(self._current_actions):
            return self._current_actions[num - 1]
        return None

    async def prompt_input(self, loop: asyncio.AbstractEventLoop) -> str:
        self._pending_input = loop.create_future()

        def _reader() -> None:
            try:
                line = input()
                if self._pending_input and not self._pending_input.done():
                    self._pending_input.set_result(line)
            except EOFError:
                if self._pending_input and not self._pending_input.done():
                    self._pending_input.set_result("")
            except Exception:
                if self._pending_input and not self._pending_input.done():
                    self._pending_input.set_result("")

        loop.run_in_executor(None, _reader)
        return await self._pending_input

    def parse_selection(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        if not text:
            return None

        try:
            num = int(text)
            action = self.get_action_by_number(num)
            if action:
                return {
                    "action": {
                        "type": action["type"],
                        "params": action.get("params_schema", {}),
                    }
                }
        except ValueError:
            pass

        return {"free_text": text}
