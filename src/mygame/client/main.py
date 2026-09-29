"""CLI client entry point.

Connects to the game server via WebSocket, manages the full game flow:
  lobby → character select → ready → game loop → game over

Uses Rich for terminal rendering and websockets for the connection.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from mygame.client.ui.action_input import ActionInput
from mygame.client.ui.countdown import CountdownTimer
from mygame.client.ui.layout import (
    console,
    make_layout,
    render_actions,
    render_actors,
    render_game_over,
    render_header,
    render_inventory,
    render_lobby,
    render_map,
    render_narrative,
    render_stats,
)
from mygame.client.ws_client import WSClient

log = logging.getLogger(__name__)


class GameClient:
    def __init__(self, uri: str, player_name: str):
        self.uri = uri
        self.player_name = player_name
        self.ws = WSClient(uri)
        self.timer = CountdownTimer()
        self.action_input = ActionInput()

        self.player_id: str = ""
        self.phase: str = "lobby"
        self.round_num: int = 0
        self.time_left: int = 0
        self.location: str = ""
        self.stats: dict = {}
        self.inventory: dict = {}
        self.weapon: str | None = None
        self.armor: str | None = None
        self.effects: list = []
        self.actors: list[dict] = []
        self.actions: list[dict] = []
        self.location_id: str = ""
        self.map_data: list[dict] = []
        self.traveling: bool = False
        self.travel_to: str | None = None
        self.log_lines: list[str] = []
        self.narrative_lines: list[str] = []
        self.input_message: str = ""

        self._lobby_data: dict = {}
        self._game_started = False
        self._game_over = False
        self._game_over_data: dict = {}
        self._action_submitted = asyncio.Event()
        self._input_event = asyncio.Event()
        self._clarify_event = asyncio.Event()
        self._clarify_options: list[dict] = []
        self._rejected_event = asyncio.Event()

        self.ws.on("room_state", self._handle_room_state)
        self.ws.on("game_start", self._handle_game_start)
        self.ws.on("round_start", self._handle_round_start)
        self.ws.on("tick", self._handle_tick)
        self.ws.on("narrative", self._handle_narrative)
        self.ws.on("public_log", self._handle_public_log)
        self.ws.on("action_rejected", self._handle_action_rejected)
        self.ws.on("clarify", self._handle_clarify)
        self.ws.on("game_over", self._handle_game_over)
        self.ws.on("error", self._handle_error)
        self.ws.on("pong", self._handle_pong)

    async def run(self) -> None:
        await self.ws.connect()
        await self.ws.send("hello", {"player_name": self.player_name})
        console.print(f"[bold green]Connected to {self.uri}[/bold green]")

        try:
            await self._lobby_loop()
            if self._game_started:
                await self._game_loop()
        except KeyboardInterrupt:
            console.print("\n[yellow]Disconnecting...[/yellow]")
        finally:
            await self.ws.disconnect()

    async def _lobby_loop(self) -> None:
        console.print("\n[bold]=== LOBBY ===[/bold]")
        console.print("[dim]Commands: create [scenario_id] | join <code> | select <char_id> | ready | quit[/dim]\n")

        while not self._game_started:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, lambda: input("> "))
            except (EOFError, KeyboardInterrupt):
                return

            line = line.strip()
            if not line:
                continue

            if line == "quit":
                return
            elif line.startswith("create"):
                parts = line.split(maxsplit=1)
                scenario_id = parts[1] if len(parts) > 1 else None
                payload = {"create": True}
                if scenario_id:
                    payload["scenario_id"] = scenario_id
                await self.ws.send("join_room", payload)
            elif line.startswith("join "):
                code = line.split()[1]
                await self.ws.send("join_room", {"room_code": code})
            elif line.startswith("select "):
                char_id = line.split()[1]
                await self.ws.send("select_character", {"character_id": char_id})
            elif line == "ready":
                await self.ws.send("ready", {})
            else:
                console.print("[red]Unknown command[/red]")

            await asyncio.sleep(0.1)

    async def _game_loop(self) -> None:
        console.print("\n[bold green]=== GAME STARTED ===[/bold green]\n")

        while not self._game_over:
            while not self._game_over and self.round_num == 0:
                await asyncio.sleep(0.1)
            self._last_round = self.round_num
            self._action_submitted.clear()
            self._clarify_event.clear()
            self._clarify_options = []
            self._rejected_event.clear()
            self.action_input.set_actions(self.actions)

            console.print(f"\n[bold cyan]Round {self.round_num}[/bold cyan] — Choose your action:")
            for i, act in enumerate(self.actions):
                console.print(f"  [bold]{i + 1}[/bold]. {act.get('label', act.get('type', '?'))}")
            console.print(f"\n[dim]Timer: {self.timer.format()} | Enter number or type a command[/dim]")

            if not await self._play_round():
                return

        self._show_game_over()

    _last_round: int = 0

    async def _play_round(self) -> bool:
        """Run one decision round. Returns False when the player quits."""
        submitted = False
        while not self._game_over and not submitted:
            try:
                line = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: input("> ")),
                    timeout=max(1.0, self.timer.remaining),
                )
            except asyncio.TimeoutError:
                console.print("[yellow]Time's up! Auto-waiting.[/yellow]")
                await self.ws.send("action_submit", {"action": {"type": "wait", "params": {}}})
                submitted = True
                break
            except KeyboardInterrupt:
                raise
            except EOFError:
                return False

            result = self.action_input.parse_selection(line)
            if not result:
                console.print("[yellow]Invalid input. Try a number or type a command.[/yellow]")
                continue

            await self.ws.send("action_submit", result)
            console.print("[green]Submitted. Waiting for resolution...[/green]")

            outcome = await self._wait_outcome()
            if outcome == "round_end" or outcome == "game_over":
                submitted = True
            elif outcome == "clarify":
                choice = await self._prompt_clarify_choice()
                if choice is None:
                    return False
                if choice:
                    submitted = True
                # choice False (timeout): let the round auto-wait
            elif outcome == "rejected":
                console.print("[red]Action rejected — try again.[/red]")

        await self._wait_round_end()
        return True

    async def _wait_outcome(self) -> str:
        """Wait for the server's answer to a submitted action."""
        while not self._game_over:
            if self.round_num > self._last_round:
                self._last_round = self.round_num
                return "round_end"
            if self._clarify_event.is_set() and self.timer.remaining > 0:
                return "clarify"
            if self._rejected_event.is_set():
                self._rejected_event.clear()
                return "rejected"
            await asyncio.sleep(0.2)
        return "game_over"

    async def _wait_round_end(self) -> None:
        while not self._game_over and self.round_num <= self._last_round:
            await asyncio.sleep(0.2)

    async def _prompt_clarify_choice(self) -> bool | None:
        """Ask the player to pick a clarify option.

        Returns True when a choice was submitted, False on timeout
        (round will auto-wait), None when the player quits.
        """
        while not self._game_over:
            try:
                line = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: input("> ")),
                    timeout=max(1.0, self.timer.remaining),
                )
            except asyncio.TimeoutError:
                return False
            except KeyboardInterrupt:
                raise
            except EOFError:
                return None

            try:
                idx = int(line.strip())
            except ValueError:
                console.print("[yellow]请输入选项编号.[/yellow]")
                continue

            if 1 <= idx <= len(self._clarify_options):
                self._clarify_event.clear()
                self._clarify_options = []
                await self.ws.send("clarify_choice", {"index": idx - 1})
                console.print("[green]Choice submitted. Waiting for resolution...[/green]")
                return True
            console.print("[yellow]编号无效, 请重试.[/yellow]")
        return None

    def _show_game_over(self) -> None:
        console.print()
        panel = render_game_over(
            self._game_over_data.get("winner_ids", []),
            self._game_over_data.get("epilogue", ""),
        )
        console.print(panel)

    # -- Message handlers --

    async def _handle_room_state(self, msg_type: str, payload: dict) -> None:
        self._lobby_data = payload
        room_code = payload.get("room_code", "")
        if not room_code:
            return

        players = payload.get("players", [])
        scenario_name = payload.get("scenario_name", "Unknown")
        characters = payload.get("characters_available", [])

        for p in players:
            if p.get("player_name") == self.player_name:
                self.player_id = p.get("player_id", "")

        console.print()
        console.print(render_lobby(room_code, players, scenario_name, characters, self.player_id))
        console.print()

    async def _handle_game_start(self, msg_type: str, payload: dict) -> None:
        self._game_started = True
        intro = payload.get("scenario_intro", "")
        char = payload.get("your_character", {})

        console.print("\n" + "=" * 60)
        console.print(f"[bold green]{intro}[/bold green]")
        console.print("=" * 60)
        console.print(f"\n[bold]Your character:[/bold] [cyan]{char.get('name', '?')}[/cyan]")
        console.print(f"[dim]{char.get('backstory', '')}[/dim]")
        console.print(f"[bold yellow]Secret objective:[/bold yellow] {char.get('secret_objective', '')}")
        console.print()

    async def _handle_round_start(self, msg_type: str, payload: dict) -> None:
        self.round_num = payload.get("round", 0)
        deadline = payload.get("deadline_ts", 0)
        self.timer.start(deadline)

        view = payload.get("your_state", {})
        self.location = view.get("location_name", "")
        self.stats = view.get("stats", {})
        self.inventory = view.get("inventory", {})
        self.weapon = view.get("equipped_weapon")
        self.armor = view.get("equipped_armor")
        self.effects = view.get("status_effects", [])
        self.actors = view.get("visible_actors", [])
        self.actions = view.get("legal_actions", [])
        self.location_id = view.get("location_id", "")
        self.map_data = view.get("map", [])
        self.traveling = view.get("travel_remaining", 0) > 0
        self.travel_to = view.get("travel_to")

        console.print(f"\n[bold cyan]--- Round {self.round_num} ---[/bold cyan]")
        console.print(f"[bold]Location:[/bold] {self.location}")
        console.print(f"[bold]HP:[/bold] {self.stats.get('hp', '?')}/{self.stats.get('max_hp', '?')}  "
                       f"[bold]Stamina:[/bold] {self.stats.get('stamina', '?')}/{self.stats.get('max_stamina', '?')}")

        if self.traveling:
            console.print(f"[bold cyan]在途中，还剩 {view.get('travel_remaining')} 回合到达。[/bold cyan]")

        if self.map_data:
            console.print("[bold]Map:[/bold]")
            console.print(render_map(
                self.map_data, self.location_id, self.traveling,
                self.travel_to, self.actors,
            ))

        if self.actors:
            console.print("[bold]Others here:[/bold]")
            for a in self.actors:
                console.print(f"  {a['player_name']} ({a['character_name']})")

    async def _handle_tick(self, msg_type: str, payload: dict) -> None:
        self.time_left = payload.get("seconds_left", 0)

    async def _handle_narrative(self, msg_type: str, payload: dict) -> None:
        text = payload.get("text", "")
        if text:
            self.narrative_lines.append(text)
            console.print(f"[dim italic]{text}[/dim italic]")

    async def _handle_public_log(self, msg_type: str, payload: dict) -> None:
        lines = payload.get("lines", [])
        for line in lines:
            self.log_lines.append(line)
            console.print(f"[white]{line}[/white]")

    async def _handle_action_rejected(self, msg_type: str, payload: dict) -> None:
        reason = payload.get("reason", "Unknown")
        console.print(f"[red]Action rejected: {reason}[/red]")
        self._rejected_event.set()

    async def _handle_clarify(self, msg_type: str, payload: dict) -> None:
        options = payload.get("options", [])
        self._clarify_options = options
        console.print("[yellow]你的意图有点含糊, 请选择:[/yellow]")
        for opt in options:
            num = opt.get("index", 0) + 1
            console.print(f"  [bold]{num}[/bold]. {opt.get('label', '?')}")
        self._clarify_event.set()

    async def _handle_game_over(self, msg_type: str, payload: dict) -> None:
        self._game_over = True
        self._game_over_data = payload

    async def _handle_error(self, msg_type: str, payload: dict) -> None:
        code = payload.get("code", "")
        message = payload.get("message", "")
        console.print(f"[red]Error [{code}]: {message}[/red]")

    async def _handle_pong(self, msg_type: str, payload: dict) -> None:
        pass


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        player_name = "Player"
    else:
        player_name = sys.argv[1]

    uri = "ws://127.0.0.1:8765/ws"
    if len(sys.argv) >= 3:
        uri = sys.argv[2]

    console.print(f"[bold]MyGame Client[/bold] — connecting as [cyan]{player_name}[/cyan]")

    client = GameClient(uri, player_name)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Bye![/yellow]")


if __name__ == "__main__":
    main()
