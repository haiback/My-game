"""Rich terminal UI layout.

Provides the main game screen with panels for:
- Player stats (HP, stamina, status effects)
- Location info and visible actors
- Inventory
- Event log
- Action menu / input
- Countdown timer
"""

from __future__ import annotations

from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console()


def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )
    layout["left"].split_column(
        Layout(name="narrative", ratio=3),
        Layout(name="actions", ratio=2),
    )
    layout["right"].split_column(
        Layout(name="stats", size=10),
        Layout(name="inventory", ratio=1),
        Layout(name="actors", ratio=1),
    )
    return layout


def render_header(round_num: int, phase: str, time_left: int, location: str) -> Panel:
    header_text = Text()
    header_text.append(f"  Round {round_num}  ", style="bold white on blue")
    header_text.append(f"  {phase.upper()}  ", style="bold white on green")
    if time_left > 0:
        color = "red" if time_left <= 10 else "yellow"
        header_text.append(f"  {time_left}s  ", style=f"bold white on {color}")
    header_text.append(f"  {location}  ", style="bold cyan")
    return Panel(header_text, style="dim")


def render_stats(stats: dict, effects: list) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", width=10)
    table.add_column()

    hp = stats.get("hp", 0)
    max_hp = stats.get("max_hp", 100)
    hp_color = "green" if hp > max_hp * 0.5 else "yellow" if hp > max_hp * 0.25 else "red"
    table.add_row("HP", f"[{hp_color}]{hp}/{max_hp}[/{hp_color}]")

    stam = stats.get("stamina", 0)
    max_stam = stats.get("max_stamina", 100)
    table.add_row("Stamina", f"{stam}/{max_stam}")

    table.add_row("Attack", str(stats.get("attack", 0)))
    table.add_row("Defense", str(stats.get("defense", 0)))
    table.add_row("Speed", str(stats.get("speed", 0)))

    if effects:
        effect_strs = [f"{e['kind']}({e['rounds_left']})" for e in effects]
        table.add_row("Effects", ", ".join(effect_strs))

    return Panel(table, title="[bold]Stats[/bold]", border_style="cyan")


def render_inventory(inventory: dict, weapon: str | None, armor: str | None) -> Panel:
    lines = []
    if weapon:
        lines.append(f"[bold yellow]Weapon:[/bold yellow] {weapon}")
    if armor:
        lines.append(f"[bold blue]Armor:[/bold blue] {armor}")
    if inventory:
        lines.append("")
        for item_id, count in inventory.items():
            lines.append(f"  {item_id} x{count}")
    else:
        lines.append("[dim]Empty[/dim]")
    return Panel("\n".join(lines), title="[bold]Inventory[/bold]", border_style="green")


def render_actors(actors: list[dict]) -> Panel:
    if not actors:
        return Panel("[dim]No one here[/dim]", title="[bold]Present[/bold]", border_style="magenta")
    lines = []
    for a in actors:
        hp = a.get("stats_summary", {}).get("hp", "?")
        lines.append(f"  {a['player_name']} ({a['character_name']}) HP:{hp}")
    return Panel("\n".join(lines), title="[bold]Present[/bold]", border_style="magenta")


def render_narrative(log_lines: list[str]) -> Panel:
    if not log_lines:
        return Panel("[dim]Waiting for events...[/dim]", title="[bold]Narrative[/bold]", border_style="white")
    display_lines = log_lines[-15:]
    text = "\n".join(display_lines)
    return Panel(text, title="[bold]Narrative[/bold]", border_style="white", height=15)


def render_actions(actions: list[dict], message: str = "") -> Panel:
    if not actions:
        return Panel("[dim]No actions available[/dim]", title="[bold]Actions[/bold]", border_style="yellow")

    lines = []
    for i, act in enumerate(actions):
        label = act.get("label", act.get("type", "?"))
        lines.append(f"  [bold cyan]{i + 1}[/bold cyan]. {label}")

    if message:
        lines.append("")
        lines.append(f"[yellow]{message}[/yellow]")

    lines.append("")
    lines.append("[dim]Enter number to select, or type a command:[/dim]")

    return Panel("\n".join(lines), title="[bold]Actions[/bold]", border_style="yellow", height=15)


def render_game_over(winners: list[str], epilogue: str) -> Panel:
    text = f"[bold]Game Over![/bold]\n\n"
    if winners:
        text += f"[bold green]Winners: {', '.join(winners)}[/bold green]\n\n"
    else:
        text += "[bold red]No survivors[/bold red]\n\n"
    text += epilogue
    return Panel(text, title="[bold red]GAME OVER[/bold red]", border_style="red", height=12)


def render_lobby(
    room_code: str,
    players: list[dict],
    scenario_name: str,
    characters: list[dict],
    my_player_id: str,
) -> str:
    lines = [
        f"[bold]Room Code:[/bold] [bold cyan]{room_code}[/bold cyan]",
        f"[bold]Scenario:[/bold] [green]{scenario_name}[/green]",
        "",
        "[bold]Players:[/bold]",
    ]
    for p in players:
        marker = " [yellow](you)[/yellow]" if p["player_id"] == my_player_id else ""
        host = " [red](host)[/red]" if p.get("is_host") else ""
        char = f" -> {p['character_id']}" if p.get("character_id") else ""
        ready = " [green]READY[/green]" if p.get("ready") else ""
        lines.append(f"  {p['player_name']}{marker}{host}{char}{ready}")

    lines.append("")
    lines.append("[bold]Characters:[/bold]")
    for c in characters:
        taken = " [red](taken)[/red]" if c.get("taken") else ""
        lines.append(f"  [cyan]{c['id']}[/cyan] - {c['name']}{taken}")
        lines.append(f"    [dim]{c['backstory'][:80]}...[/dim]")

    return "\n".join(lines)
