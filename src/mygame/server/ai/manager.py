"""AIService — assembles the Ollama client, circuit breakers, parser and narrator.

Single entry point used by the server: free-text action parsing and round
narration both degrade gracefully (fallback to keyword parsing / template
narrative) whenever Ollama is down, erroring, or the breaker is open.
"""

from __future__ import annotations

from typing import Any

from mygame.server.ai.circuit import CircuitBreaker
from mygame.server.ai.library import NarrativeLibrary
from mygame.server.ai.narrator import Narrator
from mygame.server.ai.ollama_client import OllamaClient
from mygame.server.ai.parser import IntentParser, ParseResult
from mygame.shared.models import Actor, FactionDef, GameEvent, PlayerView, ScenarioDef


class AIService:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}

        self.client = OllamaClient(
            base_url=cfg.get("base_url", "http://localhost:11434"),
            timeout_seconds=cfg.get("timeout_seconds", 30.0),
            max_retries=cfg.get("max_retries", 1),
            num_ctx=cfg.get("num_ctx"),
            num_predict=cfg.get("num_predict"),
        )

        threshold = cfg.get("circuit_breaker_threshold", 3)
        probe_interval = cfg.get("probe_interval_seconds", 60.0)
        self.parser_breaker = CircuitBreaker(threshold=threshold, probe_interval=probe_interval)
        self.narrator_breaker = CircuitBreaker(threshold=threshold, probe_interval=probe_interval)

        self.parser = IntentParser(
            client=self.client,
            model=cfg.get("parser_model", "qwen2.5:3b"),
            temperature=cfg.get("parser_temperature", 0.1),
        )
        self.narrator = Narrator(
            client=self.client,
            model=cfg.get("narrator_model", "qwen2.5:7b"),
            temperature=cfg.get("narrator_temperature", 0.8),
        )
        self.library = NarrativeLibrary(
            client=self.client,
            model=cfg.get("narrator_model", "qwen2.5:7b"),
            temperature=cfg.get("narrator_temperature", 0.8),
        )

    async def parse_action(
        self,
        player_id: str,
        free_text: str,
        view: PlayerView,
        scenario: ScenarioDef,
    ) -> ParseResult:
        """Parse free text. Empty result means: fall back to keyword parser."""
        result = await self.parser_breaker.call(
            lambda: self.parser.parse(player_id, free_text, view, scenario)
        )
        return result if isinstance(result, ParseResult) else ParseResult()

    async def narrate_faction(
        self,
        faction: FactionDef,
        members: list[Actor],
        round_num: int,
        events: list[GameEvent],
        scenario: ScenarioDef,
        *,
        history: list[str] | None = None,
    ) -> str | None:
        """Faction-level round narration. None means: use template lines."""
        return await self.narrator_breaker.call(
            lambda: self.narrator.narrate_faction(
                faction, members, round_num, events, scenario, history=history
            )
        )

    async def close(self) -> None:
        await self.client.close()
