"""AIService — assembles the Ollama client, circuit breakers, parser and narrator.

Single entry point used by the server: free-text action parsing and round
narration both degrade gracefully (fallback to keyword parsing / template
narrative) whenever Ollama is down, erroring, or the breaker is open.
"""

from __future__ import annotations

from typing import Any

from mygame.server.ai.circuit import CircuitBreaker
from mygame.server.ai.narrator import Narrator
from mygame.server.ai.ollama_client import OllamaClient
from mygame.server.ai.parser import IntentParser, ParseResult
from mygame.shared.models import Actor, GameEvent, PlayerView, ScenarioDef


class AIService:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}

        self.client = OllamaClient(
            base_url=cfg.get("base_url", "http://localhost:11434"),
            timeout_seconds=cfg.get("timeout_seconds", 30.0),
            max_retries=cfg.get("max_retries", 1),
        )

        threshold = cfg.get("circuit_breaker_threshold", 3)
        probe_interval = cfg.get("probe_interval_seconds", 60.0)
        self.parser_breaker = CircuitBreaker(threshold=threshold, probe_interval=probe_interval)
        self.narrator_breaker = CircuitBreaker(threshold=threshold, probe_interval=probe_interval)
        self.world_breaker = CircuitBreaker(threshold=threshold, probe_interval=probe_interval)

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

    async def narrate_world(
        self,
        round_num: int,
        public_events: list[GameEvent],
    ) -> str | None:
        """Shared world-side summary of a round's public events. None/"" means
        the caller should skip the anchor and narrate each player directly."""
        if not public_events:
            return None
        result = await self.world_breaker.call(
            lambda: self.narrator.narrate_world(round_num, public_events)
        )
        if isinstance(result, str) and result.strip():
            return result
        return None

    async def narrate_round(
        self,
        actor: Actor,
        round_num: int,
        events: list[GameEvent],
        scenario: ScenarioDef,
        *,
        world_summary: str | None = None,
        others_summary: str | None = None,
        upcoming_threat: str | None = None,
        history: list[str] | None = None,
    ) -> str | None:
        """Personalized round narration. None means: use template lines."""
        return await self.narrator_breaker.call(
            lambda: self.narrator.narrate(
                actor, round_num, events, scenario,
                world_summary=world_summary,
                others_summary=others_summary,
                upcoming_threat=upcoming_threat,
                history=history,
            )
        )

    async def close(self) -> None:
        await self.client.close()
