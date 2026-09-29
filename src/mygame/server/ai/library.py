"""NarrativeLibrary — pre-generated "world-side" flavor text.

Location ambience is generated once before a game starts (a single batched
7b request), so ordinary rounds can be narrated from this library without a
per-round LLM call. Any failure leaves the library empty and the caller
falls back to plain narrative_seed templates.
"""

from __future__ import annotations

import json
import logging

from mygame.server.ai.ollama_client import OllamaClient
from mygame.shared.models import ScenarioDef

log = logging.getLogger(__name__)


class NarrativeLibrary:
    def __init__(self, client: OllamaClient, model: str, temperature: float = 0.8):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.location_flavor: dict[str, str] = {}

    async def generate(self, scenario: ScenarioDef) -> None:
        """Pre-generate location ambience for every location in the scenario."""
        if not scenario.locations:
            return
        prompt = self._build_prompt(scenario)
        try:
            text = await self.client.generate(
                self.model, prompt, temperature=self.temperature
            )
            self.location_flavor = self._parse(text, scenario)
        except Exception:
            log.warning(
                "NarrativeLibrary generation failed; falling back to empty",
                exc_info=True,
            )
            self.location_flavor = {}

    def location_text(self, loc_id: str) -> str | None:
        return self.location_flavor.get(loc_id)

    def _build_prompt(self, scenario: ScenarioDef) -> str:
        lines = [
            "你是文字冒险游戏的场景设计者。为下列地点各写一句中文氛围描写，",
            "突出该地点的氛围与危险感，用于玩家进入时渲染环境。",
            "",
        ]
        for loc in scenario.locations.values():
            desc = f"（{loc.description}）" if loc.description else ""
            lines.append(f"- {loc.id}: {loc.name}{desc}")
        lines.append("")
        lines.append(
            "只输出一个 JSON 对象，键为地点 id，值为一句氛围描写。"
            "不要输出任何解释或多余文字。"
        )
        return "\n".join(lines)

    def _parse(self, text: str, scenario: ScenarioDef) -> dict[str, str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}

        result: dict[str, str] = {}
        for loc_id in scenario.locations:
            val = data.get(loc_id)
            if isinstance(val, str) and val.strip():
                result[loc_id] = val.strip()
        return result
