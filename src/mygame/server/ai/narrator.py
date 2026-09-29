"""Narrator — round events → faction-level Chinese prose via a local LLM.

The engine's GameEvents are the single source of truth; the model only
rephrases them as flavor text. One LLM call per faction per round: the prompt
lists the faction objective (shared) and each member's identity (focus /
narration_style / backstory) so the prose gives every member a spotlight line.
Personal secret objectives never enter the prompt — they stay private and are
injected as template lines.

A post-generation faithfulness check fails closed: if the prose introduces an
Arabic numeral the engine never supplied, narration is rejected and the caller
falls back to template narrative_seed lines, so narration is never a blocker.
"""

from __future__ import annotations

import logging
import re

from mygame.server.ai.ollama_client import OllamaClient
from mygame.shared.models import Actor, FactionDef, GameEvent, ScenarioDef

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是多人文字生存竞技游戏的叙事者(GM)。根据本回合发生的事实事件, "
    "为一个势力的全体成员撰写一段共享的中文叙事。严格基于事实, 不得虚构数值、物品或事件。"
)


class Narrator:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        temperature: float = 0.8,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature

    async def narrate_faction(
        self,
        faction: FactionDef,
        members: list[Actor],
        round_num: int,
        events: list[GameEvent],
        scenario: ScenarioDef,
        *,
        history: list[str] | None = None,
    ) -> str:
        prompt = self._build_faction_prompt(
            faction, members, round_num, events, scenario, history
        )
        text = await self.client.generate(
            self.model,
            prompt,
            system=SYSTEM_PROMPT,
            temperature=self.temperature,
        )
        text = text.strip()
        if not text:
            raise ValueError("Narrator returned empty text")
        if self._numeral_violation(text, prompt):
            raise ValueError("Narration introduced numerals absent from the facts")
        return text

    # -- Prompt building --

    def _build_faction_prompt(
        self,
        faction: FactionDef,
        members: list[Actor],
        round_num: int,
        events: list[GameEvent],
        scenario: ScenarioDef,
        history: list[str] | None,
    ) -> str:
        lines = [f"第 {round_num} 回合结束。请为「{faction.name}」势力撰写本回合的共享叙事。"]

        lines.append("")
        lines.append("【势力信息】")
        lines.append(f"- 势力: {faction.name}")
        if faction.objective is not None and faction.objective.description:
            lines.append("- 势力目标: " + faction.objective.description)

        lines.append("")
        lines.append("【成员】")
        for actor in members:
            char = next(
                (c for c in scenario.characters if c.id == actor.character_id), None
            )
            lines.append("- " + self._member_line(actor, char, scenario))

        if history:
            lines.append("")
            lines.append("【势力前情】(前几回合的势力叙事, 用于承接与呼应)")
            for h in history[-3:]:
                lines.append(f"- {h}")

        lines.append("")
        lines.append("【本回合事实事件】(唯一的写作依据)")
        for ev in events:
            marker = "公开" if ev.visibility == "public" else "仅本势力可见"
            lines.append(f"- [{marker}] ({ev.kind}) {ev.narrative_seed}")

        lines.append("")
        lines.append("写作要求:")
        lines.append("- 写一段「势力共享」的场景叙事, 覆盖本回合事件。")
        lines.append("- 同一场景, 不同身份的成员会注意到不同细节: 按各成员的「身份关注点」与「叙事语气」, 给每个成员一句专属反应。")
        lines.append("- 严格基于事实, 不得虚构数值、物品、位置或事件。")
        lines.append("- 不透露敌对势力的隐藏位置或数值。")
        lines.append("- 中文输出, 直接输出正文, 不要标题、前缀或解释。")
        return "\n".join(lines)

    def _member_line(self, actor: Actor, char, scenario: ScenarioDef) -> str:
        parts: list[str] = [actor.player_name]
        if char is not None:
            parts[0] = f"{actor.player_name}({char.name})"
        loc = scenario.locations.get(actor.location_id)
        parts.append(f"位置: {loc.name if loc else actor.location_id}")
        s = actor.stats
        parts.append(f"HP {s.hp}/{s.max_hp} 体力 {s.stamina}/{s.max_stamina}")
        if char is not None and char.backstory:
            parts.append("背景: " + char.backstory.replace("\n", " "))
        if char is not None and char.narration_style:
            parts.append("叙事语气: " + char.narration_style)
        if char is not None and char.focus:
            parts.append("身份关注点: " + char.focus)
        if actor.status_effects:
            parts.append("效果: " + ", ".join(
                f"{e.kind}(剩{e.rounds_left}回合)" for e in actor.status_effects
            ))
        if actor.inventory:
            items = [
                f"{scenario.items[i].name if i in scenario.items else i}x{n}"
                for i, n in actor.inventory.items()
            ]
            parts.append("背包: " + ", ".join(items))
        return "；".join(parts)

    # -- Faithfulness check --

    def _numeral_violation(self, text: str, source: str) -> bool:
        """True if `text` introduces an Arabic numeral absent from `source`
        (the prompt — everything the engine handed the model)."""
        source_nums = set(re.findall(r"\d+", source))
        text_nums = set(re.findall(r"\d+", text))
        return not text_nums.issubset(source_nums)
