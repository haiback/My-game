"""Narrator — round events → personalized Chinese prose via a local LLM.

The engine's GameEvents are the single source of truth; the model only
rephrases them as flavor text. The prompt instructs it to never invent
facts (numbers, items, locations). On failure the caller falls back to
template narrative_seed lines, so narration is never a blocker.
"""

from __future__ import annotations

import logging

from mygame.server.ai.ollama_client import OllamaClient
from mygame.shared.models import Actor, GameEvent, ScenarioDef

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是多人文字生存竞技游戏的叙事者(GM)。根据本回合发生的事实事件, "
    "为指定玩家撰写个人化的中文叙事。严格基于事实, 不得虚构数值、物品或事件。"
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

    async def narrate(
        self,
        actor: Actor,
        round_num: int,
        events: list[GameEvent],
        scenario: ScenarioDef,
    ) -> str:
        prompt = self._build_prompt(actor, round_num, events, scenario)
        text = await self.client.generate(
            self.model,
            prompt,
            system=SYSTEM_PROMPT,
            temperature=self.temperature,
        )
        text = text.strip()
        if not text:
            raise ValueError("Narrator returned empty text")
        return text

    def _build_prompt(
        self,
        actor: Actor,
        round_num: int,
        events: list[GameEvent],
        scenario: ScenarioDef,
    ) -> str:
        lines = [f"第 {round_num} 回合结束。请为该玩家撰写本回合的叙事。"]

        lines.append("")
        lines.append("【玩家信息】")
        char_def = next((c for c in scenario.characters if c.id == actor.character_id), None)
        lines.append(f"- 玩家: {actor.player_name}" + (f" (角色: {char_def.name})" if char_def else ""))
        if char_def and char_def.narration_style:
            lines.append("- 叙事语气: " + char_def.narration_style)
        if char_def and char_def.backstory:
            lines.append("- 角色背景: " + char_def.backstory.replace("\n", " "))
        if char_def and char_def.secret_objective:
            lines.append("- 秘密目标: " + char_def.secret_objective)

        loc = scenario.locations.get(actor.location_id)
        lines.append(f"- 当前位置: {loc.name if loc else actor.location_id}")

        s = actor.stats
        lines.append(f"- 当前状态: HP {s.hp}/{s.max_hp}, 体力 {s.stamina}/{s.max_stamina}")

        if actor.status_effects:
            effects = [f"{e.kind}(剩{e.rounds_left}回合)" for e in actor.status_effects]
            lines.append("- 身上的效果: " + ", ".join(effects))

        if actor.inventory:
            items = []
            for item_id, count in actor.inventory.items():
                item = scenario.items.get(item_id)
                items.append(f"{item.name if item else item_id}x{count}")
            lines.append("- 背包: " + ", ".join(items))

        lines.append("")
        lines.append("【本回合事实事件】(唯一的写作依据)")
        for ev in events:
            marker = "公开" if ev.visibility == "public" else "仅你可见"
            lines.append(f"- [{marker}] ({ev.kind}) {ev.narrative_seed}")

        lines.append("")
        lines.append("写作要求:")
        lines.append("- 以第二人称\"你\"撰写, 2~5 句话。")
        lines.append("- 风格: 紧张、悬疑, 类似《黑色幸存者》的文字冒险叙事, 中文输出。")
        lines.append("- 叙事必须严格采用该角色的\"叙事语气\"(若有), 并贴合\"角色背景\": 语气、视角、用词要体现角色特征, 不同角色的叙事风格应明显不同。")
        lines.append("- 秘密目标只在相关时隐晦地呼应, 不要直接挑明。")
        lines.append("- 严格基于事实事件: 不要编造其他玩家的数值、位置或不在事件中的物品。")
        lines.append("- 直接输出叙事正文, 不要加标题、前缀或任何解释。")

        return "\n".join(lines)
