"""Narrator — round events → personalized Chinese prose via a local LLM.

The engine's GameEvents are the single source of truth; the model only
rephrases them as flavor text. The prompt instructs it to never invent
facts (numbers, items, locations). On failure the caller falls back to
template narrative_seed lines, so narration is never a blocker.

To make narration feel alive and connected, each call may be given extra
*context* (all optional, all grounded in real state):
  - world_summary:  a shared "world-side" summary of this round's public
                    events (produced once by narrate_world), so every
                    player's personal prose is anchored to the same facts.
  - others_summary: where the other players are and whether they're alive,
                    so a character can react to the same shared space.
  - upcoming_threat: the next scheduled random event, framed as a vague
                    premonition (never a confirmed future fact).
  - history:        this player's previous-round narration, for continuity
                    and foreshadowing across rounds.
The character's `focus` field (identity-driven attention) steers *what* the
prose emphasises, on top of `narration_style` (how it sounds).
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

WORLD_SYSTEM_PROMPT = (
    "你是多人文字生存竞技游戏的叙事者(GM)。根据本回合公开的事实事件, "
    "写一段客观的世界动态摘要。严格基于事实, 不得虚构数值、物品或事件。"
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

    async def narrate_world(
        self,
        round_num: int,
        public_events: list[GameEvent],
    ) -> str:
        """One shared summary of a round's public events, used as the anchor
        for every player's personal narration. Returns "" if there are no
        public events to summarise."""
        if not public_events:
            return ""
        prompt = self._build_world_prompt(round_num, public_events)
        text = await self.client.generate(
            self.model,
            prompt,
            system=WORLD_SYSTEM_PROMPT,
            temperature=self.temperature,
        )
        text = text.strip()
        if not text:
            raise ValueError("Narrator returned empty world summary")
        return text

    async def narrate(
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
    ) -> str:
        prompt = self._build_prompt(
            actor, round_num, events, scenario,
            world_summary=world_summary,
            others_summary=others_summary,
            upcoming_threat=upcoming_threat,
            history=history,
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
        return text

    # -- Prompt building --

    def _build_world_prompt(
        self,
        round_num: int,
        public_events: list[GameEvent],
    ) -> str:
        lines = [f"第 {round_num} 回合结束。请客观概述这一回合公开可见的动态。"]
        lines.append("")
        lines.append("【本回合公开事件】(唯一的写作依据)")
        for ev in public_events:
            lines.append(f"- ({ev.kind}) {ev.narrative_seed}")
        lines.append("")
        lines.append("写作要求:")
        lines.append("- 用 GM 视角、第三人称, 2~4 句。")
        lines.append("- 只概述公开事件, 不透露任何玩家的私密行动、数值或背包内容。")
        lines.append("- 语气中性客观, 中文输出, 直接输出正文, 不要标题或解释。")
        return "\n".join(lines)

    def _build_prompt(
        self,
        actor: Actor,
        round_num: int,
        events: list[GameEvent],
        scenario: ScenarioDef,
        *,
        world_summary: str | None,
        others_summary: str | None,
        upcoming_threat: str | None,
        history: list[str] | None,
    ) -> str:
        lines = [f"第 {round_num} 回合结束。请为该玩家撰写本回合的个人化叙事。"]

        char_def = next(
            (c for c in scenario.characters if c.id == actor.character_id), None
        )

        lines.append("")
        lines.append("【玩家信息】")
        lines.append(
            f"- 玩家: {actor.player_name}"
            + (f" (角色: {char_def.name})" if char_def else "")
        )
        if char_def and char_def.backstory:
            lines.append("- 角色背景: " + char_def.backstory.replace("\n", " "))
        if char_def and char_def.narration_style:
            lines.append("- 叙事语气: " + char_def.narration_style)
        if char_def and char_def.focus:
            lines.append("- 身份关注点: " + char_def.focus)
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

        if history:
            lines.append("")
            lines.append("【你的前情】(上一回合你自己的叙事, 用于承接与呼应)")
            for h in history[-3:]:
                lines.append(f"- {h}")

        if others_summary:
            lines.append("")
            lines.append("【同场态势】(同一时间与空间中其他玩家的公开状态)")
            for part in others_summary.splitlines():
                lines.append(f"- {part}")

        if upcoming_threat:
            lines.append("")
            lines.append("【隐约的预感】(可写入叙事的伏笔, 用'预感/征兆/低语'等模糊方式暗示, 不要当成已发生的事实宣布)")
            lines.append(f"- {upcoming_threat}")

        lines.append("")
        lines.append("【本回合事实事件】(唯一的写作依据)")
        if world_summary:
            lines.append("- [世界动态] " + world_summary)
        for ev in events:
            marker = "公开" if ev.visibility == "public" else "仅你可见"
            lines.append(f"- [{marker}] ({ev.kind}) {ev.narrative_seed}")

        lines.append("")
        lines.append("写作要求:")
        lines.append("- 以第二人称\"你\"撰写, 2~5 句话。")
        lines.append("- 风格: 紧张、悬疑, 类似《黑色幸存者》的文字冒险叙事, 中文输出。")
        lines.append("- 严格采用该角色的\"叙事语气\"(若有), 并贴合\"身份关注点\": 同一现场, 不同身份会注意到不同细节。")
        lines.append("- 可基于\"世界动态\"与\"前情\"做承接与呼应, 但不得编造其他玩家的数值、位置或不在事件中的物品。")
        lines.append("- 秘密目标只在相关时隐晦地呼应, 不要直接挑明。")
        lines.append("- \"隐约的预感\"仅用于营造氛围, 不得写成确凿的预言。")
        lines.append("- 直接输出叙事正文, 不要加标题、前缀或任何解释。")

        return "\n".join(lines)
