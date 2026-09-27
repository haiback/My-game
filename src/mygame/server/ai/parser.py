"""IntentParser — free text → structured Action via a local LLM.

The model never mutates state; it only maps natural language onto the
legal actions the engine computed for this player this round. Output is
validated strictly against that list, so a hallucinated action can never
reach the engine. Malformed output yields an empty ParseResult and the
caller falls back to deterministic keyword parsing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from mygame.server.ai.ollama_client import OllamaClient
from mygame.shared.models import (
    Action,
    ActionType,
    LegalAction,
    PlayerView,
    ScenarioDef,
)

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是多人文字生存竞技游戏的行动解析器。把玩家的自然语言输入映射到"
    "给定行动列表中的一项。严格输出 JSON, 参数值必须来自行动列表。"
)


@dataclass
class ClarifyOption:
    label: str
    action: Action


@dataclass
class ParseResult:
    action: Action | None = None
    ambiguous: bool = False
    options: list[ClarifyOption] = field(default_factory=list)


class IntentParser:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        temperature: float = 0.1,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature

    async def parse(
        self,
        player_id: str,
        free_text: str,
        view: PlayerView,
        scenario: ScenarioDef,
    ) -> ParseResult:
        prompt = self._build_prompt(free_text, view, scenario)
        raw = await self.client.generate(
            self.model,
            prompt,
            system=SYSTEM_PROMPT,
            temperature=self.temperature,
        )
        data = self._extract_json(raw)
        if data is None:
            log.warning("Parser produced unparseable JSON for %r: %s", free_text, raw[:200])
            return ParseResult()
        return self._to_result(player_id, data, view, scenario)

    # -- Prompt building --

    def _build_prompt(self, free_text: str, view: PlayerView, scenario: ScenarioDef) -> str:
        lines = ["当前回合的玩家视角信息:"]
        lines.append(f"- 当前位置: {view.location_name}")
        if view.location_description:
            lines.append(f"- 位置描述: {view.location_description}")

        if view.connections:
            conns = []
            for direction, loc_id in view.connections.items():
                loc_name = scenario.locations[loc_id].name if loc_id in scenario.locations else loc_id
                conns.append(f'{direction} → {loc_name}')
            lines.append("- 可去的方向: " + "; ".join(conns))

        s = view.stats
        lines.append(
            f"- 你的状态: HP {s.hp}/{s.max_hp}, 体力 {s.stamina}/{s.max_stamina}, "
            f"攻击 {s.attack}, 防御 {s.defense}"
        )

        if view.status_effects:
            effects = [f"{e.kind}(剩{e.rounds_left}回合)" for e in view.status_effects]
            lines.append("- 身上的效果: " + ", ".join(effects))

        if view.inventory:
            items = []
            for item_id, count in view.inventory.items():
                item = scenario.items.get(item_id)
                items.append(f"{item.name if item else item_id}({item_id})x{count}")
            lines.append("- 背包: " + ", ".join(items))

        if view.equipped_weapon or view.equipped_armor:
            eq = []
            if view.equipped_weapon:
                eq.append(f"武器: {view.equipped_weapon}")
            if view.equipped_armor:
                eq.append(f"护甲: {view.equipped_armor}")
            lines.append("- 装备: " + ", ".join(eq))

        if view.visible_actors:
            others = []
            for va in view.visible_actors:
                others.append(f"{va.player_name}(角色: {va.character_name}, id: {va.player_id})")
            lines.append("- 在场的其他玩家: " + "; ".join(others))

        if view.secret_objective:
            lines.append(f"- 你的秘密目标: {view.secret_objective}")

        lines.append("")
        lines.append("本回合可执行的行动(参数值必须原样使用):")
        for i, la in enumerate(view.legal_actions, start=1):
            lines.append(f"{i}. [{la.type}] {la.label} — 参数: {json.dumps(la.params_schema, ensure_ascii=False)}")

        lines.append("")
        lines.append(f'玩家的输入: "{free_text}"')
        lines.append("")
        lines.append("输出规则:")
        lines.append('1. 意图明确且匹配某个行动时, 输出: {"kind": "action", "type": "行动类型", "params": {参数}}')
        lines.append('2. 意图含糊(如"攻击他"但有多人在场, 或"用那个药"但背包有多种药)时, 输出: '
                     '{"kind": "ambiguous", "options": [{"label": "给玩家的中文选项", "type": "...", "params": {...}}, ...]}, '
                     "给出 2-3 个候选, 每个候选的 type/params 必须对应一个合法行动")
        lines.append('3. 意图无法对应任何行动时, 输出: {"kind": "none"}')
        lines.append("4. 只输出一个 JSON 对象, 不要 markdown 代码块, 不要任何解释。")

        return "\n".join(lines)

    # -- Response handling --

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _to_result(
        self,
        player_id: str,
        data: dict,
        view: PlayerView,
        scenario: ScenarioDef,
    ) -> ParseResult:
        kind = data.get("kind")

        if kind == "ambiguous":
            options = data.get("options")
            if not isinstance(options, list):
                return ParseResult()
            valid: list[ClarifyOption] = []
            for opt in options[:4]:
                if not isinstance(opt, dict):
                    continue
                label = str(opt.get("label", "")).strip()
                action = self._normalize(player_id, view, scenario, opt.get("type"), opt.get("params", {}))
                if label and action:
                    valid.append(ClarifyOption(label=label, action=action))
            if len(valid) >= 2:
                return ParseResult(ambiguous=True, options=valid)
            if len(valid) == 1:
                return ParseResult(action=valid[0].action)
            return ParseResult()

        if kind != "action":
            return ParseResult()

        action = self._normalize(player_id, view, scenario, data.get("type"), data.get("params", {}))
        return ParseResult(action=action)

    # -- Validation against engine-computed legal actions --

    def _normalize(
        self,
        player_id: str,
        view: PlayerView,
        scenario: ScenarioDef,
        type_value,
        params,
    ) -> Action | None:
        if not isinstance(type_value, str):
            return None
        try:
            action_type = ActionType(type_value)
        except ValueError:
            return None
        if not isinstance(params, dict):
            params = {}

        legal = [la for la in view.legal_actions if la.type == action_type]
        if not legal:
            return None

        match action_type:
            case ActionType.MOVE:
                direction = str(params.get("direction", "")).strip()
                if not self._matches_schema(legal, {"direction": direction}):
                    return None
                return Action(player_id=player_id, type=action_type,
                              params={"direction": direction}, parse_confidence=0.9)

            case ActionType.SEARCH | ActionType.REST | ActionType.WAIT:
                return Action(player_id=player_id, type=action_type, parse_confidence=0.95)

            case ActionType.USE:
                item_id = str(params.get("item_id", "")).strip()
                if not item_id or not self._matches_schema(legal, {"item_id": item_id}):
                    item_id = self._resolve_item_id(params, view, scenario) or ""
                if not item_id:
                    return None
                matches = [la for la in legal if la.params_schema.get("item_id") == item_id]
                if not matches:
                    return None
                out: dict = {"item_id": item_id}
                if matches[0].params_schema.get("equip"):
                    out["equip"] = True
                return Action(player_id=player_id, type=action_type,
                              params=out, parse_confidence=0.9)

            case ActionType.CRAFT:
                idx = params.get("recipe_index")
                if isinstance(idx, str) and idx.strip().isdigit():
                    idx = int(idx.strip())
                if not isinstance(idx, int):
                    return None
                if not self._matches_schema(legal, {"recipe_index": idx}):
                    return None
                return Action(player_id=player_id, type=action_type,
                              params={"recipe_index": idx}, parse_confidence=0.9)

            case ActionType.ATTACK:
                target_id = self._resolve_target(params.get("target"), view)
                if not target_id:
                    return None
                if not self._matches_schema(legal, {"target": target_id}):
                    return None
                return Action(player_id=player_id, type=action_type,
                              params={"target": target_id}, parse_confidence=0.9)

            case ActionType.SPECIAL:
                ability_id = str(params.get("ability_id", "")).strip()
                if not ability_id or not self._matches_schema(legal, {"ability_id": ability_id}):
                    return None
                out = {"ability_id": ability_id}
                target_id = self._resolve_target(params.get("target"), view)
                if target_id and self._matches_schema(legal, {"ability_id": ability_id}):
                    out["target"] = target_id
                return Action(player_id=player_id, type=action_type,
                              params=out, parse_confidence=0.9)

            case _:
                return None

    @staticmethod
    def _matches_schema(legal: list[LegalAction], required: dict) -> bool:
        return any(
            all(la.params_schema.get(k) == v for k, v in required.items())
            for la in legal
        )

    @staticmethod
    def _resolve_target(target, view: PlayerView) -> str | None:
        if not isinstance(target, str) or not target.strip():
            return None
        target = target.strip()
        for va in view.visible_actors:
            if target == va.player_id:
                return va.player_id
        for va in view.visible_actors:
            if target in va.player_name or target in va.character_name:
                return va.player_id
        return None

    @staticmethod
    def _resolve_item_id(params: dict, view: PlayerView, scenario: ScenarioDef) -> str | None:
        for key in ("item_id", "item", "item_name", "name"):
            text = params.get(key)
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()
            if text in view.inventory:
                return text
            for item_id, item in scenario.items.items():
                if text == item.name or text in item.name:
                    return item_id
        return None
