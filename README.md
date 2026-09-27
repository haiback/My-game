# MyGame

类《黑色幸存者》(Black Survival) 的多人文字冒险对决游戏：多名玩家在孤岛上搜索物资、合成装备、获取知识、相互厮杀，用自由文本下达行动，由本地大模型担任 AI 主持人（解析意图 + 分角色叙事）。

## 特性

- **确定性引擎**：引擎是全知的唯一真相源（位置、血量、背包、胜负）。所有玩家在固定时间窗内密封同轮提交行动，按「类型分层」而非提交时间结算，保证公平。
- **数据驱动剧本**：地图、物品、配方、知识来源、随机事件、角色技能、胜利条件全部来自可替换的 `scenarios/*.yaml`，引擎不硬编码任何玩法。赛博朋克脑机接口、NPC 博士操控地图事件等剧本都能纯 YAML 表达。
- **AI 主持人（本地 Ollama）**：
  - 意图解析（`qwen2.5:3b`）：自由文本 → 结构化行动，含含糊意图的选项澄清。
  - 个性化叙事（`qwen2.5:7b`）：每回合按角色身份（背景 + `narration_style`）生成中文叙事，严守事实不编造。
  - 熔断降级：Ollama 离线时自动回落到关键词解析 + 模板叙事，游戏不中断。
- **知识获取 / 随机事件 / 角色技能**：统一的「效果系统」驱动，事件与技能共用同一套效果原语（伤害、治疗、状态、永久属性成长、授予知识、强制触发事件等）。

## 技术栈

Python 3.11+ · Pydantic v2 · FastAPI + WebSocket（服务端）· `websockets` + Rich（CLI 客户端）· httpx + Ollama（AI 模块）

## 前置依赖

1. **Python 3.11+**
2. **Ollama**（[ollama.com](https://ollama.com)），并拉取两个模型：

   ```bash
   ollama pull qwen2.5:3b   # 意图解析
   ollama pull qwen2.5:7b   # 叙事
   ```

   离线也能玩，只是 AI 解析/叙事会静默降级为模板。

## 安装

```bash
pip install -e ".[server,client,dev]"
```

## 运行

开 3 个终端（均在项目根目录）：

```bash
# 终端 1：服务器（默认 127.0.0.1:8765）
python -m mygame.server.main

# 终端 2：玩家 A
python -m mygame.client.main Alice

# 终端 3：玩家 B
python -m mygame.client.main Bob
```

大厅命令：

```
create [scenario_id]   # 建房（可指定剧本，默认随机）
join <code>            # 加入房间
select <char_id>       # 选择角色
ready                  # 准备
quit                   # 退出
```

两人都 `ready` 后游戏自动开始。决策回合输入自由文本（如 `搜索一下这片沙滩`、`往北边走`、`攻击 Bob`）由 AI 解析；也可输入菜单编号直接行动。意图含糊时 AI 会弹出选项让你澄清。

## 配置

见 `config.yaml`：

- `server.host` / `server.port`：服务监听地址（可用 `MYGAME_HOST` / `MYGAME_PORT` 环境变量覆盖）。
- `ollama.*`：模型名、温度、超时、熔断阈值（`circuit_breaker_threshold` / `probe_interval_seconds`）。
- `game.*`：单回合时长、最大回合数、tick 间隔等。

## 测试

```bash
# 单元测试（无需 Ollama）
python -m pytest -q

# 集成回归（需 Ollama 在线并已拉取模型，否则自动 skip）
python -m pytest -m integration -v
```

## 目录结构

```
config.yaml                       # 运行配置
scenarios/island_of_whispers.yaml # 数据驱动剧本
src/mygame/
  shared/        # 领域模型 + 消息协议
  server/
    ai/          # Ollama 客户端、熔断器、意图解析器、叙事器
    engine/      # 回合状态机、同轮结算、效果系统、胜负判定
    net/         # WebSocket 连接封装
    scenario/    # 剧本加载与校验
    session/     # 大厅 / 房间 / 对局编排
    main.py      # FastAPI 入口
  client/        # Rich CLI 客户端
tests/
```

## 剧本扩展

新剧本 = 新增一个 YAML 文件即可，无需改引擎。关键可数据化维度：

- `knowledge_sources`：知识获取（搜索/到达/合成/事件），可附带永久属性成长（`stat_bonus`）。
- `random_events`：按回合触发的随机事件，技能可通过 `trigger_event` 强制触发。
- 角色 `abilities`：技能效果与随机事件共用同一效果系统。
- 角色 `narration_style`：AI 叙事的语气/视角，决定每个角色读起来的不同「人设」。
