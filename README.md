# WolfPlay

WolfPlay 是由 seihn2 独立设计开发的多智能体战略语言博弈项目，包含完整七人局、LangGraph 非线性状态机、角色视图隔离、Planner-Evaluator-Executor 认知闭环、Reflexion、分级记忆、自博弈、潜在策略聚类、Deep CFR 和 DPO 后训练链路。

仓库提供可运行代码和小规模冒烟配置，但当前未执行正式模型训练、千局采样或统计评测，因此不声明任何胜率提升、策略涌现或“通过训练学会悍跳”的结果。

完整的架构说明、能力边界、验收标准、真实简历表述和面试问答见 [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)。

## 已实现

- **LangGraph 状态机**：夜间狼人、预言家、医生、结算、白天发言、同步投票、胜负判断和条件边循环。
- **标准七人规则**：2 狼人、1 预言家、1 医生、3 村民；狼人达到人数平衡即获胜。
- **认知闭环**：Planner 生成多个候选，Evaluator 评分，Executor 执行，低分或非法动作进入 Reflexion。
- **高级策略候选**：狼人候选策略包含隐藏身份、带票、悍跳预言家和伪造验人信息。
- **分级记忆**：工作记忆、情景记忆、语义角色信念和反思记忆。
- **异步消息总线**：`asyncio.Queue`、Lamport 逻辑时钟、公开/私有/阵营 audience 隔离。
- **离线与模型双后端**：默认启发式 Agent 无需 API；也可连接任意 OpenAI-compatible 聊天接口。
- **潜在策略空间**：支持离线 Hashing Embedding 或 OpenAI-compatible Embedding，并按角色执行确定性 K-Means 聚类。
- **抽象扩展式博弈**：发言使用离散策略簇，夜间和投票保留目标动作，包含角色私有信息、信息集、机会节点和可配置奖励。
- **Deep CFR**：Advantage/Regret Network、Strategy Network、Reservoir Buffer、External-Sampling 遍历、深度限制 rollout、checkpoint 与策略采样。
- **CFR 偏好数据**：将语言候选映射到潜在动作，使用网络预测 advantage 构造 `prompt/chosen/rejected` DPO JSONL。
- **多轮迭代器**：串联“自博弈 → 聚类 → Deep CFR → DPO → 重新采样”，支持断点续跑和逐轮 artifact manifest。
- **DPO/LoRA**：Hugging Face TRL `DPOTrainer` 训练入口，不需要修改对战代码。
- **双阵营对照评测**：challenger 与 baseline 分别控制狼人/村民阵营并交换阵营评测。

## 架构

```mermaid
flowchart LR
    S([START]) --> Setup
    Setup --> Wolf[狼人行动]
    Wolf --> Seer[预言家查验]
    Seer --> Doctor[医生守护]
    Doctor --> Night[夜晚结算]
    Night --> Check{胜负判断}
    Check -->|继续| Announce[白天公告]
    Announce --> Discuss[顺序发言]
    Discuss --> Vote[异步投票]
    Vote --> Resolve[投票结算]
    Resolve --> Check
    Check -->|下一轮| Advance[轮次推进]
    Advance --> Wolf
    Check -->|结束| E([END])
```

每个 Agent 只从 `AsyncMessageBus.events_for(player_id)` 获得可见事件。预言家查验、角色分配和狼人队友信息使用私有 audience，不会出现在其他玩家的观察或记忆中。

## 安装

推荐 Python 3.11：

```bash
uv sync --extra dev
```

训练时再安装 PyTorch/TRL：

```bash
uv sync --extra train
```

## 运行完整对局

默认启发式 Agent，不需要外部模型：

```bash
uv run wolfplay play --seed 42 --max-rounds 8 \
  --output artifacts/demo_game.json
```

也可以使用根目录入口：

```bash
uv run python main.py play --seed 42
```

### 接入聊天模型

```bash
cp .env.example .env
export WOLFPLAY_BASE_URL="https://your-endpoint.example/v1"
export WOLFPLAY_API_KEY="..."
export WOLFPLAY_MODEL="your-model"

uv run wolfplay play --backend openai-compatible
```

模型只负责 Planner 生成候选；Evaluator、规则校验和 Reflexion 仍在本地执行，因此模型输出非法 JSON 或非法动作时会自动回退。

## 潜在策略、Deep CFR 与 DPO

### 1. 生成轨迹

```bash
uv run wolfplay self-play \
  --games 100 \
  --concurrency 4 \
  --seed 2025 \
  --max-rounds 8 \
  --output data/generated/self_play.jsonl
```

每条游戏记录包含：

- 完整事件流和逻辑时间；
- 每名玩家的真实角色和最终胜负；
- 每次决策的 observation prompt；
- Planner 候选动作；
- Evaluator 分数与合法性；
- 最终选择和 Reflexion 内容。

### 2. 构建角色潜在策略空间

默认使用无外部依赖的确定性 Hashing Embedding：

```bash
uv run wolfplay build-latent \
  --input data/generated/self_play.jsonl \
  --output data/generated/latent_space.json \
  --werewolf-clusters 3 \
  --seer-clusters 2 \
  --doctor-clusters 2 \
  --villager-clusters 2
```

也可以设置 `WOLFPLAY_EMBEDDING_BASE_URL`、`WOLFPLAY_EMBEDDING_API_KEY`、`WOLFPLAY_EMBEDDING_MODEL`，再增加 `--embedding-backend openai-compatible` 使用真实 Embedding 服务。

### 3. 训练 Deep CFR

```bash
uv run wolfplay train-deep-cfr \
  --latent-space data/generated/latent_space.json \
  --output-dir checkpoints/wolfplay-cfr \
  --iterations 100 \
  --traversals-per-player 16 \
  --advantage-train-steps 200 \
  --strategy-train-steps 400 \
  --batch-size 256
```

训练会保存角色共享的 Advantage/Regret Network、Strategy Network、Reservoir Buffer 状态、动作目录、潜在策略空间、随机状态和迭代指标。`--max-traversal-depth` 控制 External-Sampling 遍历深度，超过深度后按当前 regret-matching 策略 rollout 到终局。

### 4. 使用 CFR Advantage 构造 DPO 偏好

```bash
uv run wolfplay build-cfr-dpo \
  --input data/generated/self_play.jsonl \
  --checkpoint checkpoints/wolfplay-cfr \
  --output data/generated/dpo_cfr.jsonl
```

该命令按真实对局顺序回放抽象状态，将每个语言候选映射到角色策略簇或离散目标动作，并使用对应信息集上的网络 advantage 排序候选。

仍可使用不依赖 CFR 的启发式偏好构造器：

```bash
uv run wolfplay build-dpo \
  --input data/generated/self_play.jsonl \
  --output data/generated/dpo_pairs.jsonl \
  --outcome-bonus 0.25
```

如只保留胜方轨迹：

```bash
uv run wolfplay build-dpo \
  --input data/generated/self_play.jsonl \
  --output data/generated/dpo_winners.jsonl \
  --winning-only
```

输出格式可直接交给 TRL：

```json
{"prompt":"...","chosen":"...","rejected":"..."}
```

### 5. DPO + LoRA 训练

```bash
uv run wolfplay train-dpo \
  --dataset data/generated/dpo_cfr.jsonl \
  --model Qwen/Qwen3-0.6B \
  --output-dir checkpoints/wolfplay-dpo \
  --epochs 2 \
  --learning-rate 1e-6 \
  --beta 0.1 \
  --batch-size 1 \
  --gradient-accumulation-steps 16
```

默认启用 LoRA。全参数训练增加 `--no-lora`。

### 6. 一键多轮迭代

```bash
uv run wolfplay iterate-policy \
  --output-dir artifacts/iterations \
  --iterations 3 \
  --games-per-iteration 100 \
  --cfr-iterations 100 \
  --dpo-model Qwen/Qwen3-0.6B
```

每轮目录独立保存自博弈 JSONL、潜在策略空间、Deep CFR checkpoint、CFR-DPO 数据、可选 DPO checkpoint 和 manifest。使用外部模型时，第一轮读取 `WOLFPLAY_*`，后续轮次读取 `WOLFPLAY_ITERATION_2_*`、`WOLFPLAY_ITERATION_3_*` 等配置，便于将上一轮训练模型部署后重新采样。

### 7. 汇总胜率

```bash
uv run wolfplay evaluate --input data/generated/self_play.jsonl
```

该命令只统计真实运行产生的数据。简历中的“胜率提升 23%”必须在固定基线、随机种子、模型版本和足够局数下实际测量后才能保留。

### 8. 训练后模型对基线评测

分别配置 `WOLFPLAY_CHALLENGER_*` 与 `WOLFPLAY_BASELINE_*` 后运行：

```bash
uv run wolfplay head-to-head \
  --games-per-side 100 \
  --challenger-backend openai-compatible \
  --baseline-backend openai-compatible \
  --output artifacts/head_to_head.json
```

该命令会让双方交换阵营，并分别输出 challenger 的狼人侧、村民侧和总体胜率。

## 目录

```text
src/wolfplay/
├── abstract_game.py          # 七人抽象博弈、信息集和离散动作空间
├── bus.py                    # 异步消息总线和 Lamport 时钟
├── cfr_preference.py         # CFR advantage 驱动的 DPO 偏好构造
├── cognition.py              # Planner-Evaluator-Executor + Reflexion
├── engine.py                 # LangGraph 游戏状态机
├── evaluation.py             # challenger/baseline 双阵营评测
├── iterative.py              # 多轮采样、聚类、CFR、DPO 编排
├── latent.py                 # Embedding、K-Means 与潜在策略空间
├── llm.py                    # OpenAI-compatible 异步模型后端
├── memory.py                 # 分级记忆与角色信念
├── models.py                 # 状态、事件、动作和轨迹模型
├── preference.py             # 启发式偏好数据构造
├── self_play.py              # 并发自博弈、原子输出与统计
├── cli.py                    # 对战、数据、评估和训练命令
└── training/
    ├── deep_cfr.py           # External-Sampling Deep CFR 与 checkpoint
    ├── dpo.py                # TRL DPO/LoRA 训练入口
    └── torch_models.py       # Regret Network 与 Strategy Network
```

项目早期脚本仍保留在根目录作为历史版本，当前实现统一从 `src/wolfplay/` 启动。

## 测试

```bash
uv run pytest
uv run ruff check src tests main.py
```

## 安全说明

仓库早期历史中曾提交多个疑似真实 API Key。当前工作树已移除这些值，但 Git 历史中的凭据必须由持有者在服务商后台立即吊销；仅删除当前文件不能使旧 Key 失效。
