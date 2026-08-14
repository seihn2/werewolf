# WolfPlay 潜在策略与 Deep CFR 设计

## 目标

在不依赖大规模实验的前提下，为 WolfPlay 补齐可运行的潜在策略学习代码：发言 Embedding、按角色 K-Means 聚类、七人抽象博弈、External-Sampling Deep CFR、模型与 Buffer checkpoint、CFR 驱动 DPO 数据，以及多轮“采样 → 聚类 → 求解 → 对齐 → 重新采样”编排。

## 方案选择

### 方案 A：仅从历史轨迹拼接经验博弈树

实现成本低，但未被选中的动作没有真实后继状态，CFR 只能在历史覆盖范围内优化，容易退化为离线评分器。

### 方案 B：原生抽象狼人杀状态机（采用）

新增与现有七人规则一致的可复制状态：角色分配、夜间行动、白天潜在策略发言、同步投票、平票机会节点和终局奖励。发言动作由每个角色的离散聚类表示，夜间和投票动作保留离散目标。该状态机实现标准 extensive-form game 接口，可被 External-Sampling Deep CFR 真实遍历。

### 方案 C：直接在 LangGraph 对战图中做 CFR 回溯

能最大程度复用运行时，但异步消息、LLM 调用、记忆副作用和图状态复制会显著增加遍历成本与测试难度，因此不采用。

## 模块

1. `latent.py`：文本向量接口、离线 Hashing Embedder、OpenAI-compatible Embedding 客户端、K-Means、角色潜在策略空间和 JSON 持久化。
2. `abstract_game.py`：固定动作目录、角色私有信息、公开潜在策略历史、信息集向量、机会节点、合法动作和奖励累计。
3. `training/deep_cfr.py`：Reservoir Buffer、External-Sampling 遍历、Regret Matching、深度限制 rollout、训练循环、checkpoint 和平均策略采样。
4. `training/torch_models.py`：Advantage/Regret Network 与 Strategy Network，仅在安装 `train` 依赖后加载。
5. `cfr_preference.py`：回放语言自博弈轨迹，将候选映射到潜在动作，并使用 CFR 网络输出构造 `prompt/chosen/rejected`。
6. `iterative.py`：把自博弈、聚类、Deep CFR、DPO 数据、可选 DPO 训练和下一轮重新采样串成可恢复的多轮任务。

## Deep CFR 细节

- 对每个座位轮流作为 traverser；同角色座位共享网络和经验池。
- traverser 节点枚举所有合法动作，其他玩家和机会节点仅采样一个动作。
- Advantage Memory 与 Strategy Memory 使用标准 Reservoir Sampling。
- Advantage 网络输出全局动作空间的即时 advantage；Regret Matching 只在合法动作上归一化正值。
- Strategy 网络拟合按迭代权重加权的平均策略。
- 遍历达到深度上限后，按当前策略 rollout 到终局，避免完整七人树爆炸。
- checkpoint 保存动作目录、潜在策略空间、网络、训练配置、迭代数、随机状态和可选 buffer。

## 验收方式

通过自博弈采样、聚类质量、Deep CFR checkpoint、CFR-DPO 数据、策略采样和双阵营对照评测验证完整训练链路。
