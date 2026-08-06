# WolfPlay 工程设计

## 目标

实现一个可运行、可测试、可扩展的多智能体狼人杀项目，用代码覆盖简历中的核心技术点：LangGraph 非线性状态机、Planner-Evaluator-Executor 认知闭环、Reflexion、分级记忆、异步消息总线、角色视图隔离、逻辑时钟、自博弈轨迹和 DPO 后训练入口。

项目采用论文 *Learning Strategic Language Agents in the Werewolf Game with Iterative Latent Space Policy Optimization* 的七人规则：两名狼人、一名预言家、一名医生和三名村民。工程实现聚焦完整代码链路，不复现论文的大规模 Deep CFR、千局采样和真实胜率实验。

## 架构

对战核心由 LangGraph `StateGraph` 编排。节点依次处理狼人行动、预言家查验、医生守护、夜晚结算、白天公告、发言、投票和胜负判断。胜负判断节点通过条件边路由到白天、下一夜或结束节点，因此不是固定线性脚本。

每个 Agent 只接收由消息总线投影出的个人观察，不接触完整游戏状态。消息总线使用 Lamport 逻辑时钟为事件排序，并通过显式 audience 控制公开、阵营内和私有事件。每个玩家拥有独立的工作记忆、情景记忆、语义信念和反思记忆。

认知闭环生成多个候选策略，由 Evaluator 打分，Executor 转换为合法动作；非法或低质量动作进入 Reflexion 修正。默认启发式后端可以离线运行，OpenAI-compatible HTTP 后端用于接入任意兼容聊天模型。

## 训练链路

自博弈运行器把每次决策的观察、候选策略、评分、最终动作和胜负结果写入 JSONL。偏好构造器选取最高分候选作为 chosen、最低分候选作为 rejected，生成 TRL 可直接读取的 `prompt/chosen/rejected` 数据。

DPO 入口使用 Hugging Face TRL `DPOTrainer`，支持全参数训练和 LoRA。训练依赖放在可选依赖组中，普通对战与测试不需要安装 PyTorch。

## 边界

- 不声称已获得任何胜率提升；仓库只提供评测与训练代码。
- 不使用上游仓库中公开提交的 API Key。
- 默认演示使用启发式 Agent，确保无外部服务也能完整跑通。
