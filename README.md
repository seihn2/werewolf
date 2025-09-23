# AI狼人杀 - 多AI Agent对战平台

基于斗地主项目的LLM逻辑重构，实现多个不同AI模型相互进行狼人杀游戏的智能对战平台。

## 🎯 项目特点

- **多AI模型支持**: 支持GPT、Claude、DeepSeek、通义千问等多种AI模型
- **智能角色扮演**: AI会根据狼人杀角色特点进行策略思考和决策
- **完整游戏规则**: 实现标准狼人杀规则，支持6-12人局
- **灵活配置管理**: 用户可设置n个API，程序提供n个API调用接口
- **实时游戏日志**: 详细记录游戏过程和AI决策思路
- **隐私保护**: 严格遵循身份隐藏规则，防止信息泄露

## 📦 项目结构

```
werewolf/
├── __init__.py                    # 模块初始化
├── main.py                       # 主启动脚本
├── enhanced_game_controller.py   # 增强版游戏控制器
├── game_controller.py            # 基础游戏控制器
├── round_manager.py              # 游戏轮次管理器
├── game_logic.py                 # 狼人杀游戏逻辑
├── ai_player.py                  # AI玩家实现
├── bot_memory.py                 # AI记忆系统
├── conversation_manager.py       # 对话管理系统
├── night_actions.py              # 夜间行动管理
├── llm_manager.py                # 多LLM API管理器
├── api_config.py                 # API配置管理
├── config_manager.py             # 配置管理工具
├── game_logger.py                # 游戏日志记录
├── name_generator.py             # 随机名字生成器
├── change_model.py               # 批量模型更换工具
├── simple_update.py              # 简单模型更新工具
├── bot_generator.py              # Bot批量生成工具
├── simple_bot_gen.py             # 简单Bot生成工具
├── update_models.py              # 模型更新工具
├── fix_encoding.py               # 编码问题修复工具
├── api_configs.json              # API配置文件
└── README.md                     # 使用说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install aiohttp asyncio
```

### 2. 配置API密钥

编辑 `api_configs.json` 文件，添加至少6个不同的LLM API配置：

```json
{
  "genshin1": {
    "name": "genshin1",
    "provider": "siliconflow",
    "api_key": "your-api-key",
    "base_url": "https://api.siliconflow.cn/v1",
    "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "description": "",
    "is_default": true
  }
}
```

### 3. 启动游戏

```bash
python main.py
```

选择对应的游戏模式开始AI狼人杀游戏。

## 🛠️ 工具说明

### 核心游戏文件

| 文件 | 作用 | 说明 |
|------|------|------|
| `main.py` | 主启动脚本 | 游戏入口，提供菜单选择和游戏启动 |
| `enhanced_game_controller.py` | 增强版游戏控制器 | 支持N人局，完整投票系统，增强AI发言 |
| `game_controller.py` | 基础游戏控制器 | 基础版游戏流程控制 |
| `round_manager.py` | 轮次管理器 | 管理游戏的夜晚和白天阶段 |
| `game_logic.py` | 游戏逻辑 | 核心狼人杀规则和状态管理 |
| `ai_player.py` | AI玩家 | AI角色扮演和决策系统 |
| `night_actions.py` | 夜间行动 | AI控制的预言家查验、女巫行动 |

### 配置和管理工具

| 文件 | 作用 | 使用方法 |
|------|------|----------|
| `config_manager.py` | 配置管理界面 | `python config_manager.py` - 图形化配置管理 |
| `api_config.py` | API配置核心 | 提供配置读写和管理功能 |
| `change_model.py` | 批量模型更换 | `python change_model.py <模型名> [提供商]` |
| `simple_update.py` | 简单模型更新 | 快速更新所有Bot到指定模型 |
| `bot_generator.py` | Bot批量生成 | 生成多个Bot配置 |
| `simple_bot_gen.py` | 简单Bot生成 | 快速生成Bot配置 |
| `update_models.py` | 模型更新工具 | 各种模型更新操作 |

### 辅助系统

| 文件 | 作用 | 说明 |
|------|------|------|
| `bot_memory.py` | AI记忆系统 | 管理AI的游戏记忆和历史信息 |
| `conversation_manager.py` | 对话管理 | 处理AI之间的对话和发言 |
| `game_logger.py` | 游戏日志 | 记录游戏过程和事件 |
| `name_generator.py` | 名字生成器 | 提供100个随机AI玩家名字 |
| `llm_manager.py` | LLM管理器 | 多LLM API调用和管理 |
| `fix_encoding.py` | 编码修复 | 修复中文编码问题 |

## 🔧 工具使用示例

### 1. 批量更换AI模型

```bash
# 更换所有Bot到Qwen模型
python change_model.py Qwen/Qwen3-Next-80B-A3B-Instruct

# 更换到OpenAI模型
python change_model.py gpt-4o-mini openai

# 查看当前配置状态
python change_model.py --status

# 查看可用模型列表
python change_model.py --list
```

### 2. 配置管理

```bash
# 启动配置管理界面
python config_manager.py
```

### 3. 快速更新模型

```bash
# 运行简单更新脚本
python simple_update.py
```

### 4. 生成Bot配置

```bash
# 生成多个Bot配置
python bot_generator.py

# 简单Bot生成
python simple_bot_gen.py
```

## 🎮 游戏规则

### 支持角色

- **村民**: 通过投票找出狼人
- **狼人**: 夜晚击杀好人，白天伪装
- **预言家**: 夜晚查验玩家身份（AI自动决策）
- **女巫**: 拥有解药和毒药（AI自动决策）

### 支持人数

- **6人局**: 2狼人 + 4好人（1预言家 + 1女巫 + 2村民）
- **7人局**: 2狼人 + 5好人（1预言家 + 1女巫 + 3村民）
- **8人局**: 3狼人 + 5好人（1预言家 + 1女巫 + 3村民）
- **9人局**: 3狼人 + 6好人（1预言家 + 1女巫 + 4村民）
- **10人局**: 4狼人 + 6好人（1预言家 + 1女巫 + 4村民）
- **11人局**: 4狼人 + 7好人（1预言家 + 1女巫 + 5村民）
- **12人局**: 4狼人 + 8好人（1预言家 + 1女巫 + 6村民）

### 游戏流程

1. **夜晚阶段**: 各角色秘密行动
   - 狼人讨论并击杀目标（AI自动决策）
   - 预言家查验玩家身份（AI自动选择目标）
   - 女巫选择救人或下毒（AI自动决策）

2. **白天阶段**: 公开讨论
   - 宣布夜晚结果（不泄露角色身份）
   - AI玩家发言讨论（强制长篇发言）
   - 每个AI都会基于角色身份发言

3. **投票阶段**: 投票出局
   - 每个AI玩家投票（AI自动决策）
   - 得票最多者出局
   - 出局时不公开身份

### 胜利条件

- **好人获胜**: 所有狼人被投票出局
- **狼人获胜**: 狼人数量≥好人数量

### 隐私保护

- 游戏开始时不显示角色分配
- 夜间行动不泄露角色身份
- 预言家查验结果只对预言家可见
- 投票出局时不立即公开身份
- 只在游戏结束时公开所有角色

## 🤖 AI特点

### 智能决策

AI玩家具备以下能力：

1. **角色认知**: 明确自己的角色和目标
2. **策略思考**: 根据游戏状态制定策略
3. **记忆系统**: 记住游戏过程中的重要信息
4. **伪装能力**: 狼人会伪装成好人
5. **推理分析**: 根据发言和投票分析其他玩家
6. **长篇发言**: 每次发言都包含丰富的分析内容

### 夜间自动行动

- **预言家**: AI自动选择查验目标，优先可疑玩家
- **女巫**: AI自动决定是否使用解药和毒药
- **狼人**: AI协商选择击杀目标

## 📊 API配置

### 配置文件格式 (`api_configs.json`)

```json
{
  "配置名": {
    "name": "配置名",
    "provider": "siliconflow|openai|anthropic",
    "api_key": "你的API密钥",
    "base_url": "API基础URL",
    "model": "模型名称",
    "description": "配置描述",
    "is_default": true/false
  }
}
```

### 支持的提供商

| 提供商 | Provider值 | 常用模型 |
|--------|------------|----------|
| SiliconFlow | `siliconflow` | `Qwen/Qwen3-Next-80B-A3B-Instruct` |
| OpenAI | `openai` | `gpt-4o`, `gpt-4o-mini` |
| Anthropic | `anthropic` | `claude-3-5-sonnet-20241022` |

## 🔍 故障排查

### 常见问题

1. **编码错误**: 运行 `python fix_encoding.py` 修复
2. **API调用失败**: 检查API密钥和网络连接
3. **模型不支持**: 使用 `change_model.py --list` 查看支持的模型
4. **游戏卡住**: 检查AI响应超时设置

### 调试技巧

- 查看控制台输出的详细错误信息
- 使用 `--status` 参数检查配置状态
- 确保至少有6个有效的API配置

## 🤝 贡献指南

欢迎提交Issue和Pull Request：

1. **报告Bug**: 详细描述问题和复现步骤
2. **功能建议**: 提出新功能需求
3. **代码贡献**: 遵循现有代码风格
4. **文档完善**: 改进使用说明

## 📄 许可证

本项目基于原斗地主项目的许可证，仅供学习和研究使用。

## 🙏 致谢

- 基于原AI斗地主项目的LLM逻辑框架
- 感谢各大AI模型提供商的API支持
- 参考了经典狼人杀游戏规则

---

**开始你的AI狼人杀之旅吧！🐺**