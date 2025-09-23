# AI狼人杀 - 多AI Agent对战平台

基于斗地主项目的LLM逻辑重构，实现多个不同AI模型相互进行狼人杀游戏的智能对战平台。

## 🎯 项目特点

- **多AI模型支持**: 支持GPT、Claude、DeepSeek、通义千问等多种AI模型
- **智能角色扮演**: AI会根据狼人杀角色特点进行策略思考和决策
- **完整游戏规则**: 实现标准狼人杀规则，支持6-12人局
- **灵活配置管理**: 用户可设置n个API，程序提供n个API调用接口
- **实时游戏日志**: 详细记录游戏过程和AI决策思路

## 📦 项目结构

```
werewolf/
├── __init__.py              # 模块初始化
├── llm_manager.py          # 多LLM API管理器
├── game_logic.py           # 狼人杀游戏逻辑
├── ai_player.py            # AI玩家实现
├── game_controller.py      # 游戏控制器
├── config_manager.py       # 配置管理工具
├── main.py                 # 主启动脚本
├── README.md               # 使用说明
├── llm_configs.json        # LLM配置文件
└── game_configs.json       # 游戏配置文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install aiohttp asyncio
```

### 2. 配置API密钥

运行配置管理器：

```bash
cd werewolf
python config_manager.py
```

在配置管理器中添加至少6个不同的LLM API配置，例如：

- **GPT-4o**: OpenAI的GPT-4o模型
- **Claude-3.5**: Anthropic的Claude-3.5-Sonnet
- **DeepSeek**: DeepSeek的聊天模型
- **通义千问**: 通过硅基流动调用的Qwen模型
- **其他模型**: 任何兼容OpenAI格式的API

### 3. 启动游戏

```bash
python main.py
```

选择"快速开始游戏"或"创建自定义游戏"。

## 🔧 配置说明

### LLM配置格式

每个LLM配置包含以下信息：

```json
{
  "name": "配置名称",
  "provider": "openai|anthropic|deepseek|siliconflow|custom",
  "api_key": "你的API密钥",
  "model": "模型名称",
  "base_url": "API地址(可选)",
  "max_tokens": 2000,
  "temperature": 0.8,
  "enabled": true,
  "description": "配置描述"
}
```

### 支持的API提供商

| 提供商 | 说明 | 示例模型 |
|--------|------|----------|
| OpenAI | GPT系列模型 | gpt-4o, gpt-4o-mini |
| Anthropic | Claude系列模型 | claude-3-5-sonnet-20241022 |
| DeepSeek | DeepSeek系列模型 | deepseek-chat |
| SiliconFlow | 硅基流动平台 | Qwen/Qwen2.5-72B-Instruct |
| Custom | 自定义兼容API | 任何OpenAI格式API |

### 游戏配置格式

```json
{
  "name": "游戏配置名称",
  "players": [
    {
      "name": "玩家名称",
      "llm_config": "对应的LLM配置名称"
    }
  ],
  "created_at": 1234567890
}
```

## 🎮 游戏规则

### 支持角色

- **村民**: 通过投票找出狼人
- **狼人**: 夜晚击杀好人，白天伪装
- **预言家**: 夜晚查验玩家身份
- **女巫**: 拥有解药和毒药
- **猎人**: 死亡时可开枪带人
- **守卫**: 夜晚保护玩家

### 游戏流程

1. **夜晚阶段**: 各角色秘密行动
   - 狼人讨论并击杀目标
   - 预言家查验玩家身份
   - 女巫选择救人或下毒
   - 守卫保护玩家

2. **白天阶段**: 公开讨论
   - 宣布夜晚结果
   - 玩家自由发言讨论
   - AI根据角色特点发言

3. **投票阶段**: 投票出局
   - 每个玩家投票
   - 得票最多者出局

### 胜利条件

- **好人获胜**: 所有狼人被投票出局
- **狼人获胜**: 狼人数量≥好人数量

## 🤖 AI特点

### 智能决策

AI玩家会根据以下因素做出决策：

1. **角色特点**: 严格按照角色身份行动
2. **游戏状态**: 分析当前局势和威胁
3. **历史信息**: 记忆之前的行动和发言
4. **策略思考**: 考虑长期和短期利益
5. **心理博弈**: 通过发言影响其他玩家

### 角色扮演

每个角色都有专门的策略指导：

- **狼人**: 伪装村民，保护同伴，制造混乱
- **预言家**: 查验可疑玩家，适时跳身份
- **女巫**: 合理使用解药毒药
- **村民**: 观察分析，跟随预言家指引
- **猎人**: 选择合适时机开枪
- **守卫**: 保护重要角色

## 📊 使用示例

### 基本使用

```python
from werewolf.game_controller import WerewolfGameController

# 配置6个AI玩家
player_configs = [
    {"name": "GPT玩家", "llm_config": "gpt4o"},
    {"name": "Claude玩家", "llm_config": "claude"},
    {"name": "DeepSeek玩家", "llm_config": "deepseek"},
    {"name": "Qwen玩家", "llm_config": "qwen"},
    {"name": "本地模型", "llm_config": "local_model"},
    {"name": "自定义AI", "llm_config": "custom_ai"}
]

# 创建并启动游戏
controller = WerewolfGameController(player_configs)
await controller.initialize_game()
await controller.start_game()
```

### 配置管理

```python
from werewolf.llm_manager import MultiLLMManager, LLMConfig, APIProvider

# 创建管理器
manager = MultiLLMManager()

# 添加配置
config = LLMConfig(
    name="my_gpt",
    provider=APIProvider.OPENAI,
    api_key="sk-your-api-key",
    model="gpt-4o-mini",
    description="我的GPT配置"
)

manager.add_config(config)

# 测试配置
await manager.test_config("my_gpt")
```

## 🛠️ 高级功能

### 批量测试不同AI能力

通过配置不同的AI模型，可以测试：

1. **不同模型的狼人杀策略差异**
2. **各模型的逻辑推理能力**
3. **角色扮演的一致性**
4. **长期记忆和策略规划**

### 自定义AI行为

可以通过修改`ai_player.py`中的提示词来调整AI行为：

- 修改角色指导策略
- 调整决策权重
- 自定义发言风格
- 优化记忆机制

### 扩展游戏规则

在`game_logic.py`中可以扩展：

- 添加新角色
- 修改胜利条件
- 增加特殊事件
- 自定义游戏模式

## 🔍 故障排查

### 常见问题

1. **API调用失败**
   - 检查API密钥是否正确
   - 确认账户余额充足
   - 验证网络连接

2. **配置不生效**
   - 确保配置已启用
   - 检查配置文件格式
   - 重启程序重新加载

3. **游戏卡住**
   - 查看控制台错误信息
   - 检查AI响应格式
   - 降低并发数量

### 调试模式

在`ai_player.py`中可以开启调试：

```python
# 显示完整AI响应
print(f"AI完整响应: {response}")

# 记录决策过程
self._add_to_memory(f"决策过程: {reasoning}")
```

## 📈 性能优化

### 并发控制

- 夜晚行动支持并发执行
- 合理设置思考时间
- 优化API调用频率

### 内存管理

- 限制AI记忆条目数量
- 定期清理游戏日志
- 优化状态存储

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