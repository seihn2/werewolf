"""
狼人杀AI玩家实现
基于斗地主项目的LLM逻辑，实现智能的狼人杀AI玩家
"""

import json
import asyncio
import time
from typing import List, Dict, Any, Optional
from .llm_manager import LLMAPIClient, LLMConfig
from .game_logic import Player, GameAction, ActionType, Role, GamePhase, GameState


class WerewolfAIPlayer:
    """狼人杀AI玩家基类"""

    def __init__(self, player: Player, llm_config: LLMConfig):
        self.player = player
        self.llm_config = llm_config
        self.memory: List[str] = []  # AI记忆系统
        self.strategy_notes: Dict[str, Any] = {}  # 策略笔记
        self.thinking_time = 3.0  # AI思考时间

    async def decide_action(self, game_state: GameState, available_actions: List[ActionType]) -> Optional[GameAction]:
        """决定行动"""
        if not self.player.is_alive:
            return None

        print(f"\n{self.player.name} 正在思考...")
        await asyncio.sleep(self.thinking_time)

        try:
            # 构建提示词
            prompt = self._build_decision_prompt(game_state, available_actions)

            # 调用LLM
            response = await self._call_llm(prompt)
            print(f"  {self.player.name}: AI响应长度: {len(response)}字符")

            # 解析响应
            action = self._parse_response(response, available_actions)

            if action:
                # 记录到记忆中
                self._add_to_memory(f"我的行动: {action.action_type.value} -> {action.target_id}")

            return action

        except Exception as e:
            print(f"{self.player.name}: AI决策失败 ({e})，使用随机策略")
            return self._fallback_action(available_actions)

    async def make_speech(self, game_state: GameState, speech_context: str) -> str:
        """发言"""
        if not self.player.is_alive:
            return ""

        try:
            prompt = self._build_speech_prompt(game_state, speech_context)
            response = await self._call_llm(prompt)

            # 解析发言内容
            speech = self._parse_speech_response(response)
            self._add_to_memory(f"我的发言: {speech}")

            return speech

        except Exception as e:
            print(f"{self.player.name}: AI发言失败 ({e})")
            return self._fallback_speech()

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM API"""
        system_prompt = self._build_system_prompt()
        async with LLMAPIClient(self.llm_config) as client:
            return await client.chat_completion(prompt, system_prompt)

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        role_desc = self._get_role_description()
        return f"""你是一个顶级的狼人杀AI玩家，扮演{role_desc}。

## 核心原则
1. **保持角色一致性**: 严格按照你的角色身份行动和发言
2. **逻辑推理**: 基于已知信息进行合理推理
3. **策略思考**: 考虑长期和短期利益
4. **信息管理**: 合理透露和隐藏信息
5. **心理博弈**: 通过发言影响其他玩家的判断

## 重要规则
- 严格遵守狼人杀游戏规则
- 不能透露自己的真实身份（除非策略需要）
- 基于逻辑和观察做出决策
- 保持合理的人类玩家行为模式

## 响应格式
所有决策必须以JSON格式返回，包含reasoning字段解释你的思考过程。"""

    def _get_role_description(self) -> str:
        """获取角色描述"""
        descriptions = {
            Role.VILLAGER: "村民 - 你的目标是找出所有狼人并投票出局他们",
            Role.WEREWOLF: "狼人 - 你的目标是消灭好人并隐藏身份，与其他狼人合作",
            Role.SEER: "预言家 - 你可以在夜晚查验一个玩家的身份，引导村民找出狼人",
            Role.WITCH: "女巫 - 你有一瓶解药和一瓶毒药，可以救人或杀人",
            Role.HUNTER: "猎人 - 你死亡时可以开枪带走一个玩家",
            Role.GUARD: "守卫 - 你可以在夜晚保护一个玩家免受狼人攻击"
        }
        return descriptions.get(self.player.role, "未知角色")

    def _build_decision_prompt(self, game_state: GameState, available_actions: List[ActionType]) -> str:
        """构建决策提示词"""
        # 游戏状态信息
        alive_players = game_state.get_alive_players()
        game_info = f"""## 当前游戏状态
- 回合: 第{game_state.current_round}轮
- 阶段: {game_state.current_phase.value}
- 存活玩家: {len(alive_players)}人
- 玩家列表: {[f"{p.name}(ID:{p.id})" for p in alive_players]}

## 我的信息
- 姓名: {self.player.name}
- ID: {self.player.id}
- 角色: {self.player.role.value}
- 状态: {'存活' if self.player.is_alive else '死亡'}"""

        # 可用行动
        actions_info = f"""
## 可用行动
{[action.value for action in available_actions]}"""

        # 记忆信息
        memory_info = ""
        if self.memory:
            recent_memory = self.memory[-5:]  # 最近5条记忆
            memory_info = f"""
## 我的记忆 (最近)
{chr(10).join(recent_memory)}"""

        # 游戏日志
        log_info = ""
        if game_state.game_log:
            recent_logs = game_state.game_log[-3:]  # 最近3条日志
            log_info = f"""
## 最近游戏事件
{chr(10).join(recent_logs)}"""

        # 角色特定的决策指导
        role_guidance = self._get_role_guidance(game_state, available_actions)

        prompt = f"""{game_info}{actions_info}{memory_info}{log_info}

{role_guidance}

请基于以上信息做出最优决策。必须以JSON格式返回:
{{
    \"action\": \"kill/check/save/poison/guard/vote/pass\",
    \"target_id\": 目标玩家ID(数字)或null,
    \"reasoning\": \"详细的决策理由，体现你的角色特点和策略思考\",
    \"confidence\": 0.85
}}"""

        return prompt

    def _get_role_guidance(self, game_state: GameState, available_actions: List[ActionType]) -> str:
        """获取角色特定的决策指导"""
        if self.player.role == Role.WEREWOLF:
            return self._get_werewolf_guidance(game_state, available_actions)
        elif self.player.role == Role.SEER:
            return self._get_seer_guidance(game_state, available_actions)
        elif self.player.role == Role.WITCH:
            return self._get_witch_guidance(game_state, available_actions)
        elif self.player.role == Role.VILLAGER:
            return self._get_villager_guidance(game_state, available_actions)
        elif self.player.role == Role.HUNTER:
            return self._get_hunter_guidance(game_state, available_actions)
        elif self.player.role == Role.GUARD:
            return self._get_guard_guidance(game_state, available_actions)
        else:
            return "## 角色指导\\n按照你的角色特点进行决策。"

    def _get_werewolf_guidance(self, game_state: GameState, available_actions: List[ActionType]) -> str:
        """狼人决策指导"""
        other_wolves = [p for p in game_state.get_players_by_role(Role.WEREWOLF) if p.id != self.player.id]
        good_guys = [p for p in game_state.get_alive_players() if p.role != Role.WEREWOLF]

        return f"""## 狼人策略指导
- 同伴狼人: {[f"{p.name}(ID:{p.id})" for p in other_wolves]}
- 目标好人: {[f"{p.name}(ID:{p.id})" for p in good_guys]}

### 夜晚击杀优先级:
1. 预言家/女巫等神职
2. 发言有逻辑且怀疑你的村民
3. 能力较强的玩家

### 白天策略:
1. 伪装成村民，表现出寻找狼人的积极性
2. 适当怀疑好人，制造混乱
3. 保护同伴，但不要过于明显
4. 引导投票给好人"""

    def _get_seer_guidance(self, game_state: GameState, available_actions: List[ActionType]) -> str:
        """预言家决策指导"""
        unknown_players = [p for p in game_state.get_alive_players()
                          if p.id != self.player.id and not self._is_checked(p.id)]

        return f"""## 预言家策略指导
- 待查验玩家: {[f"{p.name}(ID:{p.id})" for p in unknown_players]}

### 查验优先级:
1. 发言可疑的玩家
2. 较为沉默的玩家
3. 逻辑不清晰的玩家

### 身份策略:
1. 适时跳出身份，报告查验结果
2. 观察狼人的反应和投票
3. 团结村民，建立信任"""

    def _get_witch_guidance(self, game_state: GameState, available_actions: List[ActionType]) -> str:
        """女巫决策指导"""
        return """## 女巫策略指导
### 解药使用:
1. 优先救重要角色（预言家、守卫等）
2. 考虑救自己以保存毒药
3. 不要浪费解药在不重要的角色上

### 毒药使用:
1. 毒死确认的狼人
2. 在关键时刻使用以平衡局势
3. 考虑毒死可疑度最高的玩家"""

    def _get_villager_guidance(self, game_state: GameState, available_actions: List[ActionType]) -> str:
        """村民决策指导"""
        return """## 村民策略指导
### 分析要点:
1. 观察每个玩家的发言逻辑
2. 注意投票模式和站队情况
3. 相信预言家的查验结果
4. 团结其他好人

### 投票策略:
1. 投给最可疑的玩家
2. 跟随预言家的指引
3. 避免被狼人带节奏"""

    def _get_hunter_guidance(self, game_state: GameState, available_actions: List[ActionType]) -> str:
        """猎人决策指导"""
        return """## 猎人策略指导
### 开枪时机:
1. 被狼人杀死时
2. 被误投出局时
3. 被女巫毒死时

### 开枪目标:
1. 确认的狼人
2. 最可疑的玩家
3. 避免误伤好人"""

    def _get_guard_guidance(self, game_state: GameState, available_actions: List[ActionType]) -> str:
        """守卫决策指导"""
        return """## 守卫策略指导
### 保护优先级:
1. 预言家（已跳身份）
2. 女巫（已知身份）
3. 重要的村民
4. 自己（关键时刻）

### 保护策略:
1. 不要连续保护同一人
2. 根据局势调整保护目标
3. 考虑狼人的击杀模式"""

    def _build_speech_prompt(self, game_state: GameState, speech_context: str) -> str:
        """构建发言提示词"""
        alive_players = game_state.get_alive_players()

        prompt = f"""## 发言情况
{speech_context}

## 当前状态
- 回合: 第{game_state.current_round}轮
- 存活玩家: {[f"{p.name}(ID:{p.id})" for p in alive_players]}
- 我是: {self.player.name} ({self.player.role.value})

## 我的记忆
{chr(10).join(self.memory[-3:]) if self.memory else "无记忆"}

请根据你的角色身份和当前局势进行发言。发言要求:
1. 符合角色身份特点
2. 不要直接透露角色（除非策略需要）
3. 发言长度控制在50-100字
4. 体现逻辑推理过程

以JSON格式返回:
{{
    \"speech\": \"你的发言内容\",
    \"reasoning\": \"发言策略解释\"
}}"""

        return prompt

    def _parse_response(self, response: str, available_actions: List[ActionType]) -> Optional[GameAction]:
        """解析AI响应"""
        try:
            # 提取JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                action_str = data.get('action', '').lower()
                target_id = data.get('target_id')

                # 转换行动类型
                action_type = None
                if action_str == 'kill':
                    action_type = ActionType.KILL
                elif action_str == 'check':
                    action_type = ActionType.CHECK
                elif action_str == 'save':
                    action_type = ActionType.SAVE
                elif action_str == 'poison':
                    action_type = ActionType.POISON
                elif action_str == 'guard':
                    action_type = ActionType.GUARD
                elif action_str == 'vote':
                    action_type = ActionType.VOTE
                elif action_str == 'pass':
                    return None

                if action_type and action_type in available_actions:
                    reasoning = data.get('reasoning', '')
                    print(f"  {self.player.name}: {reasoning[:100]}...")

                    return GameAction(
                        player_id=self.player.id,
                        action_type=action_type,
                        target_id=target_id
                    )

        except Exception as e:
            print(f"  {self.player.name}: 响应解析失败: {e}")

        return None

    def _parse_speech_response(self, response: str) -> str:
        """解析发言响应"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                speech = data.get('speech', '')
                reasoning = data.get('reasoning', '')

                print(f"  {self.player.name}: 发言策略: {reasoning[:50]}...")
                return speech

        except Exception as e:
            print(f"  {self.player.name}: 发言解析失败: {e}")

        return self._fallback_speech()

    def _fallback_action(self, available_actions: List[ActionType]) -> Optional[GameAction]:
        """备用行动策略"""
        if not available_actions:
            return None

        # 简单的随机策略
        import random
        action_type = random.choice(available_actions)

        return GameAction(
            player_id=self.player.id,
            action_type=action_type,
            target_id=None  # 需要在外部设置合理的目标
        )

    def _fallback_speech(self) -> str:
        """备用发言"""
        fallback_speeches = [
            "我觉得需要更多信息才能做判断。",
            "大家都说说自己的想法吧。",
            "我还在观察中，暂时没有明确的怀疑对象。"
        ]
        import random
        return random.choice(fallback_speeches)

    def _add_to_memory(self, content: str):
        """添加记忆"""
        self.memory.append(f"[第{time.time():.0f}s] {content}")
        # 保持记忆数量在合理范围内
        if len(self.memory) > 20:
            self.memory = self.memory[-15:]

    def _is_checked(self, player_id: int) -> bool:
        """检查是否已查验过某玩家（预言家用）"""
        for memory in self.memory:
            if f"查验{player_id}" in memory:
                return True
        return False


# 工厂函数
def create_ai_player(player: Player, llm_config: LLMConfig) -> WerewolfAIPlayer:
    """创建AI玩家"""
    return WerewolfAIPlayer(player, llm_config)


if __name__ == "__main__":
    # 测试AI玩家
    from .llm_manager import LLMConfig, APIProvider

    # 创建测试配置
    config = LLMConfig(
        name="test",
        provider=APIProvider.OPENAI,
        api_key="test-key",
        model="gpt-4o-mini"
    )

    # 创建测试玩家
    test_player = Player(id=1, name="TestAI", role=Role.WEREWOLF)
    ai_player = create_ai_player(test_player, config)

    print(f"创建了AI玩家: {ai_player.player.name} ({ai_player.player.role.value})")