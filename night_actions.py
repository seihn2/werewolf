"""
夜晚神职技能AI决策模块
让预言家、女巫等神职角色使用AI来决定技能使用
"""

import asyncio
import random
from typing import List, Dict, Any, Optional
from .game_logic import Role, Player
from .bot_memory import BotMemory, MemoryManager
from .game_logger import GameLogger
from .llm_manager import LLMAPIClient, LLMConfig


class NightActionManager:
    """夜晚行动管理器"""

    def __init__(self, memory_manager: MemoryManager, logger: GameLogger, conversation_manager=None):
        self.memory_manager = memory_manager
        self.logger = logger
        self.conversation_manager = conversation_manager
        self.llm_configs: Dict[int, LLMConfig] = {}

    def set_player_llm_config(self, player_id: int, llm_config: LLMConfig):
        """设置玩家LLM配置"""
        self.llm_configs[player_id] = llm_config

    async def seer_check_decision(self, seer: Player, candidates: List[Player]) -> Optional[Player]:
        """预言家选择查验目标"""
        if not candidates:
            return None

        memory = self.memory_manager.get_bot_memory(seer.id)
        if not memory:
            return random.choice(candidates)

        llm_config = self.llm_configs.get(seer.id)
        if not llm_config:
            return self._fallback_seer_choice(candidates)

        try:
            # 添加超时处理
            target = await asyncio.wait_for(
                self._ai_seer_decision(seer, memory, candidates, llm_config),
                timeout=20.0
            )
            return target if target else self._fallback_seer_choice(candidates)

        except asyncio.TimeoutError:
            print(f"{seer.name} AI决策超时，使用随机行动")
            return self._fallback_seer_choice(candidates)
        except Exception as e:
            print(f"{seer.name} AI决策失败: {e}")
            return self._fallback_seer_choice(candidates)

    async def witch_action_decision(self, witch: Player, victim: Optional[Player], available_actions: Dict[str, Any]) -> Dict[str, Any]:
        """女巫选择行动"""
        memory = self.memory_manager.get_bot_memory(witch.id)
        if not memory:
            return self._fallback_witch_action(available_actions)

        llm_config = self.llm_configs.get(witch.id)
        if not llm_config:
            return self._fallback_witch_action(available_actions)

        try:
            # 添加超时处理
            action = await asyncio.wait_for(
                self._ai_witch_decision(witch, memory, victim, available_actions, llm_config),
                timeout=20.0
            )
            return action if action else self._fallback_witch_action(available_actions)

        except asyncio.TimeoutError:
            print(f"{witch.name} AI决策超时，使用默认行动")
            return self._fallback_witch_action(available_actions)
        except Exception as e:
            print(f"{witch.name} AI决策失败: {e}")
            return self._fallback_witch_action(available_actions)

    async def _ai_seer_decision(self, seer: Player, memory: BotMemory, candidates: List[Player], llm_config: LLMConfig) -> Optional[Player]:
        """AI预言家决策"""
        prompt = self._build_seer_prompt(seer, memory, candidates)

        async with LLMAPIClient(llm_config) as client:
            response = await client.chat_completion(prompt, self._build_seer_system_prompt())

        # 解析响应
        return self._parse_seer_response(response, candidates)

    async def _ai_witch_decision(self, witch: Player, memory: BotMemory, victim: Optional[Player], available_actions: Dict[str, Any], llm_config: LLMConfig) -> Optional[Dict[str, Any]]:
        """AI女巫决策"""
        prompt = self._build_witch_prompt(witch, memory, victim, available_actions)

        async with LLMAPIClient(llm_config) as client:
            response = await client.chat_completion(prompt, self._build_witch_system_prompt())

        # 解析响应
        return self._parse_witch_response(response, available_actions)

    def _build_seer_prompt(self, seer: Player, memory: BotMemory, candidates: List[Player]) -> str:
        """构建预言家查验提示词"""
        memory_context = memory.get_memory_context()
        recent_log = self.logger.get_formatted_log(last_n_rounds=1)

        candidate_list = []
        for candidate in candidates:
            candidate_list.append(f"- {candidate.name}(ID:{candidate.id})")

        prompt = f"""## 预言家夜晚查验决策时刻

**当前游戏状况:**
{recent_log}

**你的记忆和经验:**
{memory_context}

**可查验的玩家:**
{chr(10).join(candidate_list)}

## 战略分析要求

作为预言家，你需要选择最有价值的查验目标。请基于以下维度分析：

1. **威胁评估**: 谁的发言最可疑，最可能是狼人？
2. **信息价值**: 查验谁能获得最有用的信息？
3. **战略考虑**: 避免查验已知身份或无价值的目标
4. **时机判断**: 现在查验谁对局势帮助最大？

**请直接回答你要查验的玩家姓名，不需要额外解释。**

示例回答格式: 玩家2"""

        return prompt

    def _build_witch_prompt(self, witch: Player, memory: BotMemory, victim: Optional[Player], available_actions: Dict[str, Any]) -> str:
        """构建女巫行动提示词"""
        memory_context = memory.get_memory_context()
        recent_log = self.logger.get_formatted_log(last_n_rounds=1)

        victim_info = f"{victim.name} 被狼人杀害" if victim else "昨晚平安夜"

        actions_desc = []
        if available_actions.get('can_save', False):
            actions_desc.append("- save: 使用解药救活被杀玩家")
        if available_actions.get('can_poison', False):
            actions_desc.append("- poison [玩家名]: 使用毒药毒死指定玩家")
        actions_desc.append("- pass: 不使用任何技能")

        prompt = f"""## 女巫夜晚行动决策时刻

**当前游戏状况:**
{recent_log}

**你的记忆和经验:**
{memory_context}

**昨晚死亡情况:**
{victim_info}

**可用技能:**
{chr(10).join(actions_desc)}

## 战略决策要求

作为女巫，你掌握着生死的力量。请基于以下维度分析：

1. **生命价值**: 被杀玩家对好人阵营的价值如何？
2. **技能保存**: 现在使用技能是否值得？
3. **身份隐藏**: 行动是否会暴露自己身份？
4. **局势判断**: 当前好坏人比例，是否需要立即行动？

**请直接回答你的行动选择:**

格式示例:
- save (救人)
- poison 玩家名 (下毒)
- pass (不行动)"""

        return prompt

    def _build_seer_system_prompt(self) -> str:
        """预言家系统提示词"""
        return """你是一个经验丰富的狼人杀预言家。你拥有每晚查验一名玩家身份的神圣能力。

作为真相的守护者，你需要：
- 通过查验获取关键信息
- 选择最有价值的查验目标
- 在适当时机公布查验结果
- 引导好人阵营走向胜利

你的决策应该基于逻辑分析和战略考量，选择能获得最大信息价值的目标进行查验。"""

    def _build_witch_system_prompt(self) -> str:
        """女巫系统提示词"""
        return """你是一个智慧深邃的狼人杀女巫。你拥有解药和毒药两项强大的技能。

作为生死的裁决者，你需要：
- 精准判断生命的价值
- 在关键时刻发挥决定性作用
- 平衡技能使用的时机
- 在隐藏身份的同时守护好人阵营

你的每一个决策都可能改变游戏的走向，请慎重考虑每一次技能的使用。"""

    def _parse_seer_response(self, response: str, candidates: List[Player]) -> Optional[Player]:
        """解析预言家查验响应"""
        if not response:
            return None

        response = response.strip()

        # 尝试匹配候选人名字
        for candidate in candidates:
            if candidate.name in response:
                return candidate

        return None

    def _parse_witch_response(self, response: str, available_actions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解析女巫行动响应"""
        if not response:
            return None

        response = response.strip().lower()

        if 'save' in response and available_actions.get('can_save', False):
            return {'action': 'save'}
        elif 'poison' in response and available_actions.get('can_poison', False):
            # 尝试提取目标玩家名
            import re
            match = re.search(r'poison\s+([^\s]+)', response)
            if match:
                target_name = match.group(1)
                return {'action': 'poison', 'target': target_name}
            return {'action': 'poison', 'target': None}
        else:
            return {'action': 'pass'}

    def _fallback_seer_choice(self, candidates: List[Player]) -> Player:
        """备用预言家查验选择"""
        return random.choice(candidates)

    def _fallback_witch_action(self, available_actions: Dict[str, Any]) -> Dict[str, Any]:
        """备用女巫行动选择"""
        # 简单策略：有解药倾向于救人，否则不行动
        if available_actions.get('can_save', False) and random.random() < 0.6:
            return {'action': 'save'}
        else:
            return {'action': 'pass'}