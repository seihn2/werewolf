"""
对话管理系统
管理AI Bot的智能对话生成和上下文处理
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .game_logic import Role, Player
from .bot_memory import BotMemory, MemoryManager
from .game_logger import GameLogger
from .llm_manager import LLMAPIClient, LLMConfig


@dataclass
class SpeechRequest:
    """发言请求"""
    player: Player
    memory: BotMemory
    context: str
    speech_count: int
    max_speeches: int


class ConversationManager:
    """对话管理器"""

    def __init__(self, memory_manager: MemoryManager, logger: GameLogger):
        self.memory_manager = memory_manager
        self.logger = logger
        self.llm_configs: Dict[int, LLMConfig] = {}  # player_id -> LLMConfig

    def set_player_llm_config(self, player_id: int, llm_config: LLMConfig):
        """设置玩家的LLM配置"""
        self.llm_configs[player_id] = llm_config

    async def generate_speech(self, player: Player, round_number: int, phase: str) -> Optional[str]:
        """生成玩家发言"""
        memory = self.memory_manager.get_bot_memory(player.id)
        if not memory or not memory.can_speak_this_round():
            return None

        llm_config = self.llm_configs.get(player.id)
        if not llm_config:
            return self._generate_fallback_speech(player, memory)

        try:
            # 构建发言提示词
            prompt = self._build_speech_prompt(player, memory, round_number, phase)

            # 调用LLM生成发言
            async with LLMAPIClient(llm_config) as client:
                response = await client.chat_completion(prompt, self._build_system_prompt(player.role))

            # 解析响应
            speech = self._parse_speech_response(response)

            # 记录发言
            memory.record_speech()
            self.logger.log_speech(player.name, speech, memory.speech_count_this_round)

            # 广播给其他玩家
            self.memory_manager.broadcast_conversation(
                player.name, self._get_public_role(player), speech, phase, exclude_bot_id=player.id
            )

            return speech

        except Exception as e:
            print(f"生成发言失败 {player.name}: {e}")
            return self._generate_fallback_speech(player, memory)

    def _build_speech_prompt(self, player: Player, memory: BotMemory, round_number: int, phase: str) -> str:
        """构建发言提示词"""
        # 获取记忆上下文
        memory_context = memory.get_memory_context()

        # 获取最近游戏日志
        recent_log = self.logger.get_formatted_log(last_n_rounds=2)

        # 构建角色特定指导
        role_guidance = self._get_role_guidance(player.role, memory)

        # 构建发言要求
        speech_requirements = self._get_speech_requirements(memory.speech_count_this_round, phase)

        prompt = f"""## 当前游戏状况
{recent_log}

## 你的记忆和信息
{memory_context}

## 角色策略指导
{role_guidance}

## 发言要求
{speech_requirements}

## 高级发言指导
你需要展现出成熟玩家的水平，发言应该:

**逻辑层面:**
- 进行多层次的逻辑推理，不只是简单的"我觉得"
- 分析玩家的发言逻辑漏洞和前后矛盾
- 考虑概率和可能性，使用"如果...那么..."的推理
- 关注投票模式、站队情况、时机选择等细节

**表达技巧:**
- 使用反问、设问来引导思考
- 适当使用假设和推测来探测信息
- 运用对比和排除法来缩小怀疑范围
- 在关键时刻营造紧张感和说服力

**心理战术:**
- 观察并评论其他玩家的微表情和反应
- 适时制造信息压力或时间压力
- 运用信息不对称来获得优势
- 在必要时使用情感化的表达来增强说服力

**语言风格:**
- 避免过于机械化的表达
- 使用更自然的语言节奏和停顿
- 适当加入个人化的表达习惯
- **重要**: 发言长度必须在100-200字之间，需要详细阐述你的分析过程
- 多角度思考问题，展现思维的深度和广度

请以JSON格式回复:
{{
    "speech": "你的发言内容",
    "reasoning": "详细的发言策略、逻辑分析和心理考量",
    "confidence": 0.85,
    "target_analysis": "对主要怀疑目标的分析",
    "next_strategy": "下一步的行动计划"
}}"""

        return prompt

    def _build_system_prompt(self, role: Role) -> str:
        """构建系统提示词"""
        role_descriptions = {
            Role.VILLAGER: """你是村民，目标是通过逻辑推理找出所有狼人。作为好人阵营的核心力量，你需要:
            - 运用敏锐的观察力分析每个玩家的发言动机和逻辑漏洞
            - 建立和维护好人之间的信任关系，识别并排除伪装者
            - 在信息不充分时保持谨慎，避免被狼人的话术误导
            - 在关键投票时发挥决定性作用，承担起正义审判的责任""",

            Role.WEREWOLF: """你是狼人，目标是通过欺骗和伪装消灭好人阵营。作为黑暗势力的代表，你需要:
            - 精心构建完美的好人人设，让逻辑和情感都无懈可击
            - 巧妙地引导舆论方向，让好人自相残杀
            - 在保护同伴时要显得自然，避免暴露狼人之间的默契
            - 利用信息差和心理压力，在关键时刻扭转局势""",

            Role.SEER: """你是预言家，拥有夜晚查验的神圣能力。作为真相的守护者，你需要:
            - 策略性地选择查验目标，最大化信息价值
            - 在适当时机公布身份和结果，引导好人阵营
            - 识别假预言家的破绽，维护自己的权威性
            - 平衡信息公开和自身安全，避免过早成为狼人目标""",

            Role.WITCH: """你是女巫，掌握着生死的神秘力量。作为平衡者，你需要:
            - 精准判断何时使用解药和毒药，发挥最大效用
            - 在隐藏身份的同时默默守护好人阵营
            - 通过细致观察判断局势，在关键时刻一锤定音
            - 与其他神职配合，形成完整的信息网络""",

            Role.HUNTER: """你是猎人，拥有死亡反击的最后手段。作为正义的执行者，你需要:
            - 保持低调以避免成为优先击杀目标
            - 在生命受到威胁时准确识别最佳射击目标
            - 通过发言影响局势，但避免过早暴露身份
            - 在关键时刻承担起同归于尽的悲壮使命""",

            Role.GUARD: """你是守卫，负责保护重要人物的安全。作为暗中的守护者，你需要:
            - 精准预判狼人的击杀目标和心理
            - 在不暴露身份的前提下引导保护策略
            - 与其他角色建立隐性配合关系
            - 在关键夜晚做出可能改变游戏走向的保护决策"""
        }

        base_prompt = f"""你是一个具有高级策略思维的狼人杀专家级AI玩家。{role_descriptions.get(role, '')}

## 核心人格特质:
- **深度思考者**: 每个发言都经过多层逻辑验证，善于发现细微线索
- **心理学家**: 能够分析他人的动机、情感变化和行为模式
- **战略家**: 具备长远规划能力，每步行动都服务于整体战略
- **演技派**: 能够完美诠释角色，让表演自然而令人信服
- **哲学家**: 在道德和利益之间找到微妙平衡，体现角色的内心挣扎

## 高级原则:
1. **多维度推理**: 同时考虑逻辑、心理、概率、时机等多个维度
2. **信息层次化**: 区分确定信息、推测信息和诱导信息的价值
3. **动态策略调整**: 根据局势变化灵活调整战术和表达方式
4. **人性化表达**: 展现真实的情感波动、犹豫和坚定
5. **艺术化对抗**: 将游戏升华为智慧和演技的精彩较量

所有回复必须是完整有效的JSON格式，体现专业水准。"""

        return base_prompt

    def _get_role_guidance(self, role: Role, memory: BotMemory) -> str:
        """获取角色特定的策略指导"""
        if role == Role.WEREWOLF:
            return self._get_werewolf_guidance(memory)
        elif role == Role.SEER:
            return self._get_seer_guidance(memory)
        elif role == Role.WITCH:
            return self._get_witch_guidance(memory)
        elif role == Role.VILLAGER:
            return self._get_villager_guidance(memory)
        elif role == Role.HUNTER:
            return self._get_hunter_guidance(memory)
        elif role == Role.GUARD:
            return self._get_guard_guidance(memory)
        else:
            return "按照你的角色特点进行发言，帮助你的阵营获胜。"

    def _get_werewolf_guidance(self, memory: BotMemory) -> str:
        """狼人策略指导"""
        return """## 狼人高级伪装战术

**深层伪装技巧:**
- 展现对游戏胜负的真诚关切，偶尔表达对局势的焦虑和不安
- 主动分析其他玩家的可疑之处，但避开真正的狼人同伴
- 在投票时展现犹豫和纠结，体现"好人"的道德负担感
- 适度质疑权威（如预言家），但要有合理的逻辑支撑

**高级心理操控:**
- 利用人类的从众心理，巧妙引导群体倾向
- 在关键时刻制造时间压力，迫使好人匆忙决策
- 通过"假设分析"和"换位思考"来混淆逻辑链条
- 偶尔"无意中"透露对某个好人的"担心"，制造内部猜疑

**同伴保护艺术:**
- 不直接为同伴辩护，而是质疑指控者的动机
- 通过转移话题或提出其他怀疑目标来分散注意力
- 在同伴被怀疑时表现出"理性分析"而非情感化保护
- 必要时可以"轻微质疑"同伴来增加可信度

**信息战略运用:**
- 混合真假信息，让逻辑链条似是而非
- 利用好人的善良天性，制造道德绑架
- 在投票前制造"重大发现"的假象来扭转局势"""

    def _get_seer_guidance(self, memory: BotMemory) -> str:
        """预言家策略指导"""
        verifications = memory.get_verification_history()
        if verifications:
            recent_results = verifications[-2:]  # 最近2次查验
            results_text = "、".join([f"{v.target_player_name}({v.result})" for v in recent_results])
            guidance = f"你的神圣查验记录: {results_text}。这些真相将指引正义的道路。"
        else:
            guidance = "你尚未使用神圣的查验能力，需要谨慎选择第一个查验目标。"

        return f"""## 预言家权威建立策略
{guidance}

**真相传播艺术:**
- 不要急于在第一轮就暴露身份，先观察局势和玩家表现
- 当公布查验结果时，要配合详细的逻辑分析增强可信度
- 用"昨晚的神启"、"真相的指引"等带有神秘色彩的表达
- 在公布狼人身份时展现出正义的愤怒和使命感

**权威维护技巧:**
- 面对假预言家时，用逻辑和细节揭露其破绽
- 强调自己信息的准确性和一致性
- 通过预测其他玩家的反应来证明自己的洞察力
- 在关键时刻展现出为真相而战的坚定决心

**团队协调智慧:**
- 暗示其他神职角色与自己配合，但不要直接点名
- 引导好人阵营形成以自己为核心的信任网络
- 在投票时发挥精神领袖的作用，承担决策责任
- 必要时可以"牺牲"自己来换取关键信息的传递

**生存平衡策略:**
- 在保持影响力和避免成为击杀目标之间找到平衡
- 通过"假设自己被杀"的话术来提醒保护的重要性
- 在局势不利时可以适当示弱，等待翻盘机会"""

    def _get_witch_guidance(self, memory: BotMemory) -> str:
        """女巫策略指导"""
        return """## 女巫神秘力量运用指南

**隐秘观察艺术:**
- 在发言中展现出对细节的敏锐观察，暗示你有"特殊的感知能力"
- 通过分析死亡情况和玩家反应来展示你的"直觉"
- 偶尔表达对某个玩家的"不安预感"，为后续毒杀做铺垫
- 在讨论保护策略时表现出"莫名的自信"

**生死平衡哲学:**
- 在讨论救人话题时表现出深层的道德思考
- 对于"牺牲与拯救"展现出复杂的情感纠葛
- 在关键时刻暗示"命运的天平正在倾斜"
- 用哲学化的语言表达对正义与复仇的思考

**战略时机把控:**
- 在第一夜通常救人，但要在发言中巧妙暗示这一点
- 观察预言家的可信度，决定是否配合其策略
- 毒药的使用要配合精彩的"正义审判"演说
- 在最后关头可以暴露身份来增强说服力

**神秘威慑建立:**
- 让其他玩家感受到你的"神秘力量"带来的压迫感
- 在投票时表现出"我知道真相"的自信
- 通过预言式的表达来营造神秘感"""

    def _get_villager_guidance(self, memory: BotMemory) -> str:
        """村民策略指导"""
        return """## 村民理性分析大师指南

**深度逻辑分析:**
- 建立每个玩家的"行为档案"，追踪其逻辑一致性
- 分析发言的"动机层次": 表面动机vs深层动机vs隐藏动机
- 运用"反证法": 假设某人是狼人，其行为是否符合逻辑
- 构建"信任链条": 基于逻辑推理建立可信度排序

**心理行为观察:**
- 注意玩家在压力下的反应变化和言语破绽
- 观察谁在关键时刻表现出"过度紧张"或"异常冷静"
- 分析投票模式背后的隐藏关系和利益考量
- 识别"从众行为"和"引导行为"的区别

**团队协调智慧:**
- 作为"理性声音"在混乱中提供清晰的分析框架
- 在好人内部分歧时发挥"调解者"和"智者"的作用
- 用事实和逻辑来支持神职角色的权威
- 在关键投票时承担起"正义审判者"的责任

**战略思维运用:**
- 不仅看当前轮次，还要预测后续几轮的发展趋势
- 考虑"容错空间": 好人阵营还能承受几次错误
- 在信息不足时保持谨慎，避免被情绪化论调带偏
- 在确定性高的时候要坚定地推动正确决策

**表达艺术提升:**
- 用"数据分析师"的严谨态度来表达观点
- 在指控时要有充分的证据链支撑
- 展现出对正义的执着追求和对错误的深度反思"""

    def _get_hunter_guidance(self, memory: BotMemory) -> str:
        """猎人策略指导"""
        return """## 猎人高级战术大师指南

**隐秘威慑艺术:**
- 采用"沉默观察者"人设，在关键时刻展现出超越常人的判断力
- 通过微妙的暗示让场上感受到"未知力量"的存在
- 在发言中偶尔透露出对生死抉择的深度思考
- 展现出"一击必杀"的气质，让狼人对隐藏的威胁心存忌惮

**战略分析框架:**
- 构建"多维威胁评估模型": 分析每个玩家的危险程度
- 运用"博弈论最优解": 你的存在改变了整个游戏的均衡点
- 考虑"蝴蝶效应原理": 一枪的选择可能决定整局走向
- 实施"信息价值最大化": 在掌握最多信息时才做最终决断

**心理战术运用:**
- 观察谁在试探或暗示神职身份，这些人可能是狼人
- 分析场上的"恐惧分布": 谁最害怕猎人的存在
- 识别"诱导性发言": 狼人可能试图引导你的枪口方向
- 注意"异常保护行为": 过度为某人辩护的玩家值得怀疑

**开枪哲学体系:**
- **正义审判原则**: 开枪不仅是游戏行为，更是正义的执行
- **概率最优化**: 在信息充分且胜率最高时出手
- **责任承担理念**: 为好人阵营承担最后的审判责任
- **战略时机把控**: 选择对己方最有利的时机发挥作用

**高级表达模式:**
"从战略平衡的角度分析..."
"这种行为模式在博弈论中通常意味着..."
"作为一个观察者，我注意到了一些微妙的变化..."
"有时候，沉默比言语更有力量..."
"在关键时刻，每个选择都承载着巨大的责任..."
"""

    def _get_guard_guidance(self, memory: BotMemory) -> str:
        """守卫策略指导"""
        return """## 守卫暗影保护大师指南

**隐形守护哲学:**
- 成为"看不见的守护天使"，在幕后操控生死平衡
- 通过细微的言行暗示你对某些玩家的"特殊关注"
- 在讨论保护策略时展现出超乎寻常的敏感度
- 偶尔表达对"意外死亡"和"幸运存活"的深度思考

**威胁评估系统:**
- 建立"死亡概率预测模型": 分析狼人的击杀逻辑
- 运用"价值保护理论": 优先保护对好人阵营最有价值的玩家
- 考虑"心理博弈层次": 预测狼人的预期和反预期
- 实施"动态调整策略": 根据场上变化调整守护重点

**守护艺术精髓:**
- **预判狼人心理**: 站在狼人角度思考今晚最想杀谁
- **神职协同作战**: 与预言家、女巫形成完美的保护网
- **时机选择智慧**: 在关键轮次发挥决定性作用
- **信息隐藏技巧**: 让守护行为看起来像是"运气"或"巧合"

**战略层次思考:**
- 分析"连环保护效应": 你的守护如何影响整体局势
- 考虑"心理威慑价值": 狼人知道有守卫会改变策略
- 运用"不确定性武器": 让狼人无法准确预测你的行为
- 在必要时可以"战略性暴露"来保护更重要的目标

**神秘感营造:**
- 在有人幸存时表现出"并不意外"的神态
- 对死亡事件展现出深层的分析和预见性
- 在关键时刻暗示"有些事情并非偶然"
- 用"命运守护者"的气质来增强说服力

**高级表达模式:**
"从风险管理的角度来看..."
"这种攻击模式背后的逻辑值得深思..."
"有时候，最好的保护就是让对方无法预测..."
"在黑暗中，总有人在默默守护着正义..."
"真正的守护不仅是阻止死亡，更是维护希望..."
"""

    def _get_speech_requirements(self, current_speech_count: int, phase: str) -> str:
        """获取发言要求"""
        if current_speech_count == 0:
            return f"""这是你在第{phase}阶段的第1次发言(必须100-200字):
**必须包含以下内容:**
- 详细分析当前局势和已知信息
- 阐述你的整体观察和逻辑推理过程
- 明确表达你的怀疑对象并说明理由
- 分享你认为重要的信息和线索
- 表达你的立场和后续行动计划

**发言要求:**
- 必须使用完整的句子和段落
- 体现深度思考和多维度分析
- 展现角色的个性和智慧
- 长度严格控制在100-200字之间"""
        else:
            return f"""这是你在第{phase}阶段的第2次发言(必须100-200字):
**必须包含以下内容:**
- 详细回应其他玩家的关键发言
- 分析他人发言中的逻辑漏洞或可疑之处
- 补充或修正你之前的观点和分析
- 强调你的核心判断和证据链
- 提出新的发现或策略调整

**发言要求:**
- 针对性回应至少2-3个其他玩家的观点
- 展现更深层次的战略思考
- 保持角色一致性和逻辑连贯性
- 长度严格控制在100-200字之间"""

    def _get_public_role(self, player: Player) -> str:
        """获取公开角色（可能是伪装的）"""
        # 在实际游戏中，玩家可能会声称不同的身份
        # 这里简化处理，返回真实角色
        return player.role.value

    def _parse_speech_response(self, response: str) -> str:
        """解析LLM发言响应"""
        try:
            # 提取JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                speech = data.get('speech', '')
                if speech:
                    # 检查长度，如果太短则要求更详细
                    if len(speech) < 80:
                        return f"{speech} 基于目前的情况分析，我认为这个判断是合理的，因为从逻辑推理的角度来看，各种线索都指向这个结论。我们需要更仔细地观察每个人的行为模式和发言逻辑。"
                    elif len(speech) > 300:
                        return speech[:280] + "..."
                    return speech

        except Exception as e:
            print(f"解析发言响应失败: {e}")

        # 如果解析失败，尝试直接提取文本
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('{') and not line.startswith('}'):
                # 确保有足够长度
                if len(line) < 80:
                    return f"{line} 从当前的局势来看，我需要更仔细地分析每个玩家的行为模式。基于已有的信息和逻辑推理，我认为我们应该重点关注那些发言中存在矛盾或可疑之处的玩家。"
                return line[:280]

        return "我需要更多时间来分析当前的复杂局势。从目前掌握的信息来看，每个玩家的发言都值得深入研究。我会仔细观察大家的行为模式和逻辑推理过程，以便做出更准确的判断。"

    def _generate_fallback_speech(self, player: Player, memory: BotMemory) -> str:
        """生成备用发言"""
        fallback_speeches = {
            Role.WEREWOLF: [
                "我觉得需要更仔细地分析大家的发言，从目前的情况来看，有些人的逻辑确实存在问题。我们作为好人阵营必须保持冷静和理性，不能被情绪化的发言所影响。我建议大家仔细回顾一下每个人的发言记录，看看是否存在前后矛盾的地方。",
                "从我的观察来看，某些人的动机确实不太纯粹，他们的发言中透露出一些可疑的信号。我们需要团结起来找出真正的狼人，而不是被他们的伪装所迷惑。我认为应该重点关注那些试图转移注意力或者制造混乱的玩家。",
                "我们要团结起来找出真正的狼人，这需要我们运用逻辑推理和细致观察。从目前的发言来看，我注意到有些玩家在关键问题上总是回避或者转移话题，这种行为模式值得我们深入分析和思考。"
            ],
            Role.SEER: [
                "作为一个仔细观察局势的玩家，我有一些重要的信息要与大家分享。从我的角度来看，目前的局势比较复杂，需要我们综合各种线索来做出判断。我建议大家不要急于下结论，而是要基于事实和逻辑进行分析。",
                "根据我对整个游戏进程的观察，情况确实比较复杂，但并非没有线索可循。我注意到某些玩家的行为模式存在一定的规律性，这可能是我们破解谜题的关键。我建议大家仔细分析每个人的发言逻辑和行为动机。",
                "从当前的局势分析来看，我建议大家仔细考虑投票对象的选择。我们不能仅仅基于直觉或情感来做决定，而是要基于充分的证据和逻辑推理。每一票都关系到游戏的走向，我们必须慎重对待。"
            ],
            Role.VILLAGER: [
                "作为一个普通村民，我深知找出狼人的重要性和紧迫性。从目前的发言情况来看，我们需要更多的线索和证据来支撑我们的判断。我建议大家保持理性，不要被情绪化的言论所影响，要基于事实进行分析。",
                "我觉得我们确实需要更多的线索来帮助我们做出正确的判断。从逻辑分析的角度来看，每个人的发言都包含着重要的信息，我们需要仔细挖掘和分析这些信息。我支持那些逻辑清晰、证据充分的分析和推理。",
                "我一直支持逻辑清晰的分析方法，因为只有这样我们才能在复杂的局势中找到真相。从目前的情况来看，我们需要更加仔细地观察每个人的行为模式和发言特点，寻找其中的矛盾和破绽。"
            ],
            Role.WITCH: [
                "我一直在仔细观察大家的表现，从中寻找有用的线索和信息。作为一个善于观察的玩家，我注意到某些细节可能对我们的判断有重要意义。我会继续保持警觉，在关键时刻为好人阵营提供帮助。",
                "在这个关键时刻，我相信每个有能力的玩家都会发挥自己的作用。从目前的局势来看，我们需要更加团结和协调，才能有效地对抗狼人阵营。我会根据情况的发展做出相应的决策。",
                "我们确实需要更好地保护好人阵营的利益，这需要我们每个人都贡献自己的智慧和力量。从战略的角度来看，我们应该重点关注那些对好人阵营威胁最大的目标，并采取相应的措施。"
            ],
            Role.HUNTER: [
                "我会继续仔细观察每个人的表现，并在适当的时候做出正确的判断。从目前的情况来看，局势确实比较复杂，需要我们保持冷静和理性。我不会轻易做出决定，但一旦确定目标，我会毫不犹豫地行动。",
                "在这个游戏中，我们确实不能轻易相信任何人，每个人都可能有自己的隐藏动机。我会基于客观的观察和分析来做出判断，而不是被表面的现象所迷惑。关键时刻我会为正义而战。",
                "我相信在关键时刻，每个有责任感的玩家都会站出来为正义发声。从目前的发言情况来看，我需要更多的时间来观察和分析，以便做出最准确的判断。我的行动将基于充分的证据和理由。"
            ],
            Role.GUARD: [
                "从保护的角度来看，我觉得某些玩家确实需要我们的特别关注和保护。狼人的攻击目标往往是那些对他们威胁最大的玩家，我们需要提前识别这些潜在的目标，并采取相应的保护措施。",
                "狼人的攻击目标和策略确实有一定的规律可循，通过仔细分析他们的行为模式，我们可以在一定程度上预判他们的下一步行动。我会根据局势的发展调整自己的保护策略，确保关键玩家的安全。",
                "我们确实需要更好地预判狼人的行动和意图，这需要我们综合各种信息进行分析。从防守的角度来看，我会重点关注那些最有可能成为狼人攻击目标的玩家，并在必要时提供保护。"
            ]
        }

        import random
        speeches = fallback_speeches.get(player.role, [
            "我需要更多时间来分析当前的复杂局势。从目前掌握的信息来看，每个玩家的发言都值得深入研究。我会仔细观察大家的行为模式和逻辑推理过程，以便做出更准确的判断。在这个关键时刻，我们需要保持冷静和理性。"
        ])
        speech = random.choice(speeches)

        # 记录发言
        memory.record_speech()
        return speech

    async def generate_vote(self, player: Player, candidates: List[Player]) -> Optional[str]:
        """生成玩家投票"""
        memory = self.memory_manager.get_bot_memory(player.id)
        if not memory:
            return None

        llm_config = self.llm_configs.get(player.id)
        if not llm_config:
            return self._generate_fallback_vote(player, candidates)

        try:
            # 构建投票提示词
            prompt = self._build_vote_prompt(player, memory, candidates)

            # 调用LLM生成投票决策
            async with LLMAPIClient(llm_config) as client:
                response = await client.chat_completion(prompt, self._build_system_prompt(player.role))

            # 解析响应
            vote_target = self._parse_vote_response(response, candidates)
            return vote_target

        except Exception as e:
            print(f"生成投票失败 {player.name}: {e}")
            return self._generate_fallback_vote(player, candidates)

    def _build_vote_prompt(self, player: Player, memory: BotMemory, candidates: List[Player]) -> str:
        """构建投票提示词"""
        # 获取记忆上下文
        memory_context = memory.get_memory_context()

        # 获取最近游戏日志
        recent_log = self.logger.get_formatted_log(last_n_rounds=1)

        # 候选人列表
        candidate_list = [f"{p.name}(ID:{p.id})" for p in candidates]

        prompt = f"""## 投票决策时刻

**当前游戏状况:**
{recent_log}

**你的记忆和信息:**
{memory_context}

**投票候选人:**
{', '.join(candidate_list)}

## 投票决策要求

你需要基于以下因素做出投票决策:

**分析要素:**
1. **逻辑一致性**: 分析每个候选人发言的逻辑是否一致
2. **行为模式**: 观察可疑的行为和反应
3. **阵营判断**: 根据角色特点判断谁最可能是狼人
4. **威胁程度**: 评估谁对你的阵营威胁最大
5. **信息价值**: 考虑淘汰谁能获得最多信息

**决策过程:**
- 详细分析每个候选人的可疑程度(1-10分)
- 说明你的投票理由和逻辑链条
- 考虑这次投票对后续局势的影响

请以JSON格式回复:
{{
    "analysis": "详细的候选人分析过程",
    "vote_target": "被投票人的姓名(必须完全匹配候选人列表中的姓名)",
    "reasoning": "投票理由和战略考虑",
    "confidence": 0.85
}}

**重要**: vote_target必须是候选人列表中的确切姓名，不能是其他内容。"""

        return prompt

    def _parse_vote_response(self, response: str, candidates: List[Player]) -> Optional[str]:
        """解析投票响应"""
        candidate_names = [p.name for p in candidates]

        try:
            # 提取JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                vote_target = data.get('vote_target', '')
                if vote_target in candidate_names:
                    return vote_target

        except Exception as e:
            print(f"解析投票响应失败: {e}")

        # 如果解析失败，尝试从文本中提取
        for name in candidate_names:
            if name in response:
                return name

        return None

    def _generate_fallback_vote(self, player: Player, candidates: List[Player]) -> Optional[str]:
        """生成备用投票"""
        if not candidates:
            return None

        import random
        return random.choice(candidates).name


if __name__ == "__main__":
    # 测试对话管理器
    from .bot_memory import MemoryManager
    from .game_logger import GameLogger
    from .llm_manager import LLMConfig, APIProvider

    # 创建测试环境
    memory_manager = MemoryManager("test_game")
    logger = GameLogger("test_game")
    conversation_manager = ConversationManager(memory_manager, logger)

    # 创建测试玩家和记忆
    test_player = Player(1, "测试玩家", Role.SEER)
    memory = memory_manager.create_bot_memory(test_player.id, test_player.name, test_player.role)

    # 添加一些测试记忆
    memory.start_new_round(1)
    memory.add_verification_result(2, "玩家2", "bad")
    memory.add_conversation("玩家3", "村民", "我觉得玩家2很可疑")

    # 生成发言提示词
    prompt = conversation_manager._build_speech_prompt(test_player, memory, 1, "day")
    print("发言提示词:")
    print(prompt)