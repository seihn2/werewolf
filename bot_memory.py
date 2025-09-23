"""
Bot记忆管理系统
为每个Bot提供独立的记忆存储和管理功能
"""

import json
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from .game_logic import Role


@dataclass
class VerificationResult:
    """查验结果"""
    target_player_id: int
    target_player_name: str
    result: str  # "good" or "bad"
    round_number: int
    timestamp: float


@dataclass
class ConversationEntry:
    """对话记录条目"""
    round_number: int
    phase: str  # "day", "night", "voting"
    speaker_name: str
    speaker_role: str  # 公开的角色（可能是伪装的）
    content: str
    timestamp: float
    is_private: bool = False  # 是否是私人信息


@dataclass
class RoundSummary:
    """轮次总结"""
    round_number: int
    deaths: List[str]  # 死亡玩家列表
    votes: Dict[str, str]  # 投票结果 {voter: target}
    eliminated: Optional[str]  # 被淘汰的玩家
    key_events: List[str]  # 关键事件


class BotMemory:
    """Bot记忆系统"""

    def __init__(self, bot_id: int, name: str, role: Role, game_id: str):
        self.bot_id = bot_id
        self.name = name
        self.role = role
        self.game_id = game_id
        self.current_round = 0
        self.speech_count_this_round = 0
        self.max_speeches_per_round = 2

        # 记忆内容
        self.verification_results: List[VerificationResult] = []
        self.conversation_history: List[ConversationEntry] = []
        self.round_summaries: List[RoundSummary] = []
        self.suspicions: Dict[str, float] = {}  # {player_name: suspicion_level}
        self.private_notes: List[str] = []  # 私人笔记

        # 存储路径
        self.memory_dir = f"game_memories/game_{game_id}"
        self.memory_file = f"{self.memory_dir}/bot_{bot_id}_{name}.json"

        # 确保目录存在
        os.makedirs(self.memory_dir, exist_ok=True)

        # 初始化记忆
        self._initialize_memory()

    def _initialize_memory(self):
        """初始化记忆"""
        initial_note = f"我是{self.name}，真实身份是{self.role.value}。新游戏开始。"
        self.add_private_note(initial_note)
        self.save_memory()

    def add_verification_result(self, target_id: int, target_name: str, result: str):
        """添加查验结果（预言家专用）"""
        if self.role != Role.SEER:
            return

        verification = VerificationResult(
            target_player_id=target_id,
            target_player_name=target_name,
            result=result,
            round_number=self.current_round,
            timestamp=time.time()
        )
        self.verification_results.append(verification)

        # 添加到私人笔记
        note = f"第{self.current_round}轮：我查验了{target_name}，结果是{result}"
        self.add_private_note(note)

        self.save_memory()

    def add_conversation(self, speaker_name: str, speaker_role: str, content: str,
                        phase: str = "day", is_private: bool = False):
        """添加对话记录"""
        entry = ConversationEntry(
            round_number=self.current_round,
            phase=phase,
            speaker_name=speaker_name,
            speaker_role=speaker_role,
            content=content,
            timestamp=time.time(),
            is_private=is_private
        )
        self.conversation_history.append(entry)
        self.save_memory()

    def add_round_summary(self, deaths: List[str], votes: Dict[str, str],
                         eliminated: Optional[str], key_events: List[str]):
        """添加轮次总结"""
        summary = RoundSummary(
            round_number=self.current_round,
            deaths=deaths,
            votes=votes,
            eliminated=eliminated,
            key_events=key_events
        )
        self.round_summaries.append(summary)
        self.save_memory()

    def update_suspicion(self, player_name: str, suspicion_level: float):
        """更新对某玩家的怀疑度 (0.0-1.0)"""
        self.suspicions[player_name] = max(0.0, min(1.0, suspicion_level))
        self.save_memory()

    def add_private_note(self, note: str):
        """添加私人笔记"""
        timestamped_note = f"[第{self.current_round}轮] {note}"
        self.private_notes.append(timestamped_note)
        self.save_memory()

    def can_speak_this_round(self) -> bool:
        """检查本轮是否还能发言"""
        return self.speech_count_this_round < self.max_speeches_per_round

    def record_speech(self):
        """记录一次发言"""
        self.speech_count_this_round += 1
        self.save_memory()

    def start_new_round(self, round_number: int):
        """开始新轮次"""
        self.current_round = round_number
        self.speech_count_this_round = 0
        self.add_private_note(f"第{round_number}轮开始")
        self.save_memory()

    def get_recent_conversations(self, last_n_rounds: int = 2) -> List[ConversationEntry]:
        """获取最近N轮的对话"""
        min_round = max(1, self.current_round - last_n_rounds + 1)
        return [conv for conv in self.conversation_history
                if conv.round_number >= min_round]

    def get_verification_history(self) -> List[VerificationResult]:
        """获取查验历史（预言家专用）"""
        return self.verification_results.copy()

    def get_memory_context(self) -> str:
        """生成记忆上下文用于LLM"""
        context_parts = []

        # 基本身份信息
        context_parts.append(f"你是{self.name}，真实身份：{self.role.value}")
        context_parts.append(f"当前第{self.current_round}轮，本轮已发言{self.speech_count_this_round}次")

        # 查验结果（预言家）
        if self.verification_results:
            context_parts.append("\n你的查验记录：")
            for result in self.verification_results:
                context_parts.append(f"- 第{result.round_number}轮：{result.target_player_name} -> {result.result}")

        # 最近对话
        recent_convs = self.get_recent_conversations(2)
        if recent_convs:
            context_parts.append("\n最近对话：")
            for conv in recent_convs[-10:]:  # 最近10条
                phase_str = f"[第{conv.round_number}轮-{conv.phase}]"
                context_parts.append(f"{phase_str} {conv.speaker_name}：{conv.content}")

        # 怀疑度
        if self.suspicions:
            context_parts.append("\n当前怀疑度：")
            sorted_suspicions = sorted(self.suspicions.items(), key=lambda x: x[1], reverse=True)
            for name, level in sorted_suspicions[:5]:  # 前5个最可疑的
                context_parts.append(f"- {name}: {level:.2f}")

        # 私人笔记（最近几条）
        if self.private_notes:
            context_parts.append("\n重要笔记：")
            for note in self.private_notes[-5:]:  # 最近5条笔记
                context_parts.append(f"- {note}")

        return "\n".join(context_parts)

    def save_memory(self):
        """保存记忆到文件"""
        try:
            memory_data = {
                "bot_id": self.bot_id,
                "name": self.name,
                "role": self.role.value,
                "game_id": self.game_id,
                "current_round": self.current_round,
                "speech_count_this_round": self.speech_count_this_round,
                "verification_results": [asdict(vr) for vr in self.verification_results],
                "conversation_history": [asdict(conv) for conv in self.conversation_history],
                "round_summaries": [asdict(rs) for rs in self.round_summaries],
                "suspicions": self.suspicions,
                "private_notes": self.private_notes,
                "last_updated": time.time()
            }

            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"保存记忆失败 {self.name}: {e}")

    def load_memory(self) -> bool:
        """从文件加载记忆"""
        try:
            if not os.path.exists(self.memory_file):
                return False

            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)

            self.current_round = memory_data.get("current_round", 0)
            self.speech_count_this_round = memory_data.get("speech_count_this_round", 0)
            self.suspicions = memory_data.get("suspicions", {})
            self.private_notes = memory_data.get("private_notes", [])

            # 重建对象列表
            self.verification_results = [
                VerificationResult(**vr) for vr in memory_data.get("verification_results", [])
            ]
            self.conversation_history = [
                ConversationEntry(**conv) for conv in memory_data.get("conversation_history", [])
            ]
            self.round_summaries = [
                RoundSummary(**rs) for rs in memory_data.get("round_summaries", [])
            ]

            return True

        except Exception as e:
            print(f"加载记忆失败 {self.name}: {e}")
            return False


class MemoryManager:
    """记忆管理器"""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.bot_memories: Dict[int, BotMemory] = {}

    def create_bot_memory(self, bot_id: int, name: str, role: Role) -> BotMemory:
        """为Bot创建记忆"""
        memory = BotMemory(bot_id, name, role, self.game_id)
        self.bot_memories[bot_id] = memory
        return memory

    def get_bot_memory(self, bot_id: int) -> Optional[BotMemory]:
        """获取Bot记忆"""
        return self.bot_memories.get(bot_id)

    def broadcast_conversation(self, speaker_name: str, speaker_role: str,
                              content: str, phase: str = "day", exclude_bot_id: Optional[int] = None):
        """向所有Bot广播对话（除了指定排除的）"""
        for bot_id, memory in self.bot_memories.items():
            if exclude_bot_id and bot_id == exclude_bot_id:
                continue
            memory.add_conversation(speaker_name, speaker_role, content, phase)

    def start_new_round_for_all(self, round_number: int):
        """为所有Bot开始新轮次"""
        for memory in self.bot_memories.values():
            memory.start_new_round(round_number)

    def cleanup_game_memories(self):
        """清理游戏记忆文件"""
        memory_dir = f"game_memories/game_{self.game_id}"
        if os.path.exists(memory_dir):
            import shutil
            shutil.rmtree(memory_dir)


if __name__ == "__main__":
    # 测试记忆系统
    memory = BotMemory(1, "测试玩家", Role.SEER, "test_game")
    memory.start_new_round(1)
    memory.add_verification_result(2, "玩家2", "good")
    memory.add_conversation("玩家3", "村民", "我觉得玩家2很可疑")
    memory.update_suspicion("玩家3", 0.7)

    print("记忆上下文：")
    print(memory.get_memory_context())