"""
游戏日志系统
记录完整的游戏进程，提供标准化的日志格式
"""

import json
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class EventType(Enum):
    """事件类型"""
    GAME_START = "game_start"
    GAME_END = "game_end"
    ROUND_START = "round_start"
    PHASE_CHANGE = "phase_change"
    DEATH = "death"
    SPEECH = "speech"
    VOTE = "vote"
    ELIMINATION = "elimination"
    VERIFICATION = "verification"
    SPECIAL_ACTION = "special_action"
    SYSTEM_MESSAGE = "system_message"


@dataclass
class GameEvent:
    """游戏事件"""
    timestamp: float
    round_number: int
    phase: str  # "night", "day", "voting", "elimination"
    event_type: EventType
    player_name: Optional[str]
    content: str
    metadata: Dict[str, Any] = None


class GameLogger:
    """游戏日志记录器"""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.events: List[GameEvent] = []
        self.current_round = 0
        self.current_phase = "preparation"

        # 日志文件路径
        self.log_dir = f"game_logs"
        self.log_file = f"{self.log_dir}/game_{game_id}.log"
        self.json_file = f"{self.log_dir}/game_{game_id}.json"

        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化日志
        self._write_header()

    def _write_header(self):
        """写入日志头部"""
        header = f"""
========================================
AI狼人杀游戏日志
游戏ID: {self.game_id}
开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
========================================
"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(header)

    def log_event(self, event_type: EventType, content: str, player_name: Optional[str] = None,
                  metadata: Dict[str, Any] = None):
        """记录游戏事件"""
        event = GameEvent(
            timestamp=time.time(),
            round_number=self.current_round,
            phase=self.current_phase,
            event_type=event_type,
            player_name=player_name,
            content=content,
            metadata=metadata or {}
        )

        self.events.append(event)
        self._write_to_log(event)
        self._save_json()

    def _write_to_log(self, event: GameEvent):
        """写入文本日志"""
        timestamp_str = time.strftime('%H:%M:%S', time.localtime(event.timestamp))

        if event.round_number > 0:
            round_phase = f"[第{event.round_number}轮-{event.phase}]"
        else:
            round_phase = f"[{event.phase}]"

        if event.player_name:
            log_line = f"{timestamp_str} {round_phase} {event.player_name}：{event.content}\n"
        else:
            log_line = f"{timestamp_str} {round_phase} 系统：{event.content}\n"

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)

    def _save_json(self):
        """保存JSON格式的完整日志"""
        try:
            # 转换事件为可序列化的格式
            serializable_events = []
            for event in self.events:
                event_dict = asdict(event)
                event_dict['event_type'] = event.event_type.value  # 转换枚举为字符串
                serializable_events.append(event_dict)

            log_data = {
                "game_id": self.game_id,
                "events": serializable_events,
                "last_updated": time.time()
            }

            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"保存JSON日志失败: {e}")

    def set_round_and_phase(self, round_number: int, phase: str):
        """设置当前轮次和阶段"""
        self.current_round = round_number
        self.current_phase = phase

    def log_game_start(self, players: List[Dict[str, Any]]):
        """记录游戏开始"""
        player_info = []
        for player in players:
            player_info.append(f"{player['name']}({player['role']})")

        content = f"游戏开始，参与玩家：{', '.join(player_info)}"
        self.log_event(EventType.GAME_START, content, metadata={"players": players})

    def log_round_start(self, round_number: int):
        """记录轮次开始"""
        self.set_round_and_phase(round_number, "night")
        content = f"第{round_number}轮开始"
        self.log_event(EventType.ROUND_START, content)

    def log_phase_change(self, new_phase: str):
        """记录阶段变化"""
        old_phase = self.current_phase
        self.current_phase = new_phase
        content = f"{old_phase} -> {new_phase}"
        self.log_event(EventType.PHASE_CHANGE, content)

    def log_death(self, victim_name: str, cause: str = "狼人杀害"):
        """记录死亡事件"""
        content = f"{victim_name}被{cause}"
        self.log_event(EventType.DEATH, content, metadata={"victim": victim_name, "cause": cause})

    def log_speech(self, speaker_name: str, speech_content: str, speech_count: int = 1):
        """记录发言"""
        content = f"{speech_content}"
        metadata = {"speech_count": speech_count, "content": speech_content}
        self.log_event(EventType.SPEECH, content, speaker_name, metadata)

    def log_vote(self, voter_name: str, target_name: str):
        """记录投票"""
        content = f"投票给{target_name}"
        metadata = {"voter": voter_name, "target": target_name}
        self.log_event(EventType.VOTE, content, voter_name, metadata)

    def log_elimination(self, eliminated_name: str, vote_count: int):
        """记录淘汰"""
        content = f"{eliminated_name}被投票淘汰（得票{vote_count}票）"
        metadata = {"eliminated": eliminated_name, "votes": vote_count}
        self.log_event(EventType.ELIMINATION, content, metadata=metadata)

    def log_verification(self, seer_name: str, target_name: str, result: str):
        """记录查验（仅系统知道，不公开）"""
        content = f"{seer_name}查验{target_name}，结果：{result}"
        metadata = {"seer": seer_name, "target": target_name, "result": result}
        self.log_event(EventType.VERIFICATION, content, metadata=metadata)

    def log_special_action(self, player_name: str, action: str, target: Optional[str] = None):
        """记录特殊行动"""
        if target:
            content = f"对{target}使用{action}"
        else:
            content = f"使用{action}"
        metadata = {"action": action, "target": target}
        self.log_event(EventType.SPECIAL_ACTION, content, player_name, metadata)

    def log_system_message(self, message: str):
        """记录系统消息"""
        self.log_event(EventType.SYSTEM_MESSAGE, message)

    def log_game_end(self, winner: str, reason: str):
        """记录游戏结束"""
        content = f"游戏结束，{winner}获胜。原因：{reason}"
        metadata = {"winner": winner, "reason": reason}
        self.log_event(EventType.GAME_END, content, metadata=metadata)

    def get_recent_events(self, count: int = 10) -> List[GameEvent]:
        """获取最近的事件"""
        return self.events[-count:]

    def get_round_events(self, round_number: int) -> List[GameEvent]:
        """获取指定轮次的所有事件"""
        return [event for event in self.events if event.round_number == round_number]

    def get_player_speeches(self, player_name: str, round_number: Optional[int] = None) -> List[GameEvent]:
        """获取指定玩家的发言记录"""
        speeches = [event for event in self.events
                   if event.event_type == EventType.SPEECH and event.player_name == player_name]

        if round_number is not None:
            speeches = [speech for speech in speeches if speech.round_number == round_number]

        return speeches

    def get_formatted_log(self, last_n_rounds: int = 2) -> str:
        """获取格式化的日志内容（用于AI上下文）"""
        if last_n_rounds <= 0:
            relevant_events = self.events
        else:
            min_round = max(1, self.current_round - last_n_rounds + 1)
            relevant_events = [event for event in self.events
                             if event.round_number >= min_round or event.round_number == 0]

        log_lines = []
        for event in relevant_events[-30:]:  # 最近30个事件
            if event.round_number > 0:
                round_phase = f"[第{event.round_number}轮-{event.phase}]"
            else:
                round_phase = f"[{event.phase}]"

            if event.player_name:
                log_lines.append(f"{round_phase} {event.player_name}：{event.content}")
            else:
                log_lines.append(f"{round_phase} 系统：{event.content}")

        return "\n".join(log_lines)

    def export_summary(self) -> Dict[str, Any]:
        """导出游戏摘要"""
        speech_counts = {}
        deaths = []
        eliminations = []

        for event in self.events:
            if event.event_type == EventType.SPEECH and event.player_name:
                speech_counts[event.player_name] = speech_counts.get(event.player_name, 0) + 1
            elif event.event_type == EventType.DEATH:
                deaths.append(event.metadata.get("victim"))
            elif event.event_type == EventType.ELIMINATION:
                eliminations.append(event.metadata.get("eliminated"))

        return {
            "game_id": self.game_id,
            "total_rounds": self.current_round,
            "total_events": len(self.events),
            "speech_counts": speech_counts,
            "deaths": deaths,
            "eliminations": eliminations,
            "duration_minutes": (time.time() - self.events[0].timestamp) / 60 if self.events else 0
        }


if __name__ == "__main__":
    # 测试日志系统
    logger = GameLogger("test_game")

    # 模拟游戏事件
    logger.log_game_start([
        {"name": "玩家1", "role": "werewolf"},
        {"name": "玩家2", "role": "seer"},
        {"name": "玩家3", "role": "villager"}
    ])

    logger.log_round_start(1)
    logger.log_death("玩家3", "狼人杀害")
    logger.log_phase_change("day")
    logger.log_speech("玩家2", "我是预言家，昨晚验了玩家1是狼人", 1)
    logger.log_speech("玩家1", "玩家2在说谎，我是好人", 1)

    print("格式化日志：")
    print(logger.get_formatted_log())

    print("\n游戏摘要：")
    print(json.dumps(logger.export_summary(), ensure_ascii=False, indent=2))