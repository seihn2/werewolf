"""
轮次控制系统
管理游戏的轮次流程和阶段切换
"""

import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .game_logic import Role, Player, ActionType
from .bot_memory import MemoryManager
from .game_logger import GameLogger, EventType
from .night_actions import NightActionManager


class GamePhase(Enum):
    """游戏阶段"""
    PREPARATION = "preparation"
    NIGHT = "night"
    DAY_DISCUSSION = "day"
    VOTING = "voting"
    ELIMINATION = "elimination"
    GAME_OVER = "game_over"


@dataclass
class RoundResult:
    """轮次结果"""
    round_number: int
    deaths: List[str]
    eliminations: List[str]
    votes: Dict[str, str]
    is_game_over: bool
    winner: Optional[str]


class RoundManager:
    """轮次管理器"""

    def __init__(self, players: List[Player], memory_manager: MemoryManager, logger: GameLogger, conversation_manager=None):
        self.players = players
        self.memory_manager = memory_manager
        self.logger = logger
        self.conversation_manager = conversation_manager

        # 初始化夜晚行动管理器
        self.night_action_manager = NightActionManager(memory_manager, logger, conversation_manager)

        self.current_round = 0
        self.current_phase = GamePhase.PREPARATION
        self.alive_players: List[Player] = players.copy()

        # 游戏状态
        self.game_over = False
        self.winner = None

        # 特殊道具状态
        self.witch_save_used = False
        self.witch_poison_used = False

    def set_player_llm_configs(self, llm_configs: Dict[int, Any]):
        """设置玩家LLM配置到夜晚行动管理器"""
        if hasattr(self.night_action_manager, 'llm_configs'):
            self.night_action_manager.llm_configs = llm_configs

    def get_alive_players(self) -> List[Player]:
        """获取存活玩家"""
        return [p for p in self.players if p.is_alive]

    def get_players_by_role(self, role: Role) -> List[Player]:
        """获取指定角色的存活玩家"""
        return [p for p in self.get_alive_players() if p.role == role]

    def check_game_over(self) -> Tuple[bool, Optional[str]]:
        """检查游戏是否结束"""
        alive_players = self.get_alive_players()
        werewolves = self.get_players_by_role(Role.WEREWOLF)
        good_guys = [p for p in alive_players if p.role != Role.WEREWOLF]

        # 狼人获胜：狼人数量 >= 好人数量
        if len(werewolves) >= len(good_guys):
            return True, "狼人"

        # 好人获胜：所有狼人被淘汰
        if len(werewolves) == 0:
            return True, "好人"

        return False, None

    async def start_new_round(self) -> RoundResult:
        """开始新轮次"""
        self.current_round += 1
        self.logger.log_round_start(self.current_round)
        self.memory_manager.start_new_round_for_all(self.current_round)

        print(f"\n========== 第{self.current_round}轮 ==========")

        # 夜晚阶段
        night_deaths = await self._run_night_phase()

        # 检查游戏是否结束
        is_over, winner = self.check_game_over()
        if is_over:
            self.game_over = True
            self.winner = winner
            self.logger.log_game_end(winner, "所有狼人被淘汰" if winner == "好人" else "狼人数量占优")
            return RoundResult(self.current_round, night_deaths, [], {}, True, winner)

        # 白天讨论阶段
        await self._run_day_discussion_phase()

        # 投票阶段
        votes, eliminated = await self._run_voting_phase()

        # 淘汰阶段
        elimination_deaths = await self._run_elimination_phase(eliminated)

        # 再次检查游戏是否结束
        is_over, winner = self.check_game_over()
        if is_over:
            self.game_over = True
            self.winner = winner
            self.logger.log_game_end(winner, "投票淘汰后游戏结束")

        all_deaths = night_deaths + elimination_deaths
        return RoundResult(
            self.current_round,
            all_deaths,
            [eliminated] if eliminated else [],
            votes,
            is_over,
            winner
        )

    async def _run_night_phase(self) -> List[str]:
        """运行夜晚阶段"""
        self.current_phase = GamePhase.NIGHT
        self.logger.set_round_and_phase(self.current_round, "night")
        self.logger.log_phase_change("night")

        print(f"第{self.current_round}轮夜晚开始...")

        deaths = []

        # 狼人杀人
        werewolves = self.get_players_by_role(Role.WEREWOLF)
        if werewolves:
            victim = await self._werewolf_kill(werewolves)
            if victim:
                # 预言家查验
                seers = self.get_players_by_role(Role.SEER)
                if seers:
                    await self._seer_check(seers[0])

                # 女巫行动
                witches = self.get_players_by_role(Role.WITCH)
                if witches:
                    saved = await self._witch_action(witches[0], victim)
                    if not saved:
                        deaths.append(victim.name)
                        victim.is_alive = False
                        self.logger.log_death(victim.name, "狼人杀害")
                        print(f"玩家{victim.name}被狼人杀害")
                else:
                    # 没有女巫，直接死亡
                    deaths.append(victim.name)
                    victim.is_alive = False
                    self.logger.log_death(victim.name, "狼人杀害")
                    print(f"玩家{victim.name}被狼人杀害")

        await asyncio.sleep(2)  # 夜晚间隔
        return deaths

    async def _werewolf_kill(self, werewolves: List[Player]) -> Optional[Player]:
        """狼人杀人"""
        alive_non_werewolves = [p for p in self.get_alive_players() if p.role != Role.WEREWOLF]
        if not alive_non_werewolves:
            return None

        # 简单策略：随机选择一个非狼人杀死
        victim = random.choice(alive_non_werewolves)

        # 记录到狼人记忆中
        for werewolf in werewolves:
            memory = self.memory_manager.get_bot_memory(werewolf.id)
            if memory:
                memory.add_private_note(f"我们杀死了{victim.name}")

        self.logger.log_special_action(werewolves[0].name, "杀人", victim.name)
        return victim

    async def _seer_check(self, seer: Player):
        """预言家查验 - 使用AI决策"""
        alive_others = [p for p in self.get_alive_players() if p.id != seer.id]
        if not alive_others:
            return

        # 使用AI决策选择查验目标
        target = await self.night_action_manager.seer_check_decision(seer, alive_others)
        if not target:
            target = random.choice(alive_others)

        result = "bad" if target.role == Role.WEREWOLF else "good"

        # 记录到预言家记忆中
        memory = self.memory_manager.get_bot_memory(seer.id)
        if memory:
            memory.add_verification_result(target.id, target.name, result)
            # 记录查验决策思路
            memory.add_private_note(f"我选择查验{target.name}，结果是{result}")

        # 将查验结果记录到日志，但不公开显示
        self.logger.log_verification(seer.name, target.name, result)
        # 只显示预言家进行了查验，不泄露具体目标和结果
        print(f"预言家{seer.name}进行了查验")

    async def _witch_action(self, witch: Player, victim: Player) -> bool:
        """女巫行动 - 使用AI决策"""
        memory = self.memory_manager.get_bot_memory(witch.id)

        # 构建可用行动
        available_actions = {
            'can_save': not self.witch_save_used,
            'can_poison': not self.witch_poison_used
        }

        # 使用AI决策女巫行动
        action = await self.night_action_manager.witch_action_decision(witch, victim, available_actions)

        if not action:
            action = {'action': 'pass'}

        # 执行行动
        if action.get('action') == 'save' and available_actions['can_save']:
            self.witch_save_used = True
            if memory:
                memory.add_private_note(f"我救了{victim.name}")
            self.logger.log_special_action(witch.name, "救人", victim.name)
            print(f"女巫{witch.name}救了{victim.name}")
            return True

        elif action.get('action') == 'poison' and available_actions['can_poison']:
            self.witch_poison_used = True
            # 这里简化处理，实际需要选择目标和执行下毒
            if memory:
                memory.add_private_note(f"我选择下毒")
            self.logger.log_special_action(witch.name, "下毒", "某玩家")
            print(f"女巫{witch.name}使用了毒药")
            return False

        else:
            if memory:
                memory.add_private_note(f"我选择不行动，保留技能")
            print(f"女巫{witch.name}选择不行动")
            return False

    async def _run_day_discussion_phase(self):
        """运行白天讨论阶段"""
        self.current_phase = GamePhase.DAY_DISCUSSION
        self.logger.set_round_and_phase(self.current_round, "day")
        self.logger.log_phase_change("day")

        print(f"第{self.current_round}轮白天讨论开始...")

        alive_players = self.get_alive_players()

        # 每个玩家进行两轮发言
        for speech_round in range(2):
            print(f"\n--- 第{speech_round + 1}轮发言 ---")

            for player in alive_players:
                if not player.is_alive:
                    continue

                memory = self.memory_manager.get_bot_memory(player.id)
                if memory and memory.can_speak_this_round():
                    # 生成发言内容（简化版本）
                    speech = await self._generate_speech(player, memory)

                    # 记录发言
                    self.logger.log_speech(player.name, speech, memory.speech_count_this_round + 1)
                    memory.record_speech()

                    # 广播给其他玩家
                    self.memory_manager.broadcast_conversation(
                        player.name, player.role.value, speech, "day", exclude_bot_id=player.id
                    )

                    print(f"{player.name}：{speech}")
                    await asyncio.sleep(1)  # 发言间隔

    async def _generate_speech(self, player: Player, memory) -> str:
        """生成发言内容"""
        # 优先使用conversation_manager生成AI发言
        if self.conversation_manager:
            try:
                # 添加超时处理，避免AI发言卡死
                speech = await asyncio.wait_for(
                    self.conversation_manager.generate_speech(player, self.current_round, "day"),
                    timeout=30.0  # 30秒超时，给AI充足时间
                )
                if speech and len(speech.strip()) > 20:  # 确保有实际内容
                    return speech
            except asyncio.TimeoutError:
                print(f"{player.name} AI发言超时，使用备用发言")
            except Exception as e:
                print(f"{player.name} AI发言生成失败: {e}")

        # 如果AI发言失败，使用增强版备用发言
        enhanced_speeches = {
            Role.WEREWOLF: [
                "昨晚的死亡情况很奇怪，我觉得可能有女巫参与救人或下毒。大家需要仔细分析谁的行为最可疑。",
                "我强烈怀疑某些人在隐瞒身份，特别是那些一直保持沉默或者发言模糊的玩家。我们必须找出真正的威胁。",
                "从目前的发言来看，我认为预言家的身份存在问题。真正的预言家应该更主动地分享验人信息才对。"
            ],
            Role.SEER: [
                "作为预言家，我昨晚验证了一个人的身份。根据我的查验结果，我建议大家重点关注某些玩家的发言逻辑。",
                "我已经掌握了重要的身份信息，现在需要判断什么时候公开最合适。请大家仔细听我的分析和建议。",
                "基于我的验人结果和观察，我认为我们应该优先淘汰那些行为异常或逻辑矛盾的玩家。"
            ],
            Role.VILLAGER: [
                "作为村民，我没有特殊能力，只能通过观察和分析来判断。目前我觉得某些玩家的发言很值得怀疑。",
                "我仔细分析了大家的发言和行为模式，发现了一些可疑的地方。我们村民必须团结起来找出狼人。",
                "从逻辑推理的角度来看，我倾向于相信预言家的判断，但同时也要防范假预言家的误导。"
            ],
            Role.WITCH: [
                "昨晚我使用了我的能力，现在掌握了一些关键信息。在合适的时机我会分享给大家。",
                "基于我的特殊能力和观察，我对昨晚的事件有不同的看法。建议大家不要急于下结论。",
                "我会在关键时刻发挥我的作用，现在需要更多信息来做出正确的判断。请大家相信我的经验。"
            ],
            Role.HUNTER: [
                "作为猎人，我会在被淘汰时带走一个人。所以狼人们要考虑清楚是否要对我下手。",
                "我正在观察每个人的行为模式，一旦确定目标，我会毫不犹豫地开枪。大家的发言我都记在心里。",
                "现在局面复杂，我需要谨慎选择我的目标。希望大家能提供更多有价值的信息。"
            ],
            Role.GUARD: [
                "昨晚我保护了一个人，现在看来这个决定是否正确还需要进一步验证。",
                "作为守卫，我需要预测狼人的行动模式。根据目前的情况，我对下一步的保护目标有了想法。",
                "守护的艺术在于预判和心理博弈。我会根据大家的发言来调整我的保护策略。"
            ]
        }

        role_speeches = enhanced_speeches.get(player.role, [
            "我正在仔细观察每个人的发言和行为，试图从中找出可疑的地方。目前局面还不够明朗，需要更多信息。"
        ])
        return random.choice(role_speeches)

    async def _run_voting_phase(self) -> Tuple[Dict[str, str], Optional[str]]:
        """运行投票阶段"""
        self.current_phase = GamePhase.VOTING
        self.logger.set_round_and_phase(self.current_round, "voting")
        self.logger.log_phase_change("voting")

        print(f"第{self.current_round}轮投票开始...")

        alive_players = self.get_alive_players()
        votes = {}

        # 每个玩家投票
        for voter in alive_players:
            if not voter.is_alive:
                continue

            # 获取候选人（除了自己）
            candidates = [p for p in alive_players if p.id != voter.id]
            if not candidates:
                continue

            # 使用AI投票决策或备用随机投票
            target_name = None
            if self.conversation_manager:
                try:
                    # 添加超时处理，避免无限等待
                    target_name = await asyncio.wait_for(
                        self.conversation_manager.generate_vote(voter, candidates),
                        timeout=30.0  # 30秒超时，给AI充足时间
                    )
                except asyncio.TimeoutError:
                    print(f"{voter.name} AI投票超时，使用随机投票")
                    target_name = None
                except Exception as e:
                    print(f"{voter.name} AI投票失败: {e}")
                    target_name = None

            # 如果AI投票失败，使用随机投票
            if not target_name:
                target = random.choice(candidates)
                target_name = target.name

            votes[voter.name] = target_name
            self.logger.log_vote(voter.name, target_name)
            print(f"{voter.name} 投票给 {target_name}")

        # 统计票数
        vote_counts = {}
        for target in votes.values():
            vote_counts[target] = vote_counts.get(target, 0) + 1

        # 显示投票结果
        print(f"\n投票统计:")
        for candidate, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {candidate}: {count}票")

        # 找出得票最多的玩家
        if vote_counts:
            max_votes = max(vote_counts.values())
            candidates = [name for name, count in vote_counts.items() if count == max_votes]

            # 如果有平票，随机选择
            eliminated_name = random.choice(candidates)
            self.logger.log_elimination(eliminated_name, max_votes)

            if len(candidates) > 1:
                print(f"\n平票情况，随机选择淘汰: {eliminated_name}")
            else:
                print(f"\n投票结果: {eliminated_name} 被淘汰（得票{max_votes}票）")

            return votes, eliminated_name

        return votes, None

    async def _run_elimination_phase(self, eliminated_name: Optional[str]) -> List[str]:
        """运行淘汰阶段"""
        self.current_phase = GamePhase.ELIMINATION
        self.logger.set_round_and_phase(self.current_round, "elimination")

        deaths = []

        if eliminated_name:
            # 找到被淘汰的玩家
            eliminated_player = None
            for player in self.players:
                if player.name == eliminated_name:
                    eliminated_player = player
                    break

            if eliminated_player:
                eliminated_player.is_alive = False
                deaths.append(eliminated_name)

                # 猎人技能
                if eliminated_player.role == Role.HUNTER:
                    shot_target = await self._hunter_shoot(eliminated_player)
                    if shot_target:
                        deaths.append(shot_target.name)

        return deaths

    async def _hunter_shoot(self, hunter: Player) -> Optional[Player]:
        """猎人开枪"""
        alive_others = [p for p in self.get_alive_players() if p.id != hunter.id]
        if not alive_others:
            return None

        # 简单策略：随机选择
        target = random.choice(alive_others)
        target.is_alive = False

        self.logger.log_special_action(hunter.name, "开枪", target.name)
        self.logger.log_death(target.name, "猎人开枪")
        print(f"猎人{hunter.name}开枪带走了{target.name}")

        return target

    def get_game_summary(self) -> Dict[str, Any]:
        """获取游戏摘要"""
        return {
            "total_rounds": self.current_round,
            "game_over": self.game_over,
            "winner": self.winner,
            "alive_players": [p.name for p in self.get_alive_players()],
            "dead_players": [p.name for p in self.players if not p.is_alive]
        }


if __name__ == "__main__":
    # 测试轮次管理器
    from .bot_memory import MemoryManager
    from .game_logger import GameLogger

    # 创建测试玩家
    test_players = [
        Player(1, "玩家1", Role.WEREWOLF),
        Player(2, "玩家2", Role.WEREWOLF),
        Player(3, "玩家3", Role.SEER),
        Player(4, "玩家4", Role.WITCH),
        Player(5, "玩家5", Role.VILLAGER),
        Player(6, "玩家6", Role.VILLAGER),
    ]

    memory_manager = MemoryManager("test_game")
    logger = GameLogger("test_game")

    # 为每个玩家创建记忆
    for player in test_players:
        memory_manager.create_bot_memory(player.id, player.name, player.role)

    round_manager = RoundManager(test_players, memory_manager, logger)

    async def test_round():
        result = await round_manager.start_new_round()
        print(f"\n轮次结果: {result}")
        print(f"游戏摘要: {round_manager.get_game_summary()}")

    asyncio.run(test_round())