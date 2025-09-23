"""
狼人杀游戏逻辑框架
定义游戏规则、角色、阶段等核心逻辑
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import random
from abc import ABC, abstractmethod


class Role(Enum):
    """游戏角色"""
    VILLAGER = "villager"      # 村民
    WEREWOLF = "werewolf"      # 狼人
    SEER = "seer"              # 预言家
    WITCH = "witch"            # 女巫
    HUNTER = "hunter"          # 猎人
    GUARD = "guard"            # 守卫


class GamePhase(Enum):
    """游戏阶段"""
    PREPARATION = "preparation"    # 准备阶段
    NIGHT = "night"               # 夜晚
    DAY = "day"                   # 白天
    VOTING = "voting"             # 投票阶段
    GAME_OVER = "game_over"       # 游戏结束


class ActionType(Enum):
    """行动类型"""
    KILL = "kill"              # 杀人
    CHECK = "check"            # 查验
    SAVE = "save"              # 救人
    POISON = "poison"          # 下毒
    SHOOT = "shoot"            # 开枪
    GUARD = "guard"            # 守护
    VOTE = "vote"              # 投票
    SPEAK = "speak"            # 发言


@dataclass
class Player:
    """玩家类"""
    id: int
    name: str
    role: Role
    is_alive: bool = True
    is_protected: bool = False     # 是否被守护
    is_poisoned: bool = False      # 是否被下毒
    is_saved: bool = False         # 是否被救治
    vote_target: Optional[int] = None    # 投票目标
    last_action: Optional[Dict[str, Any]] = None   # 最后一次行动

    def __str__(self):
        status = "存活" if self.is_alive else "死亡"
        return f"{self.name}({self.role.value}) - {status}"


@dataclass
class GameAction:
    """游戏行动"""
    player_id: int
    action_type: ActionType
    target_id: Optional[int] = None
    message: str = ""
    phase: GamePhase = GamePhase.DAY
    round_number: int = 0


@dataclass
class GameState:
    """游戏状态"""
    players: List[Player] = field(default_factory=list)
    current_phase: GamePhase = GamePhase.PREPARATION
    current_round: int = 0
    actions_history: List[GameAction] = field(default_factory=list)
    night_actions: Dict[int, GameAction] = field(default_factory=dict)
    vote_counts: Dict[int, int] = field(default_factory=dict)
    game_log: List[str] = field(default_factory=list)
    winner: Optional[str] = None

    def get_alive_players(self) -> List[Player]:
        """获取存活玩家"""
        return [p for p in self.players if p.is_alive]

    def get_players_by_role(self, role: Role) -> List[Player]:
        """按角色获取玩家"""
        return [p for p in self.players if p.role == role and p.is_alive]

    def get_player_by_id(self, player_id: int) -> Optional[Player]:
        """根据ID获取玩家"""
        for player in self.players:
            if player.id == player_id:
                return player
        return None


class GameRule:
    """游戏规则"""

    @staticmethod
    def create_standard_roles(num_players: int) -> List[Role]:
        """创建标准角色配置"""
        if num_players < 6:
            raise ValueError("玩家数量至少需要6人")

        roles = []

        # 根据人数分配角色
        if num_players == 6:
            # 6人局：2狼3民1预言家
            roles = [Role.WEREWOLF, Role.WEREWOLF,
                    Role.VILLAGER, Role.VILLAGER, Role.VILLAGER,
                    Role.SEER]
        elif num_players == 8:
            # 8人局：2狼4民1预言家1女巫
            roles = [Role.WEREWOLF, Role.WEREWOLF,
                    Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER,
                    Role.SEER, Role.WITCH]
        elif num_players == 9:
            # 9人局：3狼4民1预言家1女巫
            roles = [Role.WEREWOLF, Role.WEREWOLF, Role.WEREWOLF,
                    Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER,
                    Role.SEER, Role.WITCH]
        elif num_players == 10:
            # 10人局：3狼4民1预言家1女巫1猎人
            roles = [Role.WEREWOLF, Role.WEREWOLF, Role.WEREWOLF,
                    Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER,
                    Role.SEER, Role.WITCH, Role.HUNTER]
        elif num_players == 12:
            # 12人局：4狼4民1预言家1女巫1猎人1守卫
            roles = [Role.WEREWOLF, Role.WEREWOLF, Role.WEREWOLF, Role.WEREWOLF,
                    Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER,
                    Role.SEER, Role.WITCH, Role.HUNTER, Role.GUARD]
        else:
            # 动态按比例分配
            num_werewolves = max(1, num_players // 3)
            num_good = num_players - num_werewolves

            # 先分配狼人
            roles = [Role.WEREWOLF] * num_werewolves

            # 好人角色分配优先级：预言家 > 女巫 > 猎人 > 守卫 > 村民
            special_roles = [Role.SEER, Role.WITCH, Role.HUNTER, Role.GUARD]
            assigned_special = 0

            # 分配特殊角色
            for role in special_roles:
                if assigned_special < num_good and assigned_special < len(special_roles):
                    roles.append(role)
                    assigned_special += 1

            # 剩余位置分配村民
            remaining_villagers = num_good - assigned_special
            roles.extend([Role.VILLAGER] * remaining_villagers)

        random.shuffle(roles)
        return roles

    @staticmethod
    def check_win_condition(state: GameState) -> Optional[str]:
        """检查胜利条件"""
        alive_players = state.get_alive_players()
        werewolves = [p for p in alive_players if p.role == Role.WEREWOLF]
        good_guys = [p for p in alive_players if p.role != Role.WEREWOLF]

        if not werewolves:
            return "好人获胜"
        elif len(werewolves) >= len(good_guys):
            return "狼人获胜"

        return None

    @staticmethod
    def can_perform_action(player: Player, action_type: ActionType,
                          phase: GamePhase) -> bool:
        """检查玩家是否可以执行特定行动"""
        if not player.is_alive:
            return False

        # 夜晚行动权限
        if phase == GamePhase.NIGHT:
            if action_type == ActionType.KILL and player.role == Role.WEREWOLF:
                return True
            elif action_type == ActionType.CHECK and player.role == Role.SEER:
                return True
            elif action_type in [ActionType.SAVE, ActionType.POISON] and player.role == Role.WITCH:
                return True
            elif action_type == ActionType.GUARD and player.role == Role.GUARD:
                return True

        # 白天行动权限
        elif phase == GamePhase.DAY:
            if action_type == ActionType.SPEAK:
                return True

        # 投票阶段权限
        elif phase == GamePhase.VOTING:
            if action_type == ActionType.VOTE:
                return True

        return False


class WerewolfGame:
    """狼人杀游戏控制器"""

    def __init__(self, player_names: List[str]):
        if len(player_names) < 6:
            raise ValueError("玩家数量至少需要6人")

        self.state = GameState()
        self._initialize_game(player_names)

    def _initialize_game(self, player_names: List[str]):
        """初始化游戏"""
        num_players = len(player_names)
        roles = GameRule.create_standard_roles(num_players)

        # 创建玩家
        for i, name in enumerate(player_names):
            player = Player(
                id=i,
                name=name,
                role=roles[i]
            )
            self.state.players.append(player)

        self.state.current_phase = GamePhase.PREPARATION
        self.add_log(f"游戏初始化完成，{num_players}名玩家参与")

        # 记录角色分配
        role_counts = {}
        for role in roles:
            role_counts[role.value] = role_counts.get(role.value, 0) + 1

        role_summary = ", ".join([f"{role}x{count}" for role, count in role_counts.items()])
        self.add_log(f"角色配置: {role_summary}")

    def start_game(self):
        """开始游戏"""
        self.state.current_phase = GamePhase.NIGHT
        self.state.current_round = 1
        self.add_log("=== 游戏开始 ===")
        self.add_log("第1个夜晚开始")

    def start_night_phase(self):
        """开始夜晚阶段"""
        self.state.current_phase = GamePhase.NIGHT
        self.state.night_actions.clear()

        # 重置保护状态
        for player in self.state.players:
            player.is_protected = False

        self.add_log(f"第{self.state.current_round}个夜晚开始")

    def start_day_phase(self):
        """开始白天阶段"""
        self.state.current_phase = GamePhase.DAY
        self._resolve_night_actions()
        self.add_log(f"第{self.state.current_round}个白天开始")

    def start_voting_phase(self):
        """开始投票阶段"""
        self.state.current_phase = GamePhase.VOTING
        self.state.vote_counts.clear()

        # 清除投票目标
        for player in self.state.players:
            player.vote_target = None

        self.add_log("投票阶段开始")

    def _resolve_night_actions(self):
        """解决夜晚行动"""
        # 先处理守卫行动
        guard_action = self._get_night_action(ActionType.GUARD)
        if guard_action and guard_action.target_id is not None:
            target = self.state.get_player_by_id(guard_action.target_id)
            if target:
                target.is_protected = True
                self.add_log(f"守卫保护了{target.name}")

        # 处理狼人杀人
        kill_action = self._get_night_action(ActionType.KILL)
        killed_player = None
        if kill_action and kill_action.target_id is not None:
            target = self.state.get_player_by_id(kill_action.target_id)
            if target and not target.is_protected:
                killed_player = target
                self.add_log(f"狼人杀死了{target.name}")

        # 处理女巫行动
        save_action = self._get_night_action(ActionType.SAVE)
        poison_action = self._get_night_action(ActionType.POISON)

        # 女巫救人
        if save_action and killed_player:
            killed_player.is_saved = True
            killed_player = None  # 被救活
            self.add_log("女巫救了人")

        # 女巫下毒
        poisoned_player = None
        if poison_action and poison_action.target_id is not None:
            target = self.state.get_player_by_id(poison_action.target_id)
            if target:
                poisoned_player = target
                self.add_log(f"女巫毒死了{target.name}")

        # 执行死亡
        dead_players = []
        if killed_player:
            killed_player.is_alive = False
            dead_players.append(killed_player)

        if poisoned_player:
            poisoned_player.is_alive = False
            dead_players.append(poisoned_player)

        if dead_players:
            names = [p.name for p in dead_players]
            self.add_log(f"昨夜死亡: {', '.join(names)}")
        else:
            self.add_log("昨夜平安")

    def _get_night_action(self, action_type: ActionType) -> Optional[GameAction]:
        """获取特定类型的夜晚行动"""
        for action in self.state.night_actions.values():
            if action.action_type == action_type:
                return action
        return None

    def perform_action(self, player_id: int, action: GameAction) -> bool:
        """执行玩家行动"""
        player = self.state.get_player_by_id(player_id)
        if not player:
            return False

        # 检查行动是否合法
        if not GameRule.can_perform_action(player, action.action_type, self.state.current_phase):
            return False

        # 记录行动
        action.player_id = player_id
        action.phase = self.state.current_phase
        action.round_number = self.state.current_round

        if self.state.current_phase == GamePhase.NIGHT:
            self.state.night_actions[player_id] = action
        elif action.action_type == ActionType.VOTE:
            player.vote_target = action.target_id
            self.state.vote_counts[action.target_id] = self.state.vote_counts.get(action.target_id, 0) + 1

        self.state.actions_history.append(action)
        player.last_action = {
            'type': action.action_type.value,
            'target': action.target_id,
            'round': self.state.current_round
        }

        return True

    def resolve_voting(self) -> Optional[Player]:
        """解决投票"""
        if not self.state.vote_counts:
            self.add_log("没有人投票")
            return None

        # 找出得票最多的玩家
        max_votes = max(self.state.vote_counts.values())
        candidates = [pid for pid, votes in self.state.vote_counts.items() if votes == max_votes]

        if len(candidates) > 1:
            # 平票，随机选择一个或者无人出局
            self.add_log(f"平票情况：{[self.state.get_player_by_id(pid).name for pid in candidates]}")
            return None

        # 执行出局
        eliminated_id = candidates[0]
        eliminated_player = self.state.get_player_by_id(eliminated_id)
        eliminated_player.is_alive = False

        self.add_log(f"{eliminated_player.name}被投票出局")
        return eliminated_player

    def check_game_end(self) -> bool:
        """检查游戏是否结束"""
        winner = GameRule.check_win_condition(self.state)
        if winner:
            self.state.winner = winner
            self.state.current_phase = GamePhase.GAME_OVER
            self.add_log(f"=== 游戏结束：{winner} ===")
            return True
        return False

    def next_round(self):
        """进入下一轮"""
        self.state.current_round += 1
        self.start_night_phase()

    def add_log(self, message: str):
        """添加游戏日志"""
        self.state.game_log.append(f"[第{self.state.current_round}轮] {message}")

    def get_game_summary(self) -> Dict[str, Any]:
        """获取游戏摘要"""
        alive_players = self.state.get_alive_players()
        return {
            "current_round": self.state.current_round,
            "current_phase": self.state.current_phase.value,
            "alive_players": len(alive_players),
            "players_info": [
                {
                    "id": p.id,
                    "name": p.name,
                    "role": p.role.value,
                    "is_alive": p.is_alive
                }
                for p in self.state.players
            ],
            "winner": self.state.winner,
            "recent_logs": self.state.game_log[-5:] if self.state.game_log else []
        }


# 工具函数
def create_test_game() -> WerewolfGame:
    """创建测试游戏"""
    player_names = [
        "Alice", "Bob", "Charlie", "Diana",
        "Eve", "Frank", "Grace", "Henry"
    ]
    return WerewolfGame(player_names)


if __name__ == "__main__":
    # 测试游戏逻辑
    game = create_test_game()
    game.start_game()

    print("游戏初始化完成")
    print(f"玩家配置：")
    for player in game.state.players:
        print(f"  {player}")

    summary = game.get_game_summary()
    print(f"\n游戏摘要: {summary}")