"""
狼人杀游戏控制器
整合游戏逻辑、AI玩家、LLM管理器的主控制器
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from .game_logic import WerewolfGame, Player, GamePhase, ActionType, Role, GameAction
from .ai_player import WerewolfAIPlayer, create_ai_player
from .llm_manager import MultiLLMManager, LLMConfig


class WerewolfGameController:
    """狼人杀游戏控制器"""

    def __init__(self, player_configs: List[Dict[str, str]]):
        """
        初始化游戏控制器

        Args:
            player_configs: 玩家配置列表，格式如:
            [
                {"name": "GPT玩家", "llm_config": "gpt4o"},
                {"name": "Claude玩家", "llm_config": "claude"},
                ...
            ]
        """
        self.llm_manager = MultiLLMManager()
        self.player_configs = player_configs
        self.game: Optional[WerewolfGame] = None
        self.ai_players: Dict[int, WerewolfAIPlayer] = {}
        self.game_running = False

    async def initialize_game(self) -> bool:
        """初始化游戏"""
        try:
            # 检查配置
            if len(self.player_configs) < 6:
                print("错误：玩家数量至少需要6人")
                return False

            # 验证LLM配置
            for config in self.player_configs:
                llm_config_name = config.get("llm_config")
                if not self.llm_manager.get_config(llm_config_name):
                    print(f"错误：LLM配置 '{llm_config_name}' 不存在")
                    return False

            # 创建游戏
            player_names = [config["name"] for config in self.player_configs]
            self.game = WerewolfGame(player_names)

            # 创建AI玩家
            for i, config in enumerate(self.player_configs):
                game_player = self.game.state.players[i]
                llm_config = self.llm_manager.get_config(config["llm_config"])

                ai_player = create_ai_player(game_player, llm_config)
                self.ai_players[game_player.id] = ai_player

            print(f"游戏初始化成功，{len(player_names)}名AI玩家参与")
            return True

        except Exception as e:
            print(f"游戏初始化失败: {e}")
            return False

    async def start_game(self):
        """开始游戏"""
        if not self.game:
            print("请先初始化游戏")
            return

        self.game_running = True
        self.game.start_game()

        print("=== AI狼人杀游戏开始 ===")
        self._print_game_info()

        try:
            while self.game_running and not self.game.check_game_end():
                await self._run_game_round()

                # 检查游戏是否结束
                if self.game.check_game_end():
                    break

                # 短暂休息
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            print("\n游戏被用户中断")
        except Exception as e:
            print(f"游戏运行出错: {e}")
        finally:
            self.game_running = False

        self._print_game_result()

    async def _run_game_round(self):
        """运行一轮游戏"""
        print(f"\n{'='*50}")
        print(f"第 {self.game.state.current_round} 轮")
        print(f"{'='*50}")

        # 夜晚阶段
        if self.game.state.current_phase == GamePhase.NIGHT:
            await self._run_night_phase()

        # 白天阶段
        await self._run_day_phase()

        # 投票阶段
        await self._run_voting_phase()

        # 检查游戏结束
        if not self.game.check_game_end():
            self.game.next_round()

    async def _run_night_phase(self):
        """运行夜晚阶段"""
        print(f"\n🌙 第{self.game.state.current_round}个夜晚")
        print("=" * 30)

        # 收集所有夜晚行动
        night_actions = []

        # 狼人行动
        werewolves = self.game.state.get_players_by_role(Role.WEREWOLF)
        if werewolves:
            print("\n🐺 狼人阶段")
            action = await self._get_collective_werewolf_action(werewolves)
            if action:
                night_actions.append(action)

        # 预言家行动
        seers = self.game.state.get_players_by_role(Role.SEER)
        for seer in seers:
            print(f"\n🔮 {seer.name} (预言家) 的回合")
            action = await self._get_player_action(seer, [ActionType.CHECK])
            if action:
                night_actions.append(action)

        # 女巫行动
        witches = self.game.state.get_players_by_role(Role.WITCH)
        for witch in witches:
            print(f"\n🧙 {witch.name} (女巫) 的回合")
            available_actions = [ActionType.SAVE, ActionType.POISON]
            action = await self._get_player_action(witch, available_actions)
            if action:
                night_actions.append(action)

        # 守卫行动
        guards = self.game.state.get_players_by_role(Role.GUARD)
        for guard in guards:
            print(f"\n🛡️ {guard.name} (守卫) 的回合")
            action = await self._get_player_action(guard, [ActionType.GUARD])
            if action:
                night_actions.append(action)

        # 执行所有夜晚行动
        for action in night_actions:
            self.game.perform_action(action.player_id, action)

        print(f"\n夜晚结束，共收集 {len(night_actions)} 个行动")

    async def _get_collective_werewolf_action(self, werewolves: List[Player]) -> Optional[GameAction]:
        """获取狼人集体行动"""
        if not werewolves:
            return None

        # 让每个狼人提出建议
        suggestions = []
        for wolf in werewolves:
            print(f"  {wolf.name} 正在考虑击杀目标...")
            ai_player = self.ai_players[wolf.id]
            action = await ai_player.decide_action(self.game.state, [ActionType.KILL])
            if action and action.target_id:
                suggestions.append((wolf, action.target_id))

        if not suggestions:
            print("  狼人们决定今晚不杀人")
            return None

        # 选择得票最多的目标（简单多数决）
        target_votes = {}
        for wolf, target_id in suggestions:
            target_votes[target_id] = target_votes.get(target_id, 0) + 1
            target_name = self.game.state.get_player_by_id(target_id).name
            print(f"  {wolf.name} 建议击杀 {target_name}")

        # 找出得票最多的目标
        if target_votes:
            target_id = max(target_votes.keys(), key=lambda x: target_votes[x])
            target_name = self.game.state.get_player_by_id(target_id).name
            print(f"  狼人决定击杀 {target_name}")

            return GameAction(
                player_id=werewolves[0].id,  # 代表狼人团队
                action_type=ActionType.KILL,
                target_id=target_id
            )

        return None

    async def _get_player_action(self, player: Player, available_actions: List[ActionType]) -> Optional[GameAction]:
        """获取单个玩家的行动"""
        ai_player = self.ai_players[player.id]
        action = await ai_player.decide_action(self.game.state, available_actions)

        if action:
            target_name = "无目标"
            if action.target_id:
                target = self.game.state.get_player_by_id(action.target_id)
                target_name = target.name if target else "未知"

            print(f"  {player.name} 选择: {action.action_type.value} -> {target_name}")
        else:
            print(f"  {player.name} 选择: 不行动")

        return action

    async def _run_day_phase(self):
        """运行白天阶段"""
        self.game.start_day_phase()

        print(f"\n☀️ 第{self.game.state.current_round}个白天")
        print("=" * 30)

        # 显示夜晚结果
        if self.game.state.game_log:
            recent_events = [log for log in self.game.state.game_log
                           if f"第{self.game.state.current_round}轮" in log]
            for event in recent_events[-3:]:
                print(f"📰 {event}")

        # 自由发言阶段
        await self._run_discussion_phase()

    async def _run_discussion_phase(self):
        """运行讨论阶段"""
        print("\n💬 自由讨论阶段")
        print("-" * 20)

        alive_players = self.game.state.get_alive_players()

        # 每个玩家发言
        for player in alive_players:
            print(f"\n{player.name} 的发言:")
            ai_player = self.ai_players[player.id]

            speech_context = self._build_speech_context()
            speech = await ai_player.make_speech(self.game.state, speech_context)

            if speech:
                print(f"  💭 {speech}")
            else:
                print(f"  💭 {player.name} 选择沉默")

            # 短暂停顿
            await asyncio.sleep(1)

    def _build_speech_context(self) -> str:
        """构建发言上下文"""
        alive_count = len(self.game.state.get_alive_players())
        recent_events = self.game.state.game_log[-2:] if self.game.state.game_log else []

        context = f"当前存活 {alive_count} 人。"
        if recent_events:
            context += f" 最近事件: {'; '.join(recent_events)}"

        return context

    async def _run_voting_phase(self):
        """运行投票阶段"""
        self.game.start_voting_phase()

        print(f"\n🗳️ 投票阶段")
        print("=" * 20)

        alive_players = self.game.state.get_alive_players()

        # 收集投票
        votes = []
        for player in alive_players:
            print(f"\n{player.name} 正在投票...")
            ai_player = self.ai_players[player.id]

            action = await ai_player.decide_action(self.game.state, [ActionType.VOTE])
            if action and action.target_id:
                target = self.game.state.get_player_by_id(action.target_id)
                if target and target.is_alive:
                    votes.append((player, target))
                    self.game.perform_action(player.id, action)
                    print(f"  {player.name} 投票给 {target.name}")
                else:
                    print(f"  {player.name} 投票无效")
            else:
                print(f"  {player.name} 弃票")

        # 统计投票结果
        print(f"\n📊 投票结果:")
        for target_id, count in self.game.state.vote_counts.items():
            target = self.game.state.get_player_by_id(target_id)
            if target:
                print(f"  {target.name}: {count} 票")

        # 执行投票结果
        eliminated = self.game.resolve_voting()
        if eliminated:
            print(f"\n⚱️ {eliminated.name} 被投票出局")
            print(f"   真实身份: {eliminated.role.value}")
        else:
            print(f"\n🤝 没有人被投票出局")

    def _print_game_info(self):
        """打印游戏信息"""
        print(f"\n📋 游戏信息:")
        print(f"  玩家数量: {len(self.game.state.players)}")

        # 统计角色
        role_counts = {}
        for player in self.game.state.players:
            role = player.role.value
            role_counts[role] = role_counts.get(role, 0) + 1

        print(f"  角色配置: {', '.join([f'{role}x{count}' for role, count in role_counts.items()])}")

        print(f"\n👥 AI玩家列表:")
        for i, config in enumerate(self.player_configs):
            player = self.game.state.players[i]
            llm_name = config["llm_config"]
            print(f"  {player.name}: {player.role.value} (使用 {llm_name})")

    def _print_game_result(self):
        """打印游戏结果"""
        print(f"\n{'='*60}")
        print(f"🎮 游戏结束")
        print(f"{'='*60}")

        if self.game.state.winner:
            print(f"🏆 获胜方: {self.game.state.winner}")

        print(f"\n📊 最终统计:")
        print(f"  总轮数: {self.game.state.current_round}")
        print(f"  总行动数: {len(self.game.state.actions_history)}")

        print(f"\n👥 玩家结果:")
        for i, player in enumerate(self.game.state.players):
            status = "🟢存活" if player.is_alive else "💀死亡"
            llm_name = self.player_configs[i]["llm_config"]
            print(f"  {player.name}: {player.role.value} {status} (AI: {llm_name})")

        print(f"\n📜 游戏日志:")
        for log in self.game.state.game_log[-10:]:  # 显示最后10条日志
            print(f"  {log}")

    def stop_game(self):
        """停止游戏"""
        self.game_running = False
        print("游戏已停止")

    def get_game_stats(self) -> Dict[str, Any]:
        """获取游戏统计"""
        if not self.game:
            return {}

        stats = {
            "total_rounds": self.game.state.current_round,
            "total_actions": len(self.game.state.actions_history),
            "winner": self.game.state.winner,
            "players": []
        }

        for i, player in enumerate(self.game.state.players):
            llm_config = self.player_configs[i]["llm_config"]
            stats["players"].append({
                "name": player.name,
                "role": player.role.value,
                "is_alive": player.is_alive,
                "llm_config": llm_config
            })

        return stats


# 工具函数
def create_game_controller(player_configs: List[Dict[str, str]]) -> WerewolfGameController:
    """创建游戏控制器"""
    return WerewolfGameController(player_configs)


async def run_demo_game():
    """运行演示游戏"""
    # 示例配置
    demo_configs = [
        {"name": "GPT玩家1", "llm_config": "gpt4o"},
        {"name": "Claude玩家", "llm_config": "claude"},
        {"name": "DeepSeek玩家", "llm_config": "deepseek"},
        {"name": "Qwen玩家", "llm_config": "qwen"},
        {"name": "GPT玩家2", "llm_config": "gpt4o"},
        {"name": "本地玩家", "llm_config": "deepseek"},
    ]

    controller = create_game_controller(demo_configs)

    if await controller.initialize_game():
        await controller.start_game()

        stats = controller.get_game_stats()
        print(f"\n📈 游戏统计: {stats}")
    else:
        print("游戏初始化失败")


if __name__ == "__main__":
    # 运行演示
    print("AI狼人杀游戏控制器")
    print("注意：需要先配置LLM API才能运行")

    # asyncio.run(run_demo_game())