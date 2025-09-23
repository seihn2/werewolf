"""
增强版游戏控制器
整合Bot记忆、游戏日志、轮次控制和对话管理系统
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional

from .game_logic import WerewolfGame, Player, Role
from .bot_memory import MemoryManager
from .game_logger import GameLogger, EventType
from .round_manager import RoundManager
from .conversation_manager import ConversationManager
from .llm_manager import LLMConfig


class EnhancedWerewolfGameController:
    """增强版狼人杀游戏控制器"""

    def __init__(self, player_configs: List[Dict[str, Any]], num_players: int = None):
        """
        初始化游戏控制器

        Args:
            player_configs: 玩家配置列表，格式如:
            [
                {
                    "name": "AI玩家1",
                    "llm_config": LLMConfig对象
                },
                ...
            ]
            num_players: 玩家数量，如果为None则使用player_configs的长度
        """
        self.player_configs = player_configs
        self.num_players = num_players or len(player_configs)
        self.game_id = str(uuid.uuid4())[:8]

        # 核心组件
        self.game: Optional[WerewolfGame] = None
        self.players: List[Player] = []
        self.memory_manager = MemoryManager(self.game_id)
        self.logger = GameLogger(self.game_id)
        self.round_manager: Optional[RoundManager] = None
        self.conversation_manager = ConversationManager(self.memory_manager, self.logger)

        # 游戏状态
        self.game_running = False
        self.game_started = False

    async def initialize_game(self) -> bool:
        """初始化游戏"""
        try:
            print(f"初始化游戏 (ID: {self.game_id})...")

            # 检查配置
            if len(self.player_configs) < 6:
                print("错误：玩家数量至少需要6人")
                return False

            # 创建玩家
            self.players = self._create_players()
            if not self.players:
                return False

            # 分配角色
            self._assign_roles()

            # 创建游戏对象
            self.game = WerewolfGame(self.players)

            # 初始化轮次管理器
            self.round_manager = RoundManager(self.players, self.memory_manager, self.logger, self.conversation_manager)

            # 为每个玩家创建记忆和配置LLM
            for player in self.players:
                # 创建记忆
                self.memory_manager.create_bot_memory(player.id, player.name, player.role)

                # 配置LLM
                llm_config = self._get_llm_config_for_player(player.id)
                if llm_config:
                    self.conversation_manager.set_player_llm_config(player.id, llm_config)

            # 记录游戏开始
            player_info = [{"name": p.name, "role": p.role.value} for p in self.players]
            self.logger.log_game_start(player_info)

            print("游戏初始化成功！")
            print("玩家配置：")
            for player in self.players:
                print(f"  {player.name} - {player.role.value}")

            return True

        except Exception as e:
            print(f"游戏初始化失败: {e}")
            return False

    def _create_players(self) -> List[Player]:
        """创建玩家对象"""
        players = []
        for i, config in enumerate(self.player_configs):
            player = Player(
                id=i + 1,
                name=config["name"],
                role=Role.VILLAGER  # 临时角色，稍后分配
            )
            players.append(player)
        return players

    def _assign_roles(self):
        """分配角色"""
        from .game_logic import GameRule

        num_players = len(self.players)
        roles = GameRule.create_standard_roles(num_players)

        # 分配角色
        for player, role in zip(self.players, roles):
            player.role = role

    def _get_llm_config_for_player(self, player_id: int) -> Optional[LLMConfig]:
        """获取玩家的LLM配置"""
        if player_id <= len(self.player_configs):
            return self.player_configs[player_id - 1].get("llm_config")
        return None

    async def start_game(self):
        """开始游戏"""
        if not self.game or not self.round_manager:
            print("游戏未初始化")
            return

        self.game_running = True
        self.game_started = True

        print(f"\n🎮 游戏开始！")
        print("="*50)

        try:
            round_count = 0
            max_rounds = 20  # 防止无限循环

            while self.game_running and round_count < max_rounds:
                round_count += 1
                print(f"\n🌅 准备第{round_count}轮...")

                # 执行一轮游戏
                round_result = await self.round_manager.start_new_round()

                # 显示轮次结果
                self._display_round_result(round_result)

                # 检查游戏是否结束
                if round_result.is_game_over:
                    self._display_game_end(round_result.winner)
                    break

                # 轮次间隔
                await asyncio.sleep(3)

            if round_count >= max_rounds:
                print("\n⏰ 游戏达到最大轮次限制")
                self.logger.log_game_end("平局", "达到最大轮次限制")

        except KeyboardInterrupt:
            print("\n\n⚠️ 游戏被用户中断")
            self.logger.log_system_message("游戏被用户中断")
        except Exception as e:
            print(f"\n❌ 游戏运行出错: {e}")
            self.logger.log_system_message(f"游戏运行出错: {e}")
        finally:
            self.game_running = False

    def _display_round_result(self, result):
        """显示轮次结果"""
        print(f"\n📊 第{result.round_number}轮结果:")

        if result.deaths:
            print(f"  💀 死亡: {', '.join(result.deaths)}")

        if result.eliminations:
            print(f"  🗳️ 淘汰: {', '.join(result.eliminations)}")

        if result.votes:
            print("  📋 投票结果:")
            for voter, target in result.votes.items():
                print(f"    {voter} -> {target}")

        # 显示当前存活状况
        alive_players = self.round_manager.get_alive_players()
        werewolves = [p for p in alive_players if p.role == Role.WEREWOLF]
        good_guys = [p for p in alive_players if p.role != Role.WEREWOLF]

        print(f"  👥 存活: {len(alive_players)}人 (狼人{len(werewolves)}人, 好人{len(good_guys)}人)")

    def _display_game_end(self, winner: Optional[str]):
        """显示游戏结束"""
        print("\n" + "="*50)
        print("🎉 游戏结束！")
        print("="*50)

        if winner:
            print(f"🏆 获胜方: {winner}")
        else:
            print("🤝 平局")

        # 显示最终状态
        alive_players = self.round_manager.get_alive_players()
        dead_players = [p for p in self.players if not p.is_alive]

        print(f"\n📋 最终状态:")
        print(f"  存活玩家 ({len(alive_players)}人):")
        for player in alive_players:
            print(f"    {player.name} - {player.role.value}")

        if dead_players:
            print(f"  死亡玩家 ({len(dead_players)}人):")
            for player in dead_players:
                print(f"    {player.name} - {player.role.value}")

        # 显示游戏统计
        summary = self.logger.export_summary()
        print(f"\n📈 游戏统计:")
        print(f"  总轮数: {summary['total_rounds']}")
        print(f"  总事件: {summary['total_events']}")
        print(f"  游戏时长: {summary['duration_minutes']:.1f}分钟")

    async def get_game_logs(self) -> str:
        """获取游戏日志"""
        return self.logger.get_formatted_log()

    def get_game_stats(self) -> Dict[str, Any]:
        """获取游戏统计"""
        if not self.round_manager:
            return {}

        base_stats = self.round_manager.get_game_summary()
        log_stats = self.logger.export_summary()

        return {
            **base_stats,
            **log_stats,
            "game_id": self.game_id,
            "game_started": self.game_started,
            "game_running": self.game_running
        }

    def export_game_data(self) -> Dict[str, Any]:
        """导出完整游戏数据"""
        game_data = {
            "game_id": self.game_id,
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "role": p.role.value,
                    "is_alive": p.is_alive
                }
                for p in self.players
            ],
            "stats": self.get_game_stats(),
            "logs": self.logger.events
        }

        # 添加玩家记忆（如果需要的话）
        memories = {}
        for player in self.players:
            memory = self.memory_manager.get_bot_memory(player.id)
            if memory:
                memories[player.name] = {
                    "verification_results": memory.verification_results,
                    "suspicions": memory.suspicions,
                    "private_notes": memory.private_notes
                }

        game_data["memories"] = memories
        return game_data

    def cleanup(self):
        """清理游戏资源"""
        try:
            # 可选：清理记忆文件
            # self.memory_manager.cleanup_game_memories()
            print(f"游戏资源已清理 (游戏ID: {self.game_id})")
        except Exception as e:
            print(f"清理游戏资源失败: {e}")


async def test_enhanced_controller():
    """测试增强版游戏控制器"""
    from .llm_manager import LLMConfig, APIProvider

    # 创建测试配置
    test_configs = []
    for i in range(6):
        config = LLMConfig(
            name=f"test_config_{i}",
            provider=APIProvider.OPENAI,
            api_key="test-key",
            model="gpt-4o-mini"
        )
        test_configs.append({
            "name": f"AI玩家{i+1}",
            "llm_config": config
        })

    # 创建并测试游戏控制器
    controller = EnhancedWerewolfGameController(test_configs)

    if await controller.initialize_game():
        print("✅ 初始化成功，准备开始游戏...")
        # await controller.start_game()  # 注释掉避免实际运行

        print("\n📊 游戏统计:")
        stats = controller.get_game_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    controller.cleanup()


if __name__ == "__main__":
    asyncio.run(test_enhanced_controller())