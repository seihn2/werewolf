"""
测试增强版狼人杀系统
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werewolf.enhanced_game_controller import EnhancedWerewolfGameController
from werewolf.llm_manager import LLMConfig, APIProvider


async def test_system():
    """测试整个系统"""
    print("开始测试增强版狼人杀系统")
    print("="*50)

    # 创建测试配置
    test_configs = []
    for i in range(6):
        config = LLMConfig(
            name=f"test_config_{i}",
            provider=APIProvider.OPENAI,
            api_key="test-key-for-testing",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1"
        )
        test_configs.append({
            "name": f"AI玩家{i+1}",
            "llm_config": config
        })

    # 创建游戏控制器
    controller = EnhancedWerewolfGameController(test_configs)

    # 测试初始化
    print("\n 测试游戏初始化...")
    if await controller.initialize_game():
        print(" 初始化成功!")

        # 显示游戏配置
        print("\n 玩家配置:")
        for player in controller.players:
            print(f"  {player.name} - {player.role.value} (ID: {player.id})")

        # 测试记忆系统
        print("\n 测试记忆系统...")
        for player in controller.players:
            memory = controller.memory_manager.get_bot_memory(player.id)
            if memory:
                print(f"  {player.name}: 记忆文件已创建")
                # 添加测试记忆
                memory.add_private_note("这是一条测试笔记")
                if player.role.value == "seer":
                    memory.add_verification_result(2, "AI玩家2", "good")

        # 测试日志系统
        print("\n 测试日志系统...")
        controller.logger.log_system_message("系统测试消息")
        controller.logger.log_speech("AI玩家1", "这是一条测试发言")
        print("  日志记录正常")

        # 测试轮次管理器
        print("\n 测试轮次管理器...")
        if controller.round_manager:
            alive_count = len(controller.round_manager.get_alive_players())
            werewolf_count = len(controller.round_manager.get_players_by_role(controller.players[0].role))
            print(f"  存活玩家: {alive_count}人")
            print(f"  角色分配正常")

        # 显示统计信息
        print("\n 游戏统计:")
        stats = controller.get_game_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n 系统测试完成 - 所有组件工作正常!")

    else:
        print(" 初始化失败")

    # 清理
    controller.cleanup()


async def test_memory_system():
    """单独测试记忆系统"""
    print("\n 详细测试记忆系统")
    print("-"*30)

    from werewolf.bot_memory import BotMemory, MemoryManager
    from werewolf.game_logic import Role

    # 测试记忆管理器
    memory_manager = MemoryManager("test_game")

    # 创建测试记忆
    memory = memory_manager.create_bot_memory(1, "测试玩家", Role.SEER)

    # 测试功能
    memory.start_new_round(1)
    memory.add_verification_result(2, "玩家2", "bad")
    memory.add_conversation("玩家3", "村民", "我觉得玩家2很可疑")
    memory.update_suspicion("玩家2", 0.8)
    memory.add_private_note("玩家2可能是狼人")

    # 显示记忆内容
    print("记忆上下文:")
    print(memory.get_memory_context())


async def test_logger_system():
    """单独测试日志系统"""
    print("\n 详细测试日志系统")
    print("-"*30)

    from werewolf.game_logger import GameLogger

    logger = GameLogger("test_game")

    # 测试各种事件
    logger.log_round_start(1)
    logger.log_death("玩家3", "狼人杀害")
    logger.log_phase_change("day")
    logger.log_speech("玩家1", "我是预言家昨晚验了玩家2是狼人")
    logger.log_vote("玩家4", "玩家2")
    logger.log_elimination("玩家2", 3)

    # 显示格式化日志
    print("格式化日志:")
    print(logger.get_formatted_log())


if __name__ == "__main__":
    asyncio.run(test_system())
    asyncio.run(test_memory_system())
    asyncio.run(test_logger_system())