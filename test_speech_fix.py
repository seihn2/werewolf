"""
测试AI发言修复
"""

import asyncio
import sys
sys.path.append('.')

from werewolf.enhanced_game_controller import EnhancedWerewolfGameController
from werewolf.llm_manager import LLMConfig, APIProvider

async def test_speech_generation():
    print("=== 测试AI发言修复 ===")

    # 创建6个玩家的测试游戏
    test_configs = []
    for i in range(6):
        config = LLMConfig(
            name=f'test_config_{i}',
            provider=APIProvider.OPENAI,
            api_key='test-key',
            model='gpt-4o-mini'
        )
        test_configs.append({
            'name': f'测试玩家{i+1}',
            'llm_config': config
        })

    controller = EnhancedWerewolfGameController(test_configs)

    if await controller.initialize_game():
        print("✓ 游戏初始化成功")
        print("✓ 角色分配:")
        for player in controller.players:
            print(f"  {player.name}: {player.role.value}")

        # 测试一轮发言生成
        print("\n=== 测试发言生成 ===")
        test_player = controller.players[0]
        memory = controller.memory_manager.get_bot_memory(test_player.id)

        if memory and controller.round_manager:
            speech = await controller.round_manager._generate_speech(test_player, memory)
            print(f"生成的发言: {speech}")
            print(f"发言长度: {len(speech)}字符")

            # 检查是否是增强版发言（不是原来的短发言）
            old_boring_speeches = [
                "我怀疑预言家可能是假的",
                "大家要相信我的判断",
                "我是村民，我们要找出狼人",
                "我觉得某些人的发言很可疑",
                "我支持预言家的判断"
            ]

            if speech in old_boring_speeches:
                print("❌ 仍在使用旧的无聊发言")
            else:
                print("✅ 使用了增强版发言内容")
        else:
            print("❌ 无法获取玩家记忆或轮次管理器")

        return True
    else:
        print("❌ 游戏初始化失败")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_speech_generation())
    print(f"\n测试结果: {'成功' if result else '失败'}")