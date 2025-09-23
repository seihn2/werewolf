"""
测试投票卡死问题修复
"""

import asyncio
import sys
sys.path.append('.')

from werewolf.enhanced_game_controller import EnhancedWerewolfGameController
from werewolf.llm_manager import LLMConfig, APIProvider

async def test_voting_timeout():
    print("=== 测试投票超时修复 ===")

    # 创建6个玩家的测试游戏
    configs = []
    for i in range(6):
        config = LLMConfig(
            name=f'test_{i}',
            provider=APIProvider.OPENAI,
            api_key='fake-api-key',  # 故意使用假密钥触发超时
            model='gpt-4o-mini'
        )
        configs.append({
            'name': f'玩家{i+1}',
            'llm_config': config
        })

    controller = EnhancedWerewolfGameController(configs)

    if await controller.initialize_game():
        print("✓ 游戏初始化成功")

        # 测试投票阶段，看是否会卡死
        print("\n开始测试投票阶段...")

        # 手动启动一轮游戏
        try:
            result = await asyncio.wait_for(
                controller.round_manager.start_new_round(),
                timeout=60.0  # 给整个轮次60秒时间
            )
            print(f"✓ 轮次完成，没有卡死!")
            print(f"轮次结果: {result.votes if result.votes else '无投票'}")
            print(f"淘汰玩家: {result.eliminations if result.eliminations else '无淘汰'}")
            return True

        except asyncio.TimeoutError:
            print("❌ 游戏仍然卡死了")
            return False

        except Exception as e:
            print(f"❌ 游戏出错: {e}")
            return False
    else:
        print("❌ 游戏初始化失败")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_voting_timeout())
    print(f"\n测试结果: {'通过' if result else '失败'}")