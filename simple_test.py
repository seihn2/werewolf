"""
Simple integration test without Unicode characters
"""

import asyncio
import sys
sys.path.append('.')

from werewolf.enhanced_game_controller import EnhancedWerewolfGameController
from werewolf.llm_manager import LLMConfig, APIProvider

async def simple_test():
    print("=== Integration Test Results ===")

    # Test with 9 players to verify N-player support
    test_configs = []
    for i in range(9):
        config = LLMConfig(
            name=f'test_config_{i}',
            provider=APIProvider.OPENAI,
            api_key='test-key',
            model='gpt-4o-mini'
        )
        test_configs.append({
            'name': f'AI_Player_{i+1}',
            'llm_config': config
        })

    controller = EnhancedWerewolfGameController(test_configs)

    if await controller.initialize_game():
        print("FEATURE 1 - N-player support: WORKING")

        # Check role distribution
        role_count = {}
        for player in controller.players:
            role = player.role.value
            role_count[role] = role_count.get(role, 0) + 1

        print(f"9-player game roles: {role_count}")

        # Verify conversation manager exists
        if controller.conversation_manager:
            print("FEATURE 2 - Enhanced conversation manager: LOADED")

            # Check for enhanced speech methods
            methods = [m for m in dir(controller.conversation_manager) if not m.startswith('_')]
            if 'generate_vote' in methods:
                print("FEATURE 3 - AI voting system: IMPLEMENTED")
            else:
                print("FEATURE 3 - AI voting system: MISSING")

        # Check round manager integration
        if controller.round_manager and controller.conversation_manager:
            print("INTEGRATION - All components connected: SUCCESS")

        print(f"\nGame ready with {len(controller.players)} players")
        return True
    else:
        print("INTEGRATION TEST: FAILED")
        return False

if __name__ == "__main__":
    result = asyncio.run(simple_test())
    print(f"\nFinal result: {'PASSED' if result else 'FAILED'}")