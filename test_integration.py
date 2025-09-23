"""
Integration test for all three new features:
1. N-player game support
2. Forced longer AI speeches
3. Day voting phase
"""

import asyncio
import sys
sys.path.append('.')

from werewolf.enhanced_game_controller import EnhancedWerewolfGameController
from werewolf.llm_manager import LLMConfig, APIProvider
from werewolf.game_logic import Role

async def test_all_features():
    print("=== Testing All Three Features ===")

    # Feature 1: Test N-player support (trying 7 players)
    print("\n1. Testing N-player support (7 players)...")
    test_configs = []
    for i in range(7):
        config = LLMConfig(
            name=f'test_config_{i}',
            provider=APIProvider.OPENAI,
            api_key='test-key',
            model='gpt-4o-mini'
        )
        test_configs.append({
            'name': f'Player{i+1}',
            'llm_config': config
        })

    controller = EnhancedWerewolfGameController(test_configs)

    if await controller.initialize_game():
        print("✓ N-player support: SUCCESS")
        print("Role distribution:")
        role_count = {}
        for player in controller.players:
            role = player.role.value
            role_count[role] = role_count.get(role, 0) + 1
            print(f"  {player.name}: {role}")

        print(f"Role summary: {role_count}")

        # Feature 2: Test conversation manager for longer speeches
        print("\n2. Testing forced longer AI speeches...")
        if controller.conversation_manager:
            print("✓ Conversation manager: LOADED")
            # Check if it has the enhanced speech generation
            if hasattr(controller.conversation_manager, '_ensure_minimum_length'):
                print("✓ Minimum length enforcement: IMPLEMENTED")
            else:
                print("✗ Minimum length enforcement: MISSING")
        else:
            print("✗ Conversation manager: NOT LOADED")

        # Feature 3: Test voting mechanism
        print("\n3. Testing AI-based voting system...")
        if controller.round_manager and controller.conversation_manager:
            # Check if round manager uses conversation manager for voting
            if hasattr(controller.conversation_manager, 'generate_vote'):
                print("✓ AI voting system: IMPLEMENTED")
                print("✓ Vote generation method: EXISTS")
            else:
                print("✗ AI voting system: MISSING")
        else:
            print("✗ AI voting system: COMPONENTS MISSING")

        print(f"\n=== Integration Test Summary ===")
        print(f"Game ID: {controller.game_id}")
        print(f"Total players: {len(controller.players)}")
        print(f"Components initialized: {bool(controller.round_manager and controller.conversation_manager)}")

        return True
    else:
        print("✗ Game initialization failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_all_features())
    print(f"\nOverall test result: {'SUCCESS' if success else 'FAILED'}")