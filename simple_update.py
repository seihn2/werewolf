"""
Simple bot model updater - avoid unicode issues
"""

import json
import os
import sys
sys.path.append('.')

from api_config import ConfigManager, APIProvider


def update_models():
    """Update all bot models to Ring-flash-2.0"""

    print("=== Updating all bot models to Ring-flash-2.0 ===")

    config_manager = ConfigManager()
    configs = config_manager.list_configs()

    print(f"Found {len(configs)} configurations")

    updated_count = 0

    for config in configs:
        old_model = config.model

        # Update model name
        config.model = "inclusionAI/Ring-flash-2.0"

        # Ensure using SiliconFlow provider
        if config.provider.value != "siliconflow":
            config.provider = APIProvider.SILICONFLOW
            config.base_url = "https://api.siliconflow.cn/v1"
            print(f"Updated {config.name}: {old_model} -> {config.model} (changed to SiliconFlow)")
        else:
            print(f"Updated {config.name}: {old_model} -> {config.model}")

        updated_count += 1

    # Save all updates
    config_manager.save_configs()

    print(f"\nSuccessfully updated {updated_count} configurations!")
    print("All bots now use inclusionAI/Ring-flash-2.0 model")

    return updated_count


def show_stats():
    """Show updated configuration statistics"""
    config_manager = ConfigManager()
    configs = config_manager.list_configs()

    print(f"\n=== Updated Configuration Stats ===")
    print(f"Total configs: {len(configs)}")

    model_count = {}
    provider_count = {}

    for config in configs:
        model_count[config.model] = model_count.get(config.model, 0) + 1
        provider_count[config.provider.value] = provider_count.get(config.provider.value, 0) + 1

    print("By model:")
    for model, count in model_count.items():
        print(f"  {model}: {count}")

    print("By provider:")
    for provider, count in provider_count.items():
        print(f"  {provider}: {count}")


if __name__ == "__main__":
    try:
        updated = update_models()
        if updated > 0:
            show_stats()
        else:
            print("No configurations found to update")

    except Exception as e:
        print(f"Update failed: {e}")
        import traceback
        traceback.print_exc()