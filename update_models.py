"""
批量更新所有bot配置的模型为Ring-flash-2.0
"""

import json
import os
import sys
sys.path.append('.')

from api_config import ConfigManager


def update_all_models_to_ring_flash():
    """将所有bot配置的模型更新为Ring-flash-2.0"""

    print("=== 批量更新bot模型为Ring-flash-2.0 ===")

    # 初始化配置管理器
    config_manager = ConfigManager()
    configs = config_manager.list_configs()

    print(f"找到 {len(configs)} 个配置")

    updated_count = 0

    for config in configs:
        old_model = config.model

        # 更新模型名称
        config.model = "inclusionAI/Ring-flash-2.0"

        # 确保使用SiliconFlow提供商（因为Ring-flash-2.0在SiliconFlow上）
        if config.provider.value != "siliconflow":
            from api_config import APIProvider
            config.provider = APIProvider.SILICONFLOW
            config.base_url = "https://api.siliconflow.cn/v1"
            print(f"✓ {config.name}: {old_model} -> {config.model} (同时更新为SiliconFlow)")
        else:
            print(f"✓ {config.name}: {old_model} -> {config.model}")

        updated_count += 1

    # 保存所有更新
    config_manager.save_configs()

    print(f"\n🎉 成功更新了 {updated_count} 个配置！")
    print("所有bot现在都使用 inclusionAI/Ring-flash-2.0 模型")

    return updated_count


def show_updated_configs():
    """显示更新后的配置"""
    config_manager = ConfigManager()
    configs = config_manager.list_configs()

    print("\n=== 更新后的配置列表 ===")

    model_count = {}
    provider_count = {}

    for config in configs:
        print(f"- {config.name}")
        print(f"  模型: {config.model}")
        print(f"  提供商: {config.provider.value}")
        print(f"  描述: {config.description}")
        print()

        # 统计
        model_count[config.model] = model_count.get(config.model, 0) + 1
        provider_count[config.provider.value] = provider_count.get(config.provider.value, 0) + 1

    print("=== 统计信息 ===")
    print("按模型统计:")
    for model, count in model_count.items():
        print(f"  {model}: {count}")

    print("按提供商统计:")
    for provider, count in provider_count.items():
        print(f"  {provider}: {count}")


if __name__ == "__main__":
    try:
        # 更新所有配置
        updated = update_all_models_to_ring_flash()

        if updated > 0:
            # 显示更新后的结果
            show_updated_configs()
        else:
            print("没有找到需要更新的配置")

    except Exception as e:
        print(f"更新失败: {e}")
        import traceback
        traceback.print_exc()