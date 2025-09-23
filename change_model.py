"""
批量更换AI Bot模型的通用工具
用法: python change_model.py <模型名>
例如: python change_model.py Qwen/Qwen3-Next-80B-A3B-Instruct
"""

import sys
import json
import os
from typing import Optional

# 添加当前路径到Python路径
sys.path.append('.')

try:
    from api_config import ConfigManager, APIProvider
except ImportError as e:
    print(f"Error importing api_config: {e}")
    print("Please make sure you're running this from the werewolf directory")
    sys.exit(1)


class ModelChanger:
    """模型批量更换工具"""

    def __init__(self):
        self.config_manager = ConfigManager()

    def change_all_models(self, new_model: str, provider: str = "siliconflow") -> int:
        """批量更换所有配置的模型"""

        print(f"=== Batch Model Changer ===")
        print(f"Target model: {new_model}")
        print(f"Provider: {provider}")
        print("-" * 50)

        # 获取所有配置
        configs = self.config_manager.list_configs()

        if not configs:
            print("No configurations found!")
            return 0

        print(f"Found {len(configs)} configurations")

        # 确定提供商
        try:
            if provider.lower() == "openai":
                target_provider = APIProvider.OPENAI
                base_url = "https://api.openai.com/v1"
            elif provider.lower() == "siliconflow":
                target_provider = APIProvider.SILICONFLOW
                base_url = "https://api.siliconflow.cn/v1"
            elif provider.lower() == "anthropic":
                target_provider = APIProvider.ANTHROPIC
                base_url = "https://api.anthropic.com/v1"
            else:
                target_provider = APIProvider.SILICONFLOW  # 默认使用SiliconFlow
                base_url = "https://api.siliconflow.cn/v1"
        except Exception as e:
            print(f"Error setting provider: {e}")
            return 0

        updated_count = 0

        # 批量更新每个配置
        for config in configs:
            old_model = config.model
            old_provider = config.provider.value

            # 更新模型
            config.model = new_model

            # 更新提供商和URL（如果需要）
            if config.provider != target_provider:
                config.provider = target_provider
                config.base_url = base_url
                print(f"[{config.name}] {old_model} ({old_provider}) -> {new_model} ({provider})")
            else:
                print(f"[{config.name}] {old_model} -> {new_model}")

            updated_count += 1

        # 保存所有更改
        try:
            self.config_manager.save_configs()
            print("-" * 50)
            print(f"SUCCESS: Updated {updated_count} configurations!")
            print(f"All bots now use: {new_model}")
            return updated_count

        except Exception as e:
            print(f"ERROR: Failed to save configurations: {e}")
            return 0

    def show_current_status(self):
        """显示当前配置状态"""
        configs = self.config_manager.list_configs()

        if not configs:
            print("No configurations found!")
            return

        print(f"\n=== Current Configuration Status ===")
        print(f"Total configurations: {len(configs)}")

        # 按模型统计
        model_stats = {}
        provider_stats = {}

        for config in configs:
            model = config.model
            provider = config.provider.value

            model_stats[model] = model_stats.get(model, 0) + 1
            provider_stats[provider] = provider_stats.get(provider, 0) + 1

        print("\nBy Model:")
        for model, count in sorted(model_stats.items()):
            print(f"  {model}: {count}")

        print("\nBy Provider:")
        for provider, count in sorted(provider_stats.items()):
            print(f"  {provider}: {count}")

    def list_popular_models(self):
        """显示常用模型列表"""
        popular_models = {
            "SiliconFlow Models": [
                "Qwen/Qwen3-Next-80B-A3B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "deepseek-ai/DeepSeek-V3.1",
                "deepseek-ai/DeepSeek-R1",
                "inclusionAI/Ring-flash-2.0",
                "meta-llama/Llama-3.1-70B-Instruct",
                "01-ai/Yi-1.5-34B-Chat"
            ],
            "OpenAI Models": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ],
            "Anthropic Models": [
                "claude-3-5-sonnet-20241022",
                "claude-3-haiku-20240307",
                "claude-3-opus-20240229"
            ]
        }

        print("\n=== Popular Models ===")
        for category, models in popular_models.items():
            print(f"\n{category}:")
            for model in models:
                print(f"  {model}")


def show_usage():
    """显示使用帮助"""
    print("=== Model Changer Tool ===")
    print("Usage:")
    print("  python change_model.py <model_name> [provider]")
    print("")
    print("Examples:")
    print("  python change_model.py Qwen/Qwen3-Next-80B-A3B-Instruct")
    print("  python change_model.py gpt-4o-mini openai")
    print("  python change_model.py claude-3-5-sonnet-20241022 anthropic")
    print("")
    print("Special commands:")
    print("  python change_model.py --status     (show current status)")
    print("  python change_model.py --list       (list popular models)")
    print("  python change_model.py --help       (show this help)")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        show_usage()
        return

    command = sys.argv[1].lower()

    # 特殊命令
    if command in ['--help', '-h', 'help']:
        show_usage()
        return

    changer = ModelChanger()

    if command in ['--status', 'status']:
        changer.show_current_status()
        return

    if command in ['--list', 'list']:
        changer.list_popular_models()
        return

    # 正常的模型更换
    new_model = sys.argv[1]
    provider = sys.argv[2] if len(sys.argv) > 2 else "siliconflow"

    # 验证模型名不为空
    if not new_model or new_model.startswith('-'):
        print("Error: Please provide a valid model name")
        show_usage()
        return

    # 执行更换
    try:
        updated = changer.change_all_models(new_model, provider)

        if updated > 0:
            print("\n=== Verification ===")
            changer.show_current_status()
        else:
            print("No configurations were updated.")

    except Exception as e:
        print(f"Error during model change: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()