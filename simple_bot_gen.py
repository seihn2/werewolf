"""
Simple Bot Generator - 简单的AI Bot批量生产工具
避免Unicode显示问题，专注功能实现
"""

import json
import os
import sys
sys.path.append('.')

from api_config import APIConfig, APIProvider, ConfigManager


def generate_template():
    """生成配置模板"""
    template = {
        "usage": "Edit api_key values below, then run: python simple_bot_gen.py import bot_template.json",
        "bot_configs": [
            {
                "name": "AI玩家1",
                "provider": "openai",
                "api_key": "sk-your-openai-key-here",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "description": "OpenAI AI玩家"
            },
            {
                "name": "AI玩家2",
                "provider": "siliconflow",
                "api_key": "sk-your-siliconflow-key-here",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "description": "SiliconFlow AI玩家"
            },
            {
                "name": "AI玩家3",
                "provider": "openai",
                "api_key": "sk-your-openai-key-here",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "description": "OpenAI AI玩家"
            },
            {
                "name": "AI玩家4",
                "provider": "siliconflow",
                "api_key": "sk-your-siliconflow-key-here",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-V2.5",
                "description": "SiliconFlow DeepSeek模型"
            },
            {
                "name": "AI玩家5",
                "provider": "openai",
                "api_key": "sk-your-openai-key-here",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "description": "OpenAI AI玩家"
            },
            {
                "name": "AI玩家6",
                "provider": "siliconflow",
                "api_key": "sk-your-siliconflow-key-here",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "description": "SiliconFlow AI玩家"
            },
            {
                "name": "AI玩家7",
                "provider": "openai",
                "api_key": "sk-your-openai-key-here",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "description": "OpenAI AI玩家"
            },
            {
                "name": "AI玩家8",
                "provider": "siliconflow",
                "api_key": "sk-your-siliconflow-key-here",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-V2.5",
                "description": "SiliconFlow DeepSeek模型"
            },
            {
                "name": "AI玩家9",
                "provider": "openai",
                "api_key": "sk-your-openai-key-here",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "description": "OpenAI AI玩家"
            },
            {
                "name": "AI玩家10",
                "provider": "siliconflow",
                "api_key": "sk-your-siliconflow-key-here",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "description": "SiliconFlow AI玩家"
            }
        ]
    }

    with open('bot_template.json', 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print("Template generated: bot_template.json")
    print("Edit the api_key values, then run: python simple_bot_gen.py import bot_template.json")


def import_configs(json_file):
    """导入配置"""
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
        return

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config_manager = ConfigManager()
        imported = 0

        for bot_config in data.get('bot_configs', []):
            try:
                config = APIConfig(
                    name=bot_config['name'],
                    provider=APIProvider(bot_config['provider']),
                    api_key=bot_config['api_key'],
                    base_url=bot_config['base_url'],
                    model=bot_config['model'],
                    description=bot_config.get('description', ''),
                    is_default=False
                )

                config_manager.add_config(config)
                imported += 1
                print(f"Imported: {config.name}")

            except Exception as e:
                print(f"Failed to import {bot_config.get('name', 'unknown')}: {e}")

        config_manager.save_configs()
        print(f"Successfully imported {imported} bot configs")

    except Exception as e:
        print(f"Import failed: {e}")


def batch_create():
    """快速批量创建"""
    print("Quick Batch Bot Creator")
    print("=" * 40)

    provider = input("Choose provider (openai/siliconflow): ").strip().lower()
    if provider not in ['openai', 'siliconflow']:
        print("Invalid provider")
        return

    api_key = input("Enter API key: ").strip()
    if not api_key:
        print("API key required")
        return

    try:
        count = int(input("Number of bots to create (1-20): "))
        if count < 1 or count > 20:
            print("Count must be 1-20")
            return
    except ValueError:
        print("Invalid number")
        return

    # Set defaults based on provider
    if provider == 'openai':
        base_url = "https://api.openai.com/v1"
        model = "gpt-4o-mini"
    else:  # siliconflow
        base_url = "https://api.siliconflow.cn/v1"
        model = "Qwen/Qwen2.5-7B-Instruct"

    config_manager = ConfigManager()
    created = 0

    for i in range(1, count + 1):
        try:
            config = APIConfig(
                name=f"AI玩家{i}",
                provider=APIProvider(provider),
                api_key=api_key,
                base_url=base_url,
                model=model,
                description=f"Batch created AI Player {i}",
                is_default=False
            )

            config_manager.add_config(config)
            created += 1
            print(f"Created: AI玩家{i}")

        except Exception as e:
            print(f"Failed to create AI玩家{i}: {e}")

    config_manager.save_configs()
    print(f"Successfully created {created} bots")


def show_stats():
    """显示统计"""
    config_manager = ConfigManager()
    configs = config_manager.list_configs()

    print(f"Total configs: {len(configs)}")

    # Count by provider
    stats = {}
    valid = 0

    for config in configs:
        provider = config.provider.value
        stats[provider] = stats.get(provider, 0) + 1

        # Check if valid (not default placeholder)
        if config.api_key and 'your-' not in config.api_key:
            valid += 1

    print(f"Valid configs: {valid}")
    print("By provider:")
    for provider, count in stats.items():
        print(f"  {provider}: {count}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Simple Bot Generator")
        print("Usage:")
        print("  python simple_bot_gen.py template       # Generate template")
        print("  python simple_bot_gen.py import <file>  # Import configs")
        print("  python simple_bot_gen.py create         # Interactive create")
        print("  python simple_bot_gen.py stats          # Show statistics")
        return

    command = sys.argv[1]

    if command == 'template':
        generate_template()
    elif command == 'import':
        if len(sys.argv) < 3:
            print("Specify JSON file to import")
            return
        import_configs(sys.argv[2])
    elif command == 'create':
        batch_create()
    elif command == 'stats':
        show_stats()
    else:
        print(f"Unknown command: {command}")


if __name__ == '__main__':
    main()