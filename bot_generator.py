"""
批量生产AI Bot配置工具
快速生成大量AI玩家配置，支持JSON批量导入导出
"""

import json
import os
from typing import List, Dict, Any
from api_config import APIConfig, APIProvider, ConfigManager


class BotGenerator:
    """AI Bot批量生产工具"""

    def __init__(self):
        self.config_manager = ConfigManager()

    def generate_template(self, output_file: str = "bot_templates.json"):
        """生成配置模板文件"""
        templates = {
            "templates": {
                "openai_template": {
                    "provider": "openai",
                    "api_key": "sk-your-openai-api-key-here",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini",
                    "description": "OpenAI GPT-4 模型"
                },
                "siliconflow_template": {
                    "provider": "siliconflow",
                    "api_key": "sk-your-siliconflow-api-key-here",
                    "base_url": "https://api.siliconflow.cn/v1",
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "description": "SiliconFlow Qwen模型"
                },
                "custom_template": {
                    "provider": "custom",
                    "api_key": "your-api-key",
                    "base_url": "https://your-api-endpoint.com/v1",
                    "model": "your-model-name",
                    "description": "自定义API模型"
                }
            },
            "bot_configs": [
                {
                    "name": "AI玩家1",
                    "provider": "openai",
                    "api_key": "sk-your-openai-api-key-here",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini",
                    "description": "狼人杀AI玩家1"
                },
                {
                    "name": "AI玩家2",
                    "provider": "siliconflow",
                    "api_key": "sk-your-siliconflow-api-key-here",
                    "base_url": "https://api.siliconflow.cn/v1",
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "description": "狼人杀AI玩家2"
                }
            ],
            "usage": {
                "description": "批量生产AI Bot配置工具使用说明",
                "steps": [
                    "1. 修改上面的api_key为你的真实API密钥",
                    "2. 根据需要调整模型和参数",
                    "3. 运行: python bot_generator.py import bot_templates.json",
                    "4. 或使用批量生成功能快速创建多个bot"
                ]
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)

        print(f"✅ 模板文件已生成: {output_file}")
        print("📝 请编辑模板文件中的API密钥，然后使用导入功能")

    def batch_generate(self, base_template: Dict[str, Any], count: int, name_prefix: str = "AI玩家") -> List[Dict[str, Any]]:
        """批量生成bot配置"""
        bots = []
        for i in range(1, count + 1):
            bot = base_template.copy()
            bot["name"] = f"{name_prefix}{i}"
            bot["description"] = f"批量生成的{name_prefix}{i}"
            bots.append(bot)

        return bots

    def import_from_json(self, json_file: str):
        """从JSON文件导入配置"""
        if not os.path.exists(json_file):
            print(f"❌ 文件不存在: {json_file}")
            return

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            imported_count = 0

            # 导入bot_configs
            if "bot_configs" in data:
                for bot_config in data["bot_configs"]:
                    try:
                        config = APIConfig(
                            name=bot_config["name"],
                            provider=APIProvider(bot_config["provider"]),
                            api_key=bot_config["api_key"],
                            base_url=bot_config["base_url"],
                            model=bot_config["model"],
                            description=bot_config.get("description", ""),
                            is_default=bot_config.get("is_default", False)
                        )

                        self.config_manager.add_config(config)
                        imported_count += 1
                        print(f"✅ 导入成功: {config.name}")

                    except Exception as e:
                        print(f"❌ 导入失败 {bot_config.get('name', 'unknown')}: {e}")

            self.config_manager.save_configs()
            print(f"\n🎉 成功导入 {imported_count} 个bot配置")

        except Exception as e:
            print(f"❌ 读取JSON文件失败: {e}")

    def export_configs(self, output_file: str = "exported_bots.json"):
        """导出当前所有配置"""
        configs = self.config_manager.list_configs()

        export_data = {
            "export_time": str(datetime.now()),
            "total_configs": len(configs),
            "bot_configs": []
        }

        for config in configs:
            export_data["bot_configs"].append({
                "name": config.name,
                "provider": config.provider.value,
                "api_key": config.api_key,
                "base_url": config.base_url,
                "model": config.model,
                "description": config.description,
                "is_default": config.is_default
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 已导出 {len(configs)} 个配置到: {output_file}")

    def quick_batch_create(self):
        """交互式快速批量创建"""
        print("\n🤖 快速批量创建AI Bot")
        print("=" * 50)

        # 选择模板类型
        print("选择API提供商模板:")
        print("1. OpenAI (推荐用于高质量对话)")
        print("2. SiliconFlow (国内访问快，多模型选择)")
        print("3. 自定义API")

        template_choice = input("请选择模板 (1-3): ").strip()

        if template_choice == "1":
            provider = APIProvider.OPENAI
            base_url = "https://api.openai.com/v1"
            model = "gpt-4o-mini"
        elif template_choice == "2":
            provider = APIProvider.SILICONFLOW
            base_url = "https://api.siliconflow.cn/v1"
            model = "Qwen/Qwen2.5-7B-Instruct"
        elif template_choice == "3":
            provider = APIProvider.CUSTOM
            base_url = input("请输入API基础URL: ").strip()
            model = input("请输入模型名称: ").strip()
        else:
            print("❌ 无效选择")
            return

        # 获取API密钥
        api_key = input("请输入API密钥: ").strip()
        if not api_key:
            print("❌ API密钥不能为空")
            return

        # 获取生成数量
        try:
            count = int(input("请输入要生成的Bot数量 (建议6-12个): "))
            if count < 1 or count > 50:
                print("❌ 数量应在1-50之间")
                return
        except ValueError:
            print("❌ 请输入有效数字")
            return

        # 生成配置
        print(f"\n🔄 正在生成 {count} 个Bot配置...")

        created_count = 0
        for i in range(1, count + 1):
            try:
                config = APIConfig(
                    name=f"AI玩家{i}",
                    provider=provider,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    description=f"批量生成的AI玩家{i}",
                    is_default=(i == 1)  # 第一个设为默认
                )

                self.config_manager.add_config(config)
                created_count += 1
                print(f"✅ 创建成功: AI玩家{i}")

            except Exception as e:
                print(f"❌ 创建失败 AI玩家{i}: {e}")

        self.config_manager.save_configs()
        print(f"\n🎉 成功创建 {created_count} 个Bot配置！")
        print(f"📁 配置已保存到: {self.config_manager.config_file}")

    def show_stats(self):
        """显示当前配置统计"""
        configs = self.config_manager.list_configs()

        print(f"\n📊 当前配置统计")
        print("=" * 40)
        print(f"总配置数: {len(configs)}")

        # 按提供商统计
        provider_stats = {}
        valid_configs = 0

        for config in configs:
            provider = config.provider.value
            provider_stats[provider] = provider_stats.get(provider, 0) + 1

            # 检查是否为有效配置
            if config.api_key and config.api_key not in ["your-openai-api-key", "your-siliconflow-api-key"]:
                valid_configs += 1

        print(f"有效配置: {valid_configs}")
        print(f"待配置: {len(configs) - valid_configs}")

        print("\n按提供商分布:")
        for provider, count in provider_stats.items():
            print(f"  {provider}: {count}")


def main():
    """主函数"""
    import sys
    from datetime import datetime

    generator = BotGenerator()

    if len(sys.argv) < 2:
        print("🤖 AI Bot批量生产工具")
        print("=" * 50)
        print("使用方法:")
        print("  python bot_generator.py template                # 生成配置模板")
        print("  python bot_generator.py import <json_file>      # 导入配置")
        print("  python bot_generator.py export [output_file]    # 导出配置")
        print("  python bot_generator.py create                  # 交互式批量创建")
        print("  python bot_generator.py stats                   # 显示统计信息")
        return

    command = sys.argv[1]

    if command == "template":
        output = sys.argv[2] if len(sys.argv) > 2 else "bot_templates.json"
        generator.generate_template(output)

    elif command == "import":
        if len(sys.argv) < 3:
            print("❌ 请指定要导入的JSON文件")
            return
        generator.import_from_json(sys.argv[2])

    elif command == "export":
        output = sys.argv[2] if len(sys.argv) > 2 else "exported_bots.json"
        generator.export_configs(output)

    elif command == "create":
        generator.quick_batch_create()

    elif command == "stats":
        generator.show_stats()

    else:
        print(f"❌ 未知命令: {command}")


if __name__ == "__main__":
    main()