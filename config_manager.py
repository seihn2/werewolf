"""
狼人杀配置管理工具
用于管理多个LLM API配置和游戏设置
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from .llm_manager import MultiLLMManager, LLMConfig, APIProvider, create_quick_config


class WerewolfConfigManager:
    """狼人杀配置管理器"""

    def __init__(self):
        self.llm_manager = MultiLLMManager()

    def interactive_setup(self):
        """交互式设置"""
        print("=" * 60)
        print("🐺 AI狼人杀 - 配置管理器")
        print("=" * 60)

        while True:
            self._show_main_menu()
            choice = input("\n请选择操作 (1-7): ").strip()

            if choice == "1":
                self._show_configs()
            elif choice == "2":
                self._add_config()
            elif choice == "3":
                self._edit_config()
            elif choice == "4":
                self._test_configs()
            elif choice == "5":
                self._create_game_setup()
            elif choice == "6":
                self._import_export_configs()
            elif choice == "7":
                print("退出配置管理器")
                break
            else:
                print("无效选择，请重试")

    def _show_main_menu(self):
        """显示主菜单"""
        print("\n📋 主菜单:")
        print("1. 查看所有配置")
        print("2. 添加新配置")
        print("3. 编辑配置")
        print("4. 测试配置")
        print("5. 创建游戏配置")
        print("6. 导入/导出配置")
        print("7. 退出")

    def _show_configs(self):
        """显示所有配置"""
        configs = self.llm_manager.list_configs()

        if not configs:
            print("\n❌ 没有找到任何配置")
            return

        print(f"\n📂 当前配置 (共{len(configs)}个):")
        print("-" * 80)

        for name, config in configs.items():
            status = "✅ 启用" if config.enabled else "❌ 禁用"
            print(f"🔧 {name}")
            print(f"   提供商: {config.provider.value}")
            print(f"   模型: {config.model}")
            print(f"   状态: {status}")
            print(f"   描述: {config.description or '无描述'}")
            print("-" * 40)

        # 显示统计信息
        stats = self.llm_manager.get_stats()
        print(f"\n📊 统计信息:")
        print(f"   总配置: {stats['total_configs']}")
        print(f"   启用: {stats['enabled_configs']}")
        print(f"   禁用: {stats['disabled_configs']}")

    def _add_config(self):
        """添加新配置"""
        print("\n➕ 添加新配置")
        print("-" * 30)

        # 基本信息
        name = input("配置名称: ").strip()
        if not name:
            print("❌ 名称不能为空")
            return

        if self.llm_manager.get_config(name):
            print(f"❌ 配置 '{name}' 已存在")
            return

        # 选择提供商
        print("\n🏢 选择API提供商:")
        providers = list(APIProvider)
        for i, provider in enumerate(providers, 1):
            print(f"{i}. {provider.value}")

        try:
            provider_choice = int(input("选择提供商 (1-{}): ".format(len(providers))))
            if 1 <= provider_choice <= len(providers):
                provider = providers[provider_choice - 1]
            else:
                print("❌ 无效选择")
                return
        except ValueError:
            print("❌ 请输入数字")
            return

        # API密钥
        api_key = input("API密钥: ").strip()
        if not api_key:
            print("❌ API密钥不能为空")
            return

        # 模型名称
        model = input("模型名称: ").strip()
        if not model:
            print("❌ 模型名称不能为空")
            return

        # 可选信息
        description = input("描述 (可选): ").strip()
        base_url = input("自定义API地址 (可选，回车使用默认): ").strip()

        try:
            # 创建配置
            config = LLMConfig(
                name=name,
                provider=provider,
                api_key=api_key,
                model=model,
                description=description,
                base_url=base_url if base_url else None
            )

            if self.llm_manager.add_config(config):
                print(f"✅ 配置 '{name}' 添加成功")

                # 询问是否立即测试
                test_now = input("是否立即测试配置? (y/n): ").strip().lower()
                if test_now == 'y':
                    asyncio.run(self._test_single_config(name))
            else:
                print(f"❌ 配置 '{name}' 添加失败")

        except Exception as e:
            print(f"❌ 创建配置失败: {e}")

    def _edit_config(self):
        """编辑配置"""
        configs = self.llm_manager.list_configs()
        if not configs:
            print("\n❌ 没有可编辑的配置")
            return

        print("\n✏️ 编辑配置")
        print("-" * 20)

        # 显示配置列表
        config_names = list(configs.keys())
        for i, name in enumerate(config_names, 1):
            status = "启用" if configs[name].enabled else "禁用"
            print(f"{i}. {name} ({status})")

        try:
            choice = int(input(f"选择要编辑的配置 (1-{len(config_names)}): "))
            if 1 <= choice <= len(config_names):
                config_name = config_names[choice - 1]
                self._edit_single_config(config_name)
            else:
                print("❌ 无效选择")
        except ValueError:
            print("❌ 请输入数字")

    def _edit_single_config(self, config_name: str):
        """编辑单个配置"""
        config = self.llm_manager.get_config(config_name)
        if not config:
            print(f"❌ 配置 '{config_name}' 不存在")
            return

        print(f"\n编辑配置: {config_name}")
        print("-" * 30)

        while True:
            print(f"\n当前配置:")
            print(f"1. API密钥: {config.api_key[:10]}***")
            print(f"2. 模型: {config.model}")
            print(f"3. 描述: {config.description}")
            print(f"4. 状态: {'启用' if config.enabled else '禁用'}")
            print(f"5. 保存并退出")
            print(f"6. 取消编辑")

            choice = input("选择要修改的项目 (1-6): ").strip()

            if choice == "1":
                new_key = input(f"新的API密钥 (当前: {config.api_key[:10]}***): ").strip()
                if new_key:
                    config.api_key = new_key
                    print("✅ API密钥已更新")

            elif choice == "2":
                new_model = input(f"新的模型名称 (当前: {config.model}): ").strip()
                if new_model:
                    config.model = new_model
                    print("✅ 模型已更新")

            elif choice == "3":
                new_desc = input(f"新的描述 (当前: {config.description}): ").strip()
                config.description = new_desc
                print("✅ 描述已更新")

            elif choice == "4":
                config.enabled = not config.enabled
                status = "启用" if config.enabled else "禁用"
                print(f"✅ 状态已切换为: {status}")

            elif choice == "5":
                if self.llm_manager.update_config(config_name, config):
                    print(f"✅ 配置 '{config_name}' 保存成功")
                else:
                    print(f"❌ 配置 '{config_name}' 保存失败")
                break

            elif choice == "6":
                print("❌ 编辑已取消")
                break

            else:
                print("❌ 无效选择")

    def _test_configs(self):
        """测试配置"""
        enabled_configs = self.llm_manager.get_enabled_configs()

        if not enabled_configs:
            print("\n❌ 没有启用的配置可测试")
            return

        print(f"\n🧪 测试配置 (共{len(enabled_configs)}个启用的配置)")
        print("-" * 40)

        print("1. 测试所有启用的配置")
        print("2. 测试单个配置")

        choice = input("选择测试方式 (1-2): ").strip()

        if choice == "1":
            asyncio.run(self._test_all_configs())
        elif choice == "2":
            self._test_single_config_menu()
        else:
            print("❌ 无效选择")

    async def _test_all_configs(self):
        """测试所有配置"""
        print("\n🔄 正在测试所有启用的配置...")

        results = await self.llm_manager.test_all_enabled_configs()

        print(f"\n📊 测试结果:")
        print("-" * 30)

        for config_name, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            print(f"{config_name}: {status}")

        success_count = sum(results.values())
        total_count = len(results)
        print(f"\n📈 总体结果: {success_count}/{total_count} 配置可用")

    def _test_single_config_menu(self):
        """测试单个配置菜单"""
        configs = list(self.llm_manager.list_configs().keys())

        print("\n选择要测试的配置:")
        for i, name in enumerate(configs, 1):
            print(f"{i}. {name}")

        try:
            choice = int(input(f"选择配置 (1-{len(configs)}): "))
            if 1 <= choice <= len(configs):
                config_name = configs[choice - 1]
                asyncio.run(self._test_single_config(config_name))
            else:
                print("❌ 无效选择")
        except ValueError:
            print("❌ 请输入数字")

    async def _test_single_config(self, config_name: str):
        """测试单个配置"""
        print(f"\n🔄 正在测试配置: {config_name}")
        success = await self.llm_manager.test_config(config_name)

        if success:
            print(f"✅ 配置 '{config_name}' 测试成功")
        else:
            print(f"❌ 配置 '{config_name}' 测试失败")

    def _create_game_setup(self):
        """创建游戏配置"""
        enabled_configs = self.llm_manager.get_enabled_configs()

        if len(enabled_configs) < 6:
            print(f"\n❌ 需要至少6个启用的配置才能创建游戏，当前只有{len(enabled_configs)}个")
            print("请先添加并启用更多配置")
            return

        print(f"\n🎮 创建游戏配置")
        print("-" * 25)

        # 选择玩家数量
        max_players = min(12, len(enabled_configs))
        while True:
            try:
                num_players = int(input(f"玩家数量 (6-{max_players}): "))
                if 6 <= num_players <= max_players:
                    break
                else:
                    print(f"❌ 玩家数量必须在6-{max_players}之间")
            except ValueError:
                print("❌ 请输入数字")

        # 选择配置
        config_names = list(enabled_configs.keys())
        selected_configs = []

        print(f"\n选择{num_players}个配置:")
        for i in range(num_players):
            print(f"\n玩家 {i+1}:")
            for j, name in enumerate(config_names, 1):
                if name not in [c["llm_config"] for c in selected_configs]:
                    print(f"  {j}. {name}")

            while True:
                try:
                    choice = int(input(f"选择配置 (1-{len(config_names)}): "))
                    if 1 <= choice <= len(config_names):
                        config_name = config_names[choice - 1]
                        if config_name not in [c["llm_config"] for c in selected_configs]:
                            player_name = input(f"玩家名称 (默认: AI_{i+1}): ").strip()
                            if not player_name:
                                player_name = f"AI_{i+1}"

                            selected_configs.append({
                                "name": player_name,
                                "llm_config": config_name
                            })
                            break
                        else:
                            print("❌ 该配置已被选择")
                    else:
                        print("❌ 无效选择")
                except ValueError:
                    print("❌ 请输入数字")

        # 保存游戏配置
        game_config = {
            "name": input("\n游戏配置名称 (默认: default_game): ").strip() or "default_game",
            "players": selected_configs,
            "created_at": time.time()
        }

        # 保存到文件
        try:
            with open("werewolf/game_configs.json", "w", encoding="utf-8") as f:
                json.dump(game_config, f, ensure_ascii=False, indent=2)

            print(f"✅ 游戏配置已保存")
            print(f"\n🎯 配置预览:")
            for i, player in enumerate(selected_configs, 1):
                print(f"  {i}. {player['name']} (使用 {player['llm_config']})")

        except Exception as e:
            print(f"❌ 保存游戏配置失败: {e}")

    def _import_export_configs(self):
        """导入导出配置"""
        print(f"\n📁 导入/导出配置")
        print("-" * 20)
        print("1. 导出当前配置")
        print("2. 导入配置")

        choice = input("选择操作 (1-2): ").strip()

        if choice == "1":
            self._export_configs()
        elif choice == "2":
            self._import_configs()
        else:
            print("❌ 无效选择")

    def _export_configs(self):
        """导出配置"""
        configs = self.llm_manager.list_configs()
        if not configs:
            print("❌ 没有可导出的配置")
            return

        filename = input("导出文件名 (默认: werewolf_configs.json): ").strip()
        if not filename:
            filename = "werewolf_configs.json"

        try:
            export_data = {
                "version": "1.0",
                "exported_at": time.time(),
                "configs": {name: config.to_dict() for name, config in configs.items()}
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            print(f"✅ 配置已导出到 {filename}")
            print(f"📊 导出了 {len(configs)} 个配置")

        except Exception as e:
            print(f"❌ 导出失败: {e}")

    def _import_configs(self):
        """导入配置"""
        filename = input("导入文件名: ").strip()
        if not filename:
            print("❌ 文件名不能为空")
            return

        try:
            with open(filename, "r", encoding="utf-8") as f:
                import_data = json.load(f)

            configs_data = import_data.get("configs", {})
            imported_count = 0

            for name, config_dict in configs_data.items():
                try:
                    config = LLMConfig.from_dict(config_dict)
                    if self.llm_manager.add_config(config):
                        imported_count += 1
                        print(f"✅ 导入配置: {name}")
                    else:
                        print(f"⚠️ 配置已存在，跳过: {name}")
                except Exception as e:
                    print(f"❌ 导入配置失败 {name}: {e}")

            print(f"\n📊 导入完成: {imported_count} 个配置")

        except FileNotFoundError:
            print(f"❌ 文件不存在: {filename}")
        except Exception as e:
            print(f"❌ 导入失败: {e}")


def main():
    """主函数"""
    manager = WerewolfConfigManager()
    manager.interactive_setup()


if __name__ == "__main__":
    import time
    main()