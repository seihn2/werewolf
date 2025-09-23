"""
AI狼人杀主程序 - 重构版
更清晰的菜单界面和配置管理
"""

import asyncio
import sys
import os
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werewolf.api_config import ConfigManager, ConfigUI, APIConfig
from werewolf.enhanced_game_controller import EnhancedWerewolfGameController
from werewolf.llm_manager import LLMConfig, APIProvider


class WerewolfGameLauncher:
    """狼人杀游戏启动器"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.config_ui = ConfigUI(self.config_manager)

    def show_welcome(self):
        """显示欢迎界面"""
        print("""
==================================================
              AI狼人杀 - 智能多模型对战平台
==================================================
        支持多种AI模型同台竞技
        体验不同AI的策略思考
==================================================
        """)

    def show_main_menu(self):
        """显示主菜单"""
        print("\n" + "="*50)
        print(" 主菜单")
        print("="*50)
        print("1.  快速开始游戏")
        print("2.  自定义游戏")
        print("3.  API配置管理")
        print("4.  系统状态")
        print("5.  帮助说明")
        print("6.  测试API配置")
        print("7.  批量切换模型")
        print("0.  退出程序")
        print("-"*50)

    def convert_api_to_llm_config(self, api_config: APIConfig) -> LLMConfig:
        """将APIConfig转换为LLMConfig"""
        return LLMConfig(
            name=api_config.name,
            provider=APIProvider(api_config.provider.value),
            api_key=api_config.api_key,
            model=api_config.model,
            base_url=api_config.base_url
        )

    def _get_role_configuration(self, num_players: int) -> dict:
        """获取N人局的角色配置"""
        # 基础配置: 狼人数量约为总人数的1/3
        werewolves = max(1, num_players // 3)
        good_guys = num_players - werewolves

        # 角色分配优先级：预言家 > 女巫 > 猎人 > 守卫 > 村民
        roles = []

        # 先分配狼人
        for i in range(werewolves):
            roles.append("狼人")

        # 好人角色分配
        special_roles = ["预言家", "女巫", "猎人", "守卫"]
        assigned_special = 0

        for i, role in enumerate(special_roles):
            if assigned_special < good_guys and assigned_special < len(special_roles):
                roles.append(role)
                assigned_special += 1

        # 剩余位置分配村民
        remaining_villagers = good_guys - assigned_special
        for i in range(remaining_villagers):
            roles.append("村民")

        description = " + ".join(roles)

        return {
            "werewolves": werewolves,
            "good_guys": good_guys,
            "roles": roles,
            "description": description
        }

    async def quick_start(self):
        """快速开始游戏"""
        print("\n 快速开始游戏")
        print("-"*30)

        # 获取所有配置
        configs = self.config_manager.list_configs()
        valid_configs = [c for c in configs if c.api_key and c.api_key not in ["your-openai-api-key", "your-siliconflow-api-key"]]

        if len(valid_configs) < 6:
            print(f" 需要至少6个有效的API配置才能开始游戏")
            print(f"   当前只有 {len(valid_configs)} 个有效配置")
            print(f"   总配置数: {len(configs)}")
            print("\n 请先进行API配置管理选项3添加或编辑API配置")
            return

        # 选择玩家数量
        max_players = min(16, len(valid_configs))
        print(f"\n 可用配置数: {len(valid_configs)}")
        print(f" 支持玩家数: 6-{max_players}")

        while True:
            try:
                num_players = int(input(f"请选择玩家数量 (6-{max_players}, 回车默认6): ") or "6")
                if 6 <= num_players <= max_players:
                    break
                else:
                    print(f" 玩家数量必须在6-{max_players}之间")
            except ValueError:
                print(" 请输入数字")

        # 自动选择配置
        selected_configs = valid_configs[:num_players]
        player_configs = []

        for i, config in enumerate(selected_configs):
            player_configs.append({
                "name": f"AI玩家{i+1}",
                "llm_config": self.convert_api_to_llm_config(config)
            })

        print(f"\n 自动选择了以下{num_players}个配置:")
        for i, (player, api_config) in enumerate(zip(player_configs, selected_configs), 1):
            print(f"   {i}. {player['name']} -> {api_config.name} ({api_config.provider.value} - {api_config.model})")

        # 显示角色配置
        role_config = self._get_role_configuration(num_players)
        print(f"\n 游戏规则:")
        print(f"    {num_players}人局 ({role_config['werewolves']}狼人 + {role_config['good_guys']}好人)")
        print(f"    角色配置: {role_config['description']}")
        print("    标准狼人杀规则")

        confirm = input("\n确认开始游戏? (y/N): ").strip().lower()
        if confirm == 'y':
            await self._start_game(player_configs, num_players)
        else:
            print(" 游戏已取消")

    async def custom_game(self):
        """自定义游戏"""
        print("\n 自定义游戏")
        print("-"*25)

        configs = self.config_manager.list_configs()
        valid_configs = [c for c in configs if c.api_key and c.api_key not in ["your-openai-api-key", "your-siliconflow-api-key"]]

        if len(valid_configs) < 6:
            print(f" 需要至少6个有效配置当前只有{len(valid_configs)}个")
            return

        # 选择玩家数量
        max_players = min(12, len(valid_configs))
        while True:
            try:
                num_players = int(input(f"玩家数量 (6-{max_players}): "))
                if 6 <= num_players <= max_players:
                    break
                else:
                    print(f" 玩家数量必须在6-{max_players}之间")
            except ValueError:
                print(" 请输入数字")

        # 显示可用配置
        print(f"\n 可用API配置:")
        for i, config in enumerate(valid_configs, 1):
            default_mark = " [默认]" if config.is_default else ""
            print(f"   {i}. {config.name}{default_mark}")
            print(f"      {config.provider.value} - {config.model}")
            print(f"      {config.description}")

        # 选择配置
        selected_configs = []
        selected_api_configs = []

        for i in range(num_players):
            print(f"\n 配置玩家 {i+1}:")

            # 显示未选择的配置
            available_indices = [j for j, config in enumerate(valid_configs)
                               if config not in selected_api_configs]

            if not available_indices:
                print(" 没有更多可用配置")
                break

            print("可选配置:")
            for j, idx in enumerate(available_indices, 1):
                config = valid_configs[idx]
                print(f"   {j}. {config.name} ({config.provider.value} - {config.model})")

            while True:
                try:
                    choice = int(input(f"选择配置 (1-{len(available_indices)}): "))
                    if 1 <= choice <= len(available_indices):
                        config_idx = available_indices[choice - 1]
                        api_config = valid_configs[config_idx]

                        player_name = input(f"玩家名称 (默认: {api_config.name}_Player): ").strip()
                        if not player_name:
                            player_name = f"{api_config.name}_Player"

                        selected_configs.append({
                            "name": player_name,
                            "llm_config": self.convert_api_to_llm_config(api_config)
                        })
                        selected_api_configs.append(api_config)
                        break
                    else:
                        print(" 无效选择")
                except ValueError:
                    print(" 请输入数字")

        if len(selected_configs) >= 6:
            num_players = len(selected_configs)
            print(f"\n 游戏配置完成:")
            for i, (player, api_config) in enumerate(zip(selected_configs, selected_api_configs), 1):
                print(f"   {i}. {player['name']} -> {api_config.name}")

            # 显示角色配置
            role_config = self._get_role_configuration(num_players)
            print(f"\n 游戏规则:")
            print(f"    {num_players}人局 ({role_config['werewolves']}狼人 + {role_config['good_guys']}好人)")
            print(f"    角色配置: {role_config['description']}")

            confirm = input("\n确认开始游戏? (y/N): ").strip().lower()
            if confirm == 'y':
                await self._start_game(selected_configs, num_players)
            else:
                print(" 游戏已取消")
        else:
            print(" 配置不足无法开始游戏")

    def manage_configs(self):
        """API配置管理"""
        self.config_ui.show_main_menu()

    def show_system_status(self):
        """显示系统状态"""
        print("\n 系统状态")
        print("-"*30)

        configs = self.config_manager.list_configs()
        valid_configs = [c for c in configs if c.api_key and c.api_key not in ["your-openai-api-key", "your-siliconflow-api-key"]]

        print(f" API配置统计:")
        print(f"   总配置数: {len(configs)}")
        print(f"   有效配置: {len(valid_configs)}")
        print(f"   默认配置: {sum(1 for c in configs if c.is_default)}")

        # 提供商分布
        if configs:
            provider_count = {}
            for config in configs:
                provider = config.provider.value
                provider_count[provider] = provider_count.get(provider, 0) + 1

            print(f"\n 提供商分布:")
            for provider, count in provider_count.items():
                print(f"   {provider}: {count} 个")

        # 有效配置列表
        if valid_configs:
            print(f"\n 有效配置:")
            for config in valid_configs:
                default_mark = " [默认]" if config.is_default else ""
                print(f"    {config.name}{default_mark}")
                print(f"     {config.provider.value} - {config.model}")
        else:
            print(f"\n 没有有效的配置")

        # 游戏就绪状态
        ready_for_game = len(valid_configs) >= 6
        status = " 就绪" if ready_for_game else " 未就绪"
        print(f"\n 游戏状态: {status}")
        if not ready_for_game:
            needed = 6 - len(valid_configs)
            print(f"   还需要 {needed} 个有效配置")

        input("\n按回车键继续...")

    def show_help(self):
        """显示帮助信息"""
        print("""
 AI狼人杀帮助文档
==================

 项目特点:
 支持多种AI模型同台竞技
 完整的狼人杀游戏规则实现
 智能AI角色扮演和策略思考
 清晰的配置管理界面

 快速开始:
1. 进入"API配置管理"添加至少6个API配置
2. 设置有效的API Key和模型
3. 选择"快速开始游戏"或"自定义游戏"

 支持的API提供商:
 OpenAI (GPT-4o, GPT-4o-mini等)
 SiliconFlow (Qwen, DeepSeek等)
 Anthropic (Claude-3.5-Sonnet等)
 其他兼容OpenAI格式的API

 游戏规则:
 6-12人局支持多种角色
 标准狼人杀规则
 AI会根据角色进行智能决策

 配置说明:
 每个AI需要独立的API配置
 建议使用不同模型测试AI能力差异
 可以在配置管理中测试API连接

 问题排查:
 确保API密钥正确且有余额
 检查网络连接和API访问
 在配置管理中测试API连接
        """)
        input("\n按回车键继续...")

    async def test_configs(self):
        """测试API配置"""
        print("\n 测试API配置")
        print("-"*25)

        configs = self.config_manager.list_configs()
        if not configs:
            print(" 暂无配置可测试")
            return

        print("选择要测试的配置:")
        for i, config in enumerate(configs, 1):
            status = "" if config.api_key and config.api_key not in ["your-openai-api-key", "your-siliconflow-api-key"] else ""
            print(f"   {i}. {status} {config.name} ({config.provider.value})")

        try:
            choice = int(input(f"请选择 (1-{len(configs)}): "))
            if 1 <= choice <= len(configs):
                config = configs[choice - 1]
                print(f"\n测试配置: {config.name}")

                # 这里可以添加实际的API测试逻辑
                if config.api_key and config.api_key not in ["your-openai-api-key", "your-siliconflow-api-key"]:
                    print(" 配置看起来正常")
                    print(" 建议在实际游戏中验证API连接")
                else:
                    print(" 请设置有效的API Key")
            else:
                print(" 无效选择")
        except ValueError:
            print(" 请输入数字")

        input("\n按回车键继续...")

    async def _start_game(self, player_configs: List[Dict[str, Any]], num_players: int = None):
        """启动游戏"""
        print(f"\n 正在启动游戏...")
        print("="*50)

        try:
            # 创建增强版游戏控制器
            controller = EnhancedWerewolfGameController(player_configs, num_players)

            # 初始化游戏
            print(" 初始化游戏...")
            if await controller.initialize_game():
                print(" 游戏初始化成功")

                # 开始游戏
                print(" 开始游戏...")
                await controller.start_game()

                # 显示游戏统计
                stats = controller.get_game_stats()
                print(f"\n 游戏结束统计:")
                print(f"   轮数: {stats.get('total_rounds', 0)}")
                print(f"   行动数: {stats.get('total_actions', 0)}")
                print(f"   获胜方: {stats.get('winner', '未知')}")

            else:
                print(" 游戏初始化失败")

        except KeyboardInterrupt:
            print("\n\n 游戏被用户中断")
        except Exception as e:
            print(f"\n 游戏运行出错: {e}")
            print(f"错误详情: {type(e).__name__}")

        input("\n按回车键返回主菜单...")

    def batch_switch_models(self):
        """批量切换模型"""
        print("\n 批量模型切换")
        print("-" * 30)

        # 获取所有配置
        configs = self.config_manager.list_configs()
        if not configs:
            print(" 没有找到任何API配置")
            return

        print(f" 当前共有 {len(configs)} 个配置")

        # 显示常用模型列表
        popular_models = {
            "1": {
                "name": "inclusionAI/Ling-mini-2.0",
                "description": "Ling-mini-2.0 (推荐新模型)"
            },
            "2": {
                "name": "Qwen/Qwen3-Next-80B-A3B-Instruct",
                "description": "Qwen3-Next-80B (当前主力模型)"
            },
            "3": {
                "name": "inclusionAI/Ring-flash-2.0",
                "description": "Ring-flash-2.0 (备用模型)"
            },
            "4": {
                "name": "deepseek-ai/DeepSeek-V3.1",
                "description": "DeepSeek-V3.1 (推理模型)"
            },
            "5": {
                "name": "deepseek-ai/DeepSeek-R1",
                "description": "DeepSeek-R1 (最新模型)"
            }
        }

        print("\n 可选模型:")
        for key, model in popular_models.items():
            print(f"  {key}. {model['name']}")
            print(f"     {model['description']}")
        print("  6. 自定义模型名称")
        print("  0. 返回主菜单")

        choice = input("\n请选择要切换的模型 (0-6): ").strip()

        if choice == "0":
            return
        elif choice in popular_models:
            target_model = popular_models[choice]["name"]
        elif choice == "6":
            target_model = input("\n请输入自定义模型名称: ").strip()
            if not target_model:
                print(" 模型名称不能为空")
                return
        else:
            print(" 无效选择")
            return

        # 确认操作
        print(f"\n 将把所有配置的模型切换为: {target_model}")
        confirm = input(" 确认执行? (y/N): ").strip().lower()

        if confirm not in ['y', 'yes']:
            print(" 操作已取消")
            return

        # 执行批量更新
        try:
            updated_count = 0
            print(f"\n 开始批量更新...")

            for config in configs:
                old_model = config.model
                config.model = target_model
                print(f"  {config.name}: {old_model} -> {target_model}")
                updated_count += 1

            # 保存配置
            self.config_manager.save_configs()

            print(f"\n 成功更新 {updated_count} 个配置!")
            print(f" 所有Bot现在使用: {target_model}")

            # 显示更新后的统计
            print(f"\n 更新统计:")
            model_count = {}
            for config in configs:
                model_count[config.model] = model_count.get(config.model, 0) + 1

            for model, count in model_count.items():
                print(f"  {model}: {count} 个配置")

        except Exception as e:
            print(f" 更新失败: {e}")

        input("\n按回车键返回主菜单...")

    async def run(self):
        """运行主程序"""
        self.show_welcome()

        while True:
            self.show_main_menu()
            choice = input("\n请选择操作 (0-7): ").strip()

            try:
                if choice == "0":
                    print("\n 感谢使用AI狼人杀再见")
                    break
                elif choice == "1":
                    await self.quick_start()
                elif choice == "2":
                    await self.custom_game()
                elif choice == "3":
                    self.manage_configs()
                elif choice == "4":
                    self.show_system_status()
                elif choice == "5":
                    self.show_help()
                elif choice == "6":
                    await self.test_configs()
                elif choice == "7":
                    self.batch_switch_models()
                else:
                    print(" 无效选择请重试")
            except KeyboardInterrupt:
                print("\n\n 操作被中断")
            except Exception as e:
                print(f"\n 操作出错: {e}")

            if choice != "0":
                print("\n" + "="*50)


def main():
    """主函数"""
    try:
        launcher = WerewolfGameLauncher()
        asyncio.run(launcher.run())
    except KeyboardInterrupt:
        print("\n\n程序已退出")
    except Exception as e:
        print(f"\n程序出错: {e}")


if __name__ == "__main__":
    main()