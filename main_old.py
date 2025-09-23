"""
AI狼人杀主程序
整合所有功能的启动脚本
"""

import json
import asyncio
import sys
import os
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werewolf.game_controller import WerewolfGameController
from werewolf.llm_manager import MultiLLMManager
from werewolf.config_manager import WerewolfConfigManager


class WerewolfGameLauncher:
    """狼人杀游戏启动器"""

    def __init__(self):
        self.llm_manager = MultiLLMManager()
        self.config_manager = WerewolfConfigManager()

    def show_welcome(self):
        """显示欢迎界面"""
        print("""
╔══════════════════════════════════════════════════╗
║                                                  ║
║     🐺 AI狼人杀 - 多AI Agent对战平台 🐺          ║
║                                                  ║
║     基于斗地主项目的LLM逻辑重构                   ║
║     支持多种AI模型同台竞技                       ║
║                                                  ║
╚══════════════════════════════════════════════════╝
        """)

    def show_main_menu(self):
        """显示主菜单"""
        print("\n🎮 主菜单:")
        print("1. 快速开始游戏")
        print("2. 配置管理")
        print("3. 创建自定义游戏")
        print("4. 加载游戏配置")
        print("5. 查看系统状态")
        print("6. 帮助说明")
        print("7. 退出程序")

    async def quick_start(self):
        """快速开始游戏"""
        print("\n🚀 快速开始")
        print("-" * 20)

        # 检查可用配置
        enabled_configs = self.llm_manager.get_enabled_configs()

        if len(enabled_configs) < 6:
            print(f"❌ 需要至少6个启用的LLM配置才能开始游戏")
            print(f"   当前只有 {len(enabled_configs)} 个启用的配置")
            print("\n请先进行配置管理（选项2）添加更多API配置")
            return

        # 自动选择前6个配置
        config_names = list(enabled_configs.keys())[:6]
        player_configs = []

        for i, config_name in enumerate(config_names):
            player_configs.append({
                "name": f"AI玩家{i+1}",
                "llm_config": config_name
            })

        print(f"✅ 自动选择了以下配置:")
        for i, config in enumerate(player_configs, 1):
            print(f"   {i}. {config['name']} (使用 {config['llm_config']})")

        confirm = input("\n确认开始游戏? (y/n): ").strip().lower()
        if confirm == 'y':
            await self._start_game(player_configs)
        else:
            print("❌ 游戏已取消")

    def manage_configs(self):
        """配置管理"""
        self.config_manager.interactive_setup()

    async def create_custom_game(self):
        """创建自定义游戏"""
        print("\n🎯 创建自定义游戏")
        print("-" * 25)

        enabled_configs = self.llm_manager.get_enabled_configs()

        if len(enabled_configs) < 6:
            print(f"❌ 需要至少6个启用的配置，当前只有{len(enabled_configs)}个")
            return

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

        # 显示可用配置
        config_names = list(enabled_configs.keys())
        print(f"\n📋 可用配置:")
        for i, name in enumerate(config_names, 1):
            config = enabled_configs[name]
            print(f"   {i}. {name} ({config.provider.value} - {config.model})")

        # 选择配置
        selected_configs = []
        for i in range(num_players):
            print(f"\n👤 配置玩家 {i+1}:")

            # 显示未选择的配置
            available = [name for name in config_names
                        if name not in [c["llm_config"] for c in selected_configs]]

            if not available:
                print("❌ 没有更多可用配置")
                break

            for j, name in enumerate(available, 1):
                print(f"   {j}. {name}")

            while True:
                try:
                    choice = int(input(f"选择配置 (1-{len(available)}): "))
                    if 1 <= choice <= len(available):
                        config_name = available[choice - 1]
                        player_name = input(f"玩家名称 (默认: {config_name}_Player): ").strip()
                        if not player_name:
                            player_name = f"{config_name}_Player"

                        selected_configs.append({
                            "name": player_name,
                            "llm_config": config_name
                        })
                        break
                    else:
                        print("❌ 无效选择")
                except ValueError:
                    print("❌ 请输入数字")

        if len(selected_configs) >= 6:
            print(f"\n🎮 游戏配置完成:")
            for i, config in enumerate(selected_configs, 1):
                print(f"   {i}. {config['name']} (使用 {config['llm_config']})")

            confirm = input("\n确认开始游戏? (y/n): ").strip().lower()
            if confirm == 'y':
                await self._start_game(selected_configs)
            else:
                print("❌ 游戏已取消")
        else:
            print("❌ 配置不足，无法开始游戏")

    async def load_game_config(self):
        """加载游戏配置"""
        print("\n📁 加载游戏配置")
        print("-" * 20)

        try:
            with open("werewolf/game_configs.json", "r", encoding="utf-8") as f:
                game_config = json.load(f)

            player_configs = game_config.get("players", [])
            if len(player_configs) < 6:
                print("❌ 配置文件中的玩家数量不足")
                return

            print(f"✅ 加载配置: {game_config.get('name', '未命名')}")
            print(f"📊 玩家配置:")
            for i, config in enumerate(player_configs, 1):
                print(f"   {i}. {config['name']} (使用 {config['llm_config']})")

            # 验证配置是否存在
            missing_configs = []
            for config in player_configs:
                if not self.llm_manager.get_config(config["llm_config"]):
                    missing_configs.append(config["llm_config"])

            if missing_configs:
                print(f"\n⚠️ 以下配置不存在: {', '.join(missing_configs)}")
                print("请先在配置管理中添加这些配置")
                return

            confirm = input("\n确认加载并开始游戏? (y/n): ").strip().lower()
            if confirm == 'y':
                await self._start_game(player_configs)
            else:
                print("❌ 游戏已取消")

        except FileNotFoundError:
            print("❌ 未找到游戏配置文件")
            print("请先创建游戏配置（选项3）")
        except Exception as e:
            print(f"❌ 加载配置失败: {e}")

    def show_system_status(self):
        """显示系统状态"""
        print("\n📊 系统状态")
        print("-" * 20)

        # LLM配置统计
        stats = self.llm_manager.get_stats()
        print(f"🔧 LLM配置:")
        print(f"   总配置数: {stats['total_configs']}")
        print(f"   启用配置: {stats['enabled_configs']}")
        print(f"   禁用配置: {stats['disabled_configs']}")

        # 提供商分布
        if stats['provider_distribution']:
            print(f"\n🏢 提供商分布:")
            for provider, count in stats['provider_distribution'].items():
                print(f"   {provider}: {count} 个")

        # 启用的配置列表
        enabled_configs = self.llm_manager.get_enabled_configs()
        if enabled_configs:
            print(f"\n✅ 启用的配置:")
            for name, config in enabled_configs.items():
                print(f"   {name}: {config.provider.value} - {config.model}")
        else:
            print(f"\n❌ 没有启用的配置")

        # 游戏就绪状态
        ready_for_game = len(enabled_configs) >= 6
        status = "✅ 就绪" if ready_for_game else "❌ 未就绪"
        print(f"\n🎮 游戏状态: {status}")
        if not ready_for_game:
            needed = 6 - len(enabled_configs)
            print(f"   还需要 {needed} 个启用的配置")

    def show_help(self):
        """显示帮助信息"""
        print("""
📖 AI狼人杀帮助文档
==================

🎯 项目特点:
- 基于斗地主项目的LLM逻辑重构
- 支持多种AI模型(GPT、Claude、DeepSeek等)
- 完整的狼人杀游戏规则实现
- 智能AI角色扮演和策略思考

🔧 快速开始:
1. 首先进入"配置管理"添加至少6个LLM API配置
2. 确保配置已启用并测试通过
3. 选择"快速开始游戏"或"创建自定义游戏"

📋 支持的API提供商:
- OpenAI (GPT-4o, GPT-4o-mini等)
- Anthropic (Claude-3.5-Sonnet等)
- DeepSeek (DeepSeek-Chat等)
- 硅基流动 (多种模型)
- 其他兼容OpenAI格式的API

🎮 游戏规则:
- 6-12人局，支持村民、狼人、预言家、女巫、猎人、守卫
- 标准狼人杀规则，夜晚行动+白天讨论+投票出局
- AI会根据角色特点进行智能决策和发言

⚙️ 配置说明:
- 每个AI需要独立的LLM配置
- 建议使用不同的模型来测试AI能力差异
- 可以导入导出配置以便分享

🐛 问题排查:
- 确保API密钥正确且有余额
- 检查网络连接和API访问
- 查看控制台错误信息
        """)

    async def _start_game(self, player_configs: List[Dict[str, str]]):
        """启动游戏"""
        print(f"\n🚀 正在启动游戏...")
        print("=" * 50)

        try:
            # 创建游戏控制器
            controller = WerewolfGameController(player_configs)

            # 初始化游戏
            if await controller.initialize_game():
                print("✅ 游戏初始化成功")

                # 开始游戏
                await controller.start_game()

                # 显示游戏统计
                stats = controller.get_game_stats()
                print(f"\n📈 游戏结束统计:")
                print(f"   轮数: {stats.get('total_rounds', 0)}")
                print(f"   行动数: {stats.get('total_actions', 0)}")
                print(f"   获胜方: {stats.get('winner', '未知')}")

            else:
                print("❌ 游戏初始化失败")

        except KeyboardInterrupt:
            print("\n\n⚠️ 游戏被用户中断")
        except Exception as e:
            print(f"\n❌ 游戏运行出错: {e}")

        input("\n按回车键返回主菜单...")

    async def run(self):
        """运行主程序"""
        self.show_welcome()

        while True:
            self.show_main_menu()
            choice = input("\n请选择操作 (1-7): ").strip()

            if choice == "1":
                await self.quick_start()
            elif choice == "2":
                self.manage_configs()
            elif choice == "3":
                await self.create_custom_game()
            elif choice == "4":
                await self.load_game_config()
            elif choice == "5":
                self.show_system_status()
            elif choice == "6":
                self.show_help()
            elif choice == "7":
                print("\n👋 感谢使用AI狼人杀，再见！")
                break
            else:
                print("❌ 无效选择，请重试")

            print("\n" + "="*50)


def main():
    """主函数"""
    launcher = WerewolfGameLauncher()
    try:
        asyncio.run(launcher.run())
    except KeyboardInterrupt:
        print("\n\n程序已退出")
    except Exception as e:
        print(f"\n程序出错: {e}")


if __name__ == "__main__":
    main()