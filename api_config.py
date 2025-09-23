"""
API配置管理器
统一管理LLM API配置提供友好的配置界面
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class APIProvider(Enum):
    """API提供商"""
    OPENAI = "openai"
    SILICONFLOW = "siliconflow"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


@dataclass
class APIConfig:
    """API配置"""
    name: str  # 配置名称
    provider: APIProvider  # 提供商
    api_key: str  # API密钥
    base_url: str  # API基础URL
    model: str  # 模型名称
    description: str = ""  # 配置描述
    is_default: bool = False  # 是否为默认配置


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_file: str = "werewolf/api_configs.json"):
        self.config_file = config_file
        self.configs: Dict[str, APIConfig] = {}
        self.load_configs()

    def load_configs(self):
        """加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for name, config_data in data.items():
                        config_data['provider'] = APIProvider(config_data['provider'])
                        self.configs[name] = APIConfig(**config_data)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                self.create_default_configs()
        else:
            print("配置文件不存在创建默认配置")
            self.create_default_configs()

    def save_configs(self):
        """保存配置"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            data = {}
            for name, config in self.configs.items():
                config_dict = asdict(config)
                config_dict['provider'] = config.provider.value
                data[name] = config_dict

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"配置已保存到 {self.config_file}")
        except Exception as e:
            print(f"保存配置失败: {e}")

    def create_default_configs(self):
        """创建默认配置"""
        self.configs = {
            "openai_gpt4": APIConfig(
                name="openai_gpt4",
                provider=APIProvider.OPENAI,
                api_key="your-openai-api-key",
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
                description="OpenAI GPT-4 模型",
                is_default=True
            ),
            "siliconflow_qwen": APIConfig(
                name="siliconflow_qwen",
                provider=APIProvider.SILICONFLOW,
                api_key="your-siliconflow-api-key",
                base_url="https://api.siliconflow.cn/v1",
                model="Qwen/Qwen2.5-7B-Instruct",
                description="SiliconFlow Qwen模型"
            ),
            "siliconflow_deepseek": APIConfig(
                name="siliconflow_deepseek",
                provider=APIProvider.SILICONFLOW,
                api_key="your-siliconflow-api-key",
                base_url="https://api.siliconflow.cn/v1",
                model="deepseek-ai/DeepSeek-V2.5",
                description="SiliconFlow DeepSeek模型"
            )
        }
        self.save_configs()

    def get_config(self, name: str) -> Optional[APIConfig]:
        """获取指定配置"""
        return self.configs.get(name)

    def get_default_config(self) -> Optional[APIConfig]:
        """获取默认配置"""
        for config in self.configs.values():
            if config.is_default:
                return config
        # 如果没有默认配置返回第一个
        return next(iter(self.configs.values())) if self.configs else None

    def list_configs(self) -> List[APIConfig]:
        """列出所有配置"""
        return list(self.configs.values())

    def add_config(self, config: APIConfig):
        """添加配置"""
        self.configs[config.name] = config
        self.save_configs()

    def remove_config(self, name: str) -> bool:
        """删除配置"""
        if name in self.configs:
            del self.configs[name]
            self.save_configs()
            return True
        return False

    def set_default(self, name: str) -> bool:
        """设置默认配置"""
        if name not in self.configs:
            return False

        # 清除所有默认标记
        for config in self.configs.values():
            config.is_default = False

        # 设置新的默认配置
        self.configs[name].is_default = True
        self.save_configs()
        return True


class ConfigUI:
    """配置界面"""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def show_main_menu(self):
        """显示主菜单"""
        while True:
            print("\n" + "="*50)
            print(" API配置管理")
            print("="*50)
            print("1. 查看所有配置")
            print("2. 添加新配置")
            print("3. 编辑配置")
            print("4. 删除配置")
            print("5. 设置默认配置")
            print("6. 测试配置")
            print("0. 返回主菜单")
            print("-"*50)

            choice = input("请选择操作 (0-6): ").strip()

            if choice == "0":
                break
            elif choice == "1":
                self.show_all_configs()
            elif choice == "2":
                self.add_new_config()
            elif choice == "3":
                self.edit_config()
            elif choice == "4":
                self.delete_config()
            elif choice == "5":
                self.set_default_config()
            elif choice == "6":
                self.test_config()
            else:
                print(" 无效选择请重试")

    def show_all_configs(self):
        """显示所有配置"""
        configs = self.config_manager.list_configs()
        if not configs:
            print("\n 暂无配置")
            return

        print("\n 当前所有配置:")
        print("-"*80)
        for i, config in enumerate(configs, 1):
            default_mark = " [默认]" if config.is_default else ""
            print(f"{i}. {config.name}{default_mark}")
            print(f"   提供商: {config.provider.value}")
            print(f"   模型: {config.model}")
            print(f"   描述: {config.description}")
            print(f"   API Key: {config.api_key[:10]}..." if config.api_key else "   API Key: 未设置")
            print("-"*80)

        input("\n按回车键继续...")

    def add_new_config(self):
        """添加新配置"""
        print("\n 添加新配置")
        print("-"*30)

        try:
            name = input("配置名称: ").strip()
            if not name:
                print(" 配置名称不能为空")
                return

            if name in self.config_manager.configs:
                print(" 配置名称已存在")
                return

            # 选择提供商
            providers = list(APIProvider)
            print("\n选择API提供商:")
            for i, provider in enumerate(providers, 1):
                print(f"{i}. {provider.value}")

            provider_choice = input("请选择 (1-{}): ".format(len(providers))).strip()
            try:
                provider = providers[int(provider_choice) - 1]
            except (ValueError, IndexError):
                print(" 无效选择")
                return

            # 根据提供商设置默认值
            if provider == APIProvider.OPENAI:
                default_url = "https://api.openai.com/v1"
                default_model = "gpt-4o-mini"
            elif provider == APIProvider.SILICONFLOW:
                default_url = "https://api.siliconflow.cn/v1"
                default_model = "Qwen/Qwen2.5-7B-Instruct"
            elif provider == APIProvider.ANTHROPIC:
                default_url = "https://api.anthropic.com"
                default_model = "claude-3-sonnet-20240229"
            else:
                default_url = ""
                default_model = ""

            api_key = input("API Key: ").strip()
            base_url = input(f"Base URL (默认: {default_url}): ").strip() or default_url
            model = input(f"模型名称 (默认: {default_model}): ").strip() or default_model
            description = input("配置描述 (可选): ").strip()

            config = APIConfig(
                name=name,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                description=description
            )

            self.config_manager.add_config(config)
            print(f" 配置 '{name}' 添加成功!")

        except KeyboardInterrupt:
            print("\n 操作已取消")

    def edit_config(self):
        """编辑配置"""
        configs = self.config_manager.list_configs()
        if not configs:
            print("\n 暂无配置可编辑")
            return

        print("\n 编辑配置")
        print("选择要编辑的配置:")
        for i, config in enumerate(configs, 1):
            print(f"{i}. {config.name}")

        try:
            choice = int(input("请选择 (1-{}): ".format(len(configs))).strip())
            config = configs[choice - 1]

            print(f"\n编辑配置: {config.name}")
            print("(直接回车保持原值)")

            new_api_key = input(f"API Key [{config.api_key[:10]}...]: ").strip()
            if new_api_key:
                config.api_key = new_api_key

            new_base_url = input(f"Base URL [{config.base_url}]: ").strip()
            if new_base_url:
                config.base_url = new_base_url

            new_model = input(f"模型名称 [{config.model}]: ").strip()
            if new_model:
                config.model = new_model

            new_description = input(f"描述 [{config.description}]: ").strip()
            if new_description:
                config.description = new_description

            self.config_manager.save_configs()
            print(" 配置更新成功!")

        except (ValueError, IndexError):
            print(" 无效选择")
        except KeyboardInterrupt:
            print("\n 操作已取消")

    def delete_config(self):
        """删除配置"""
        configs = self.config_manager.list_configs()
        if not configs:
            print("\n 暂无配置可删除")
            return

        print("\n 删除配置")
        print("选择要删除的配置:")
        for i, config in enumerate(configs, 1):
            print(f"{i}. {config.name}")

        try:
            choice = int(input("请选择 (1-{}): ".format(len(configs))).strip())
            config = configs[choice - 1]

            confirm = input(f"确定删除配置 '{config.name}'? (y/N): ").strip().lower()
            if confirm == 'y':
                self.config_manager.remove_config(config.name)
                print(" 配置删除成功!")
            else:
                print(" 操作已取消")

        except (ValueError, IndexError):
            print(" 无效选择")
        except KeyboardInterrupt:
            print("\n 操作已取消")

    def set_default_config(self):
        """设置默认配置"""
        configs = self.config_manager.list_configs()
        if not configs:
            print("\n 暂无配置")
            return

        print("\n 设置默认配置")
        print("选择默认配置:")
        for i, config in enumerate(configs, 1):
            default_mark = " [当前默认]" if config.is_default else ""
            print(f"{i}. {config.name}{default_mark}")

        try:
            choice = int(input("请选择 (1-{}): ".format(len(configs))).strip())
            config = configs[choice - 1]

            self.config_manager.set_default(config.name)
            print(f" 已设置 '{config.name}' 为默认配置!")

        except (ValueError, IndexError):
            print(" 无效选择")
        except KeyboardInterrupt:
            print("\n 操作已取消")

    def test_config(self):
        """测试配置"""
        configs = self.config_manager.list_configs()
        if not configs:
            print("\n 暂无配置可测试")
            return

        print("\n 测试配置")
        print("选择要测试的配置:")
        for i, config in enumerate(configs, 1):
            print(f"{i}. {config.name}")

        try:
            choice = int(input("请选择 (1-{}): ".format(len(configs))).strip())
            config = configs[choice - 1]

            print(f"\n测试配置: {config.name}")
            print("正在测试API连接...")

            # 这里可以添加实际的API测试逻辑
            # 目前只是模拟测试
            if config.api_key and config.api_key != "your-openai-api-key" and config.api_key != "your-siliconflow-api-key":
                print(" 配置看起来正常 (注意: 这只是基础检查)")
            else:
                print(" 请设置有效的API Key")

        except (ValueError, IndexError):
            print(" 无效选择")
        except KeyboardInterrupt:
            print("\n 操作已取消")


if __name__ == "__main__":
    # 测试配置管理器
    config_manager = ConfigManager()
    ui = ConfigUI(config_manager)
    ui.show_main_menu()