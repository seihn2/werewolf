"""
狼人杀多LLM API管理器
支持配置多个不同的AI API，用于测试不同AI的狼人杀能力
"""

import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class APIProvider(Enum):
    """支持的API提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"
    SILICONFLOW = "siliconflow"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """LLM配置类"""
    name: str  # 配置名称，用于标识
    provider: APIProvider
    api_key: str
    model: str
    base_url: str = None
    max_tokens: int = 2000
    temperature: float = 0.8
    enabled: bool = True
    description: str = ""  # 配置描述

    def __post_init__(self):
        if self.base_url is None:
            self.base_url = self._get_default_url()

    def _get_default_url(self) -> str:
        """获取默认API地址"""
        urls = {
            APIProvider.OPENAI: "https://api.openai.com/v1",
            APIProvider.ANTHROPIC: "https://api.anthropic.com",
            APIProvider.DEEPSEEK: "https://api.deepseek.com",
            APIProvider.MOONSHOT: "https://api.moonshot.cn/v1",
            APIProvider.SILICONFLOW: "https://api.siliconflow.cn/v1",
        }
        return urls.get(self.provider, "")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['provider'] = self.provider.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        """从字典创建"""
        data = data.copy()
        if 'provider' in data:
            data['provider'] = APIProvider(data['provider'])
        return cls(**data)


class LLMAPIClient:
    """统一的LLM API客户端"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def chat_completion(self, prompt: str, system_prompt: str = None) -> str:
        """调用聊天完成API"""
        if not self.config.enabled:
            raise Exception(f"API配置 {self.config.name} 已禁用")

        if self.config.provider == APIProvider.OPENAI:
            return await self._openai_completion(prompt, system_prompt)
        elif self.config.provider == APIProvider.ANTHROPIC:
            return await self._anthropic_completion(prompt, system_prompt)
        elif self.config.provider == APIProvider.DEEPSEEK:
            return await self._deepseek_completion(prompt, system_prompt)
        elif self.config.provider == APIProvider.MOONSHOT:
            return await self._moonshot_completion(prompt, system_prompt)
        elif self.config.provider == APIProvider.SILICONFLOW:
            return await self._siliconflow_completion(prompt, system_prompt)
        elif self.config.provider == APIProvider.CUSTOM:
            return await self._custom_completion(prompt, system_prompt)
        else:
            raise ValueError(f"不支持的提供商: {self.config.provider}")

    async def _openai_completion(self, prompt: str, system_prompt: str = None) -> str:
        """OpenAI API调用"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }

        async with self.session.post(
            f"{self.config.base_url}/chat/completions",
            headers=headers,
            json=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API错误 ({response.status}): {error_text}")

            result = await response.json()
            return result["choices"][0]["message"]["content"]

    async def _anthropic_completion(self, prompt: str, system_prompt: str = None) -> str:
        """Claude API调用"""
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        messages = [{"role": "user", "content": prompt}]

        data = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": messages
        }

        if system_prompt:
            data["system"] = system_prompt

        async with self.session.post(
            f"{self.config.base_url}/messages",
            headers=headers,
            json=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Anthropic API错误 ({response.status}): {error_text}")

            result = await response.json()
            return result["content"][0]["text"]

    async def _deepseek_completion(self, prompt: str, system_prompt: str = None) -> str:
        """DeepSeek API调用"""
        # DeepSeek使用类似OpenAI的接口
        return await self._openai_completion(prompt, system_prompt)

    async def _moonshot_completion(self, prompt: str, system_prompt: str = None) -> str:
        """Moonshot API调用"""
        # Moonshot使用类似OpenAI的接口
        return await self._openai_completion(prompt, system_prompt)

    async def _siliconflow_completion(self, prompt: str, system_prompt: str = None) -> str:
        """硅基流动API调用"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False
        }

        async with self.session.post(
            f"{self.config.base_url}/chat/completions",
            headers=headers,
            json=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"硅基流动API错误 ({response.status}): {error_text}")

            result = await response.json()

            if "error" in result:
                raise Exception(f"硅基流动API错误: {result['error']['message']}")

            return result["choices"][0]["message"]["content"]

    async def _custom_completion(self, prompt: str, system_prompt: str = None) -> str:
        """自定义API调用"""
        # 这里可以实现自定义的API调用逻辑
        # 默认使用OpenAI兼容格式
        return await self._openai_completion(prompt, system_prompt)


class MultiLLMManager:
    """多LLM管理器"""

    def __init__(self, config_file: str = "werewolf/llm_configs.json"):
        self.config_file = config_file
        self.configs: Dict[str, LLMConfig] = {}
        self.load_configs()

    def load_configs(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for name, config_data in data.items():
                    self.configs[name] = LLMConfig.from_dict(config_data)
        except FileNotFoundError:
            print(f"配置文件 {self.config_file} 不存在，将创建默认配置")
            self._create_default_configs()
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            self._create_default_configs()

    def save_configs(self):
        """保存配置文件"""
        try:
            data = {name: config.to_dict() for name, config in self.configs.items()}
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"配置已保存到 {self.config_file}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")

    def _create_default_configs(self):
        """创建默认配置"""
        self.configs = {
            "gpt4o": LLMConfig(
                name="gpt4o",
                provider=APIProvider.OPENAI,
                api_key="your-openai-api-key",
                model="gpt-4o",
                description="GPT-4o，OpenAI的高性能模型",
                enabled=False
            ),
            "claude": LLMConfig(
                name="claude",
                provider=APIProvider.ANTHROPIC,
                api_key="your-anthropic-api-key",
                model="claude-3-5-sonnet-20241022",
                description="Claude-3.5-Sonnet，Anthropic的智能模型",
                enabled=False
            ),
            "deepseek": LLMConfig(
                name="deepseek",
                provider=APIProvider.DEEPSEEK,
                api_key="your-deepseek-api-key",
                model="deepseek-chat",
                description="DeepSeek聊天模型，性价比高",
                enabled=False
            ),
            "qwen": LLMConfig(
                name="qwen",
                provider=APIProvider.SILICONFLOW,
                api_key="your-siliconflow-api-key",
                model="Qwen/Qwen2.5-72B-Instruct",
                description="通义千问2.5-72B，通过硅基流动调用",
                enabled=False
            )
        }
        self.save_configs()

    def add_config(self, config: LLMConfig) -> bool:
        """添加新配置"""
        if config.name in self.configs:
            print(f"配置 {config.name} 已存在")
            return False

        self.configs[config.name] = config
        self.save_configs()
        print(f"已添加配置: {config.name}")
        return True

    def update_config(self, name: str, config: LLMConfig) -> bool:
        """更新配置"""
        if name not in self.configs:
            print(f"配置 {name} 不存在")
            return False

        config.name = name  # 确保名称一致
        self.configs[name] = config
        self.save_configs()
        print(f"已更新配置: {name}")
        return True

    def remove_config(self, name: str) -> bool:
        """删除配置"""
        if name not in self.configs:
            print(f"配置 {name} 不存在")
            return False

        del self.configs[name]
        self.save_configs()
        print(f"已删除配置: {name}")
        return True

    def get_config(self, name: str) -> Optional[LLMConfig]:
        """获取配置"""
        return self.configs.get(name)

    def get_enabled_configs(self) -> Dict[str, LLMConfig]:
        """获取所有启用的配置"""
        return {name: config for name, config in self.configs.items() if config.enabled}

    def list_configs(self) -> Dict[str, LLMConfig]:
        """列出所有配置"""
        return self.configs.copy()

    def enable_config(self, name: str) -> bool:
        """启用配置"""
        if name not in self.configs:
            return False
        self.configs[name].enabled = True
        self.save_configs()
        return True

    def disable_config(self, name: str) -> bool:
        """禁用配置"""
        if name not in self.configs:
            return False
        self.configs[name].enabled = False
        self.save_configs()
        return True

    async def test_config(self, name: str) -> bool:
        """测试配置是否可用"""
        config = self.get_config(name)
        if not config:
            print(f"配置 {name} 不存在")
            return False

        try:
            async with LLMAPIClient(config) as client:
                response = await client.chat_completion(
                    "你好，这是一个API连接测试。请简单回复'测试成功'。",
                    "你是一个AI助手，正在进行连接测试。"
                )
                print(f"配置 {name} 测试成功: {response[:50]}...")
                return True
        except Exception as e:
            print(f"配置 {name} 测试失败: {e}")
            return False

    async def test_all_enabled_configs(self) -> Dict[str, bool]:
        """测试所有启用的配置"""
        enabled_configs = self.get_enabled_configs()
        results = {}

        tasks = []
        for name in enabled_configs:
            tasks.append(self.test_config(name))

        test_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, name in enumerate(enabled_configs):
            results[name] = not isinstance(test_results[i], Exception) and test_results[i]

        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.configs)
        enabled = len(self.get_enabled_configs())

        provider_count = {}
        for config in self.configs.values():
            provider = config.provider.value
            provider_count[provider] = provider_count.get(provider, 0) + 1

        return {
            "total_configs": total,
            "enabled_configs": enabled,
            "disabled_configs": total - enabled,
            "provider_distribution": provider_count
        }


# 配置管理工具函数
def create_quick_config(name: str, provider: str, api_key: str, model: str,
                       description: str = "") -> LLMConfig:
    """快速创建配置"""
    try:
        provider_enum = APIProvider(provider.lower())
    except ValueError:
        provider_enum = APIProvider.CUSTOM

    return LLMConfig(
        name=name,
        provider=provider_enum,
        api_key=api_key,
        model=model,
        description=description
    )


# 使用示例和工具函数
if __name__ == "__main__":
    # 使用示例
    async def main():
        # 创建管理器
        manager = MultiLLMManager()

        # 添加一些示例配置
        configs_to_add = [
            create_quick_config(
                "gpt4o-mini",
                "openai",
                "sk-your-key",
                "gpt-4o-mini",
                "GPT-4o-mini，快速且经济"
            ),
            create_quick_config(
                "yi-large",
                "siliconflow",
                "sk-your-key",
                "01-ai/Yi-Large",
                "零一万物Yi-Large，通过硅基流动"
            )
        ]

        for config in configs_to_add:
            manager.add_config(config)

        # 显示统计信息
        stats = manager.get_stats()
        print(f"配置统计: {stats}")

        # 列出所有配置
        print("\n所有配置:")
        for name, config in manager.list_configs().items():
            status = "启用" if config.enabled else "禁用"
            print(f"  {name}: {config.provider.value} - {config.model} [{status}]")

    # 运行示例
    # asyncio.run(main())
    print("狼人杀多LLM管理器已就绪")