from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class ChatBackend(Protocol):
    async def generate_json(self, *, system: str, prompt: str) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ChatModelConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 90.0
    temperature: float = 0.7
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be positive")

    @classmethod
    def from_env(cls, prefix: str = "WOLFPLAY") -> ChatModelConfig:
        prefix = prefix.rstrip("_")
        names = {
            "base_url": f"{prefix}_BASE_URL",
            "api_key": f"{prefix}_API_KEY",
            "model": f"{prefix}_MODEL",
        }
        missing = [
            environment_name
            for environment_name in names.values()
            if not os.getenv(environment_name)
        ]
        if missing:
            raise RuntimeError(f"missing environment variables: {', '.join(missing)}")
        return cls(
            base_url=os.environ[names["base_url"]],
            api_key=os.environ[names["api_key"]],
            model=os.environ[names["model"]],
        )


class OpenAICompatibleBackend:
    """Minimal async client for OpenAI-compatible /chat/completions endpoints."""

    def __init__(self, config: ChatModelConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(timeout=config.timeout_seconds)

    async def generate_json(self, *, system: str, prompt: str) -> dict[str, Any]:
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.config.model,
                        "temperature": self.config.temperature,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("model response content must be a string")
                return _extract_json(content)
            except (httpx.HTTPError, KeyError, TypeError, ValueError):
                if attempt + 1 >= self.config.max_retries:
                    raise
                await asyncio.sleep(self.config.retry_backoff_seconds * (2**attempt))
        raise RuntimeError("unreachable retry state")

    async def aclose(self) -> None:
        await self._client.aclose()


def _extract_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```")
        content = content.removesuffix("```").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("model response does not contain a JSON object") from None
        parsed = json.loads(content[start:end])
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed
