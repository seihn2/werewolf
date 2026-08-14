from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any


class RealtimeHub:
    def __init__(self, *, queue_size: int = 512) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self.queue_size = queue_size
        self._channels: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def subscribe(self, channel: str) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.queue_size)
        async with self._lock:
            self._channels[channel].add(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                subscribers = self._channels.get(channel)
                if subscribers is not None:
                    subscribers.discard(queue)
                    if not subscribers:
                        self._channels.pop(channel, None)

    async def publish(self, channel: str, message: dict[str, Any]) -> int:
        async with self._lock:
            subscribers = tuple(self._channels.get(channel, ()))
        for queue in subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(deepcopy(message))
        return len(subscribers)

    async def subscriber_count(self, channel: str) -> int:
        async with self._lock:
            return len(self._channels.get(channel, ()))
