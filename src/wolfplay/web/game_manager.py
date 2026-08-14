from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from wolfplay.engine import GameRuntime
from wolfplay.llm import ChatModelConfig, OpenAICompatibleBackend
from wolfplay.models import Faction, GameEvent

from .config import WebSettings
from .realtime import RealtimeHub
from .repository import ConflictError, StudioRepository


def _game_channel(game_id: str) -> str:
    return f"game:{game_id}"


class GameManager:
    def __init__(
        self,
        *,
        repository: StudioRepository,
        hub: RealtimeHub,
        settings: WebSettings,
    ) -> None:
        self.repository = repository
        self.hub = hub
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_games)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def create_game(self, config: dict[str, Any]) -> dict[str, Any]:
        seed = int(config.get("seed") if config.get("seed") is not None else secrets.randbits(31))
        max_rounds = int(config.get("max_rounds", 8))
        game_id = self._new_game_id()
        public_players = {
            f"player_{index}": {
                "player_id": f"player_{index}",
                "name": f"Player {index}",
                "alive": True,
            }
            for index in range(7)
        }
        normalized = {
            "seed": seed,
            "max_rounds": max_rounds,
            "pace_seconds": float(config.get("pace_seconds", 0.35)),
            "werewolf_agent_id": config.get("werewolf_agent_id", "heuristic"),
            "village_agent_id": config.get("village_agent_id", "heuristic"),
            "label": config.get("label") or f"Match {seed}",
        }
        game = await self.repository.create_game(
            game_id=game_id,
            seed=seed,
            max_rounds=max_rounds,
            config=normalized,
            players=public_players,
        )
        task = asyncio.create_task(self._run_game(game_id, normalized), name=f"game:{game_id}")
        async with self._lock:
            self._tasks[game_id] = task
        task.add_done_callback(
            lambda completed, current=game_id: self._task_finished(current, completed)
        )
        await self.hub.publish(
            _game_channel(game_id),
            {"type": "status", "game": game},
        )
        return game

    async def cancel_game(self, game_id: str) -> dict[str, Any]:
        game = await self.repository.get_game(game_id)
        if game["status"] not in {"queued", "running"}:
            raise ConflictError(f"game {game_id} is not active")
        async with self._lock:
            task = self._tasks.get(game_id)
        await self.repository.fail_game(game_id, "Cancelled by user.", status="cancelled")
        if task is not None:
            task.cancel()
        updated = await self.repository.get_game(game_id)
        await self.hub.publish(
            _game_channel(game_id),
            {"type": "status", "game": updated},
        )
        return updated

    async def wait(self, game_id: str) -> None:
        async with self._lock:
            task = self._tasks.get(game_id)
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def shutdown(self) -> None:
        async with self._lock:
            tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_game(self, game_id: str, config: dict[str, Any]) -> None:
        backends: Mapping[Faction, OpenAICompatibleBackend | None] = {}
        try:
            async with self._semaphore:
                game = await self.repository.get_game(game_id)
                if game["status"] == "cancelled":
                    return
                running = await self.repository.mark_game_running(game_id)
                await self.hub.publish(
                    _game_channel(game_id),
                    {"type": "status", "game": running},
                )
                backends = await self._build_backends(config)

                async def observe(event: GameEvent) -> None:
                    event_data = await self.repository.append_game_event(game_id, event)
                    if event.audience is None:
                        await self.hub.publish(
                            _game_channel(game_id),
                            {"type": "event", "event": event_data},
                        )

                runtime = GameRuntime(
                    game_id=game_id,
                    seed=config["seed"],
                    max_rounds=config["max_rounds"],
                    backend_by_faction=backends,
                    event_observer=observe,
                    public_event_delay_seconds=config["pace_seconds"],
                )
                result = await runtime.play()
                completed = await self.repository.complete_game(game_id, result.to_dict())
                await self.hub.publish(
                    _game_channel(game_id),
                    {"type": "status", "game": completed},
                )
        except asyncio.CancelledError:
            current = await self.repository.get_game(game_id)
            if current["status"] in {"queued", "running"}:
                await self.repository.fail_game(game_id, "Cancelled by user.", status="cancelled")
            raise
        except Exception as error:
            await self.repository.fail_game(game_id, str(error), status="failed")
            failed = await self.repository.get_game(game_id)
            await self.hub.publish(
                _game_channel(game_id),
                {"type": "error", "message": str(error)},
            )
            await self.hub.publish(
                _game_channel(game_id),
                {"type": "status", "game": failed},
            )
        finally:
            unique_backends = {
                id(backend): backend for backend in backends.values() if backend is not None
            }
            for backend in unique_backends.values():
                await backend.aclose()

    async def _build_backends(
        self, config: dict[str, Any]
    ) -> dict[Faction, OpenAICompatibleBackend | None]:
        wolf_profile = await self.repository.get_agent(config["werewolf_agent_id"])
        village_profile = await self.repository.get_agent(config["village_agent_id"])
        return {
            Faction.WEREWOLF: self._backend_from_profile(wolf_profile),
            Faction.VILLAGE: self._backend_from_profile(village_profile),
        }

    @staticmethod
    def _backend_from_profile(profile: dict[str, Any]) -> OpenAICompatibleBackend | None:
        if not profile["enabled"]:
            raise RuntimeError(f"agent profile is disabled: {profile['name']}")
        if profile["kind"] == "heuristic":
            return None
        prefix = (profile.get("env_prefix") or "WOLFPLAY").rstrip("_")
        base_url = profile.get("base_url") or os.getenv(f"{prefix}_BASE_URL")
        model = profile.get("model") or os.getenv(f"{prefix}_MODEL")
        api_key = os.getenv(f"{prefix}_API_KEY")
        missing = [
            name
            for name, value in {
                f"{prefix}_BASE_URL": base_url,
                f"{prefix}_MODEL": model,
                f"{prefix}_API_KEY": api_key,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"missing agent environment variables: {', '.join(missing)}")
        return OpenAICompatibleBackend(
            ChatModelConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=float(profile["temperature"]),
                timeout_seconds=float(profile["timeout_seconds"]),
            )
        )

    def _task_finished(self, game_id: str, task: asyncio.Task[None]) -> None:
        del task
        asyncio.create_task(self._remove_task(game_id))

    async def _remove_task(self, game_id: str) -> None:
        async with self._lock:
            self._tasks.pop(game_id, None)

    @staticmethod
    def _new_game_id() -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"game-{timestamp}-{uuid4().hex[:8]}"
