import asyncio
from pathlib import Path

from wolfplay.web.config import WebSettings
from wolfplay.web.database import Database
from wolfplay.web.game_manager import GameManager
from wolfplay.web.realtime import RealtimeHub
from wolfplay.web.repository import StudioRepository


async def make_manager(
    tmp_path: Path,
) -> tuple[Database, StudioRepository, RealtimeHub, GameManager]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'studio.db'}")
    await database.initialize()
    repository = StudioRepository(database.sessions)
    await repository.initialize_defaults()
    settings = WebSettings(
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'studio.db'}",
        frontend_dist=tmp_path / "dist",
        max_concurrent_games=2,
    )
    hub = RealtimeHub(queue_size=128)
    manager = GameManager(repository=repository, hub=hub, settings=settings)
    return database, repository, hub, manager


async def test_game_manager_persists_and_broadcasts_public_events(tmp_path):
    database, repository, hub, manager = await make_manager(tmp_path)
    try:
        game = await manager.create_game({"seed": 7, "max_rounds": 2, "pace_seconds": 0.0})
        async with hub.subscribe(f"game:{game['id']}") as queue:
            await manager.wait(game["id"])
            messages = []
            while not queue.empty():
                messages.append(queue.get_nowait())

        completed = await repository.get_game(game["id"])
        public_events = await repository.list_game_events(game["id"], public_only=True)
        all_events = await repository.list_game_events(game["id"], public_only=False)

        assert completed["status"] == "completed"
        assert completed["result"]["decision_traces"]
        assert len(all_events) > len(public_events)
        assert all(
            message.get("event", {}).get("topic") != "role_assignment" for message in messages
        )
        assert any(message["type"] == "event" for message in messages)
        assert messages[-1]["type"] == "status"
        assert messages[-1]["game"]["status"] == "completed"
    finally:
        await manager.shutdown()
        await database.dispose()


async def test_concurrent_games_are_isolated(tmp_path):
    database, repository, hub, manager = await make_manager(tmp_path)
    try:
        first = await manager.create_game({"seed": 11, "max_rounds": 2, "pace_seconds": 0.0})
        second = await manager.create_game({"seed": 12, "max_rounds": 2, "pace_seconds": 0.0})
        async with (
            hub.subscribe(f"game:{first['id']}") as first_queue,
            hub.subscribe(f"game:{second['id']}") as second_queue,
        ):
            await asyncio.gather(manager.wait(first["id"]), manager.wait(second["id"]))
            first_messages = []
            second_messages = []
            while not first_queue.empty():
                first_messages.append(first_queue.get_nowait())
            while not second_queue.empty():
                second_messages.append(second_queue.get_nowait())

        assert first["id"] != second["id"]
        assert (await repository.get_game(first["id"]))["status"] == "completed"
        assert (await repository.get_game(second["id"]))["status"] == "completed"
        assert all(
            message.get("game", {}).get("id", first["id"]) == first["id"]
            for message in first_messages
            if message["type"] == "status"
        )
        assert all(
            message.get("game", {}).get("id", second["id"]) == second["id"]
            for message in second_messages
            if message["type"] == "status"
        )
    finally:
        await manager.shutdown()
        await database.dispose()


async def test_game_can_be_cancelled(tmp_path):
    database, repository, hub, manager = await make_manager(tmp_path)
    try:
        game = await manager.create_game({"seed": 99, "max_rounds": 8, "pace_seconds": 0.1})
        for _ in range(100):
            current = await repository.get_game(game["id"])
            if current["status"] == "running":
                break
            await asyncio.sleep(0.01)

        cancelled = await manager.cancel_game(game["id"])
        await manager.wait(game["id"])

        assert cancelled["status"] == "cancelled"
        assert (await repository.get_game(game["id"]))["status"] == "cancelled"
    finally:
        await manager.shutdown()
        await database.dispose()
