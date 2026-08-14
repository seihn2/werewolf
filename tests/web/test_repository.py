from pathlib import Path

import pytest

from wolfplay.models import GameEvent, Phase
from wolfplay.web.database import Database
from wolfplay.web.repository import ConflictError, StudioRepository


async def build_repository(path: Path) -> tuple[Database, StudioRepository]:
    database = Database(f"sqlite+aiosqlite:///{path}")
    await database.initialize()
    repository = StudioRepository(database.sessions)
    await repository.initialize_defaults()
    return database, repository


async def test_game_events_persist_with_visibility_and_analytics(tmp_path):
    database, repository = await build_repository(tmp_path / "studio.db")
    try:
        players = {
            f"player_{index}": {
                "player_id": f"player_{index}",
                "name": f"Player {index}",
                "alive": True,
            }
            for index in range(7)
        }
        await repository.create_game(
            game_id="game-test",
            seed=42,
            max_rounds=3,
            config={"pace_seconds": 0.0},
            players=players,
        )
        await repository.mark_game_running("game-test")
        await repository.append_game_event(
            "game-test",
            GameEvent(
                logical_time=1,
                topic="game_started",
                round_no=1,
                phase=Phase.SETUP,
                payload={"players": list(players)},
            ),
        )
        await repository.append_game_event(
            "game-test",
            GameEvent(
                logical_time=2,
                topic="role_assignment",
                round_no=1,
                phase=Phase.SETUP,
                payload={"player_id": "player_0", "role": "werewolf"},
                audience=("player_0",),
            ),
        )
        await repository.append_game_event(
            "game-test",
            GameEvent(
                logical_time=3,
                topic="vote_result",
                round_no=1,
                phase=Phase.VOTE_RESOLUTION,
                payload={"eliminated_id": "player_3", "tally": {"player_3": 4}},
            ),
        )
        result = {
            "game_id": "game-test",
            "seed": 42,
            "rounds": 1,
            "winner": "werewolf",
            "termination_reason": "werewolf_parity",
            "players": {
                **players,
                "player_0": {**players["player_0"], "role": "werewolf", "alive": True},
            },
            "events": [],
            "decision_traces": [
                {
                    "role": "werewolf",
                    "reflection": "repair",
                    "action": {"strategy": "seer_claim"},
                    "evaluations": [{"legal": False}, {"legal": True}],
                }
            ],
        }
        completed = await repository.complete_game("game-test", result)

        public_events = await repository.list_game_events("game-test", public_only=True)
        omniscient_events = await repository.list_game_events("game-test", public_only=False)
        overview = await repository.analytics_overview()

        assert completed["status"] == "completed"
        assert completed["players"]["player_0"]["role"] == "werewolf"
        assert [event["topic"] for event in public_events] == ["game_started", "vote_result"]
        assert len(omniscient_events) == 3
        assert overview["completed_games"] == 1
        assert overview["winner_counts"] == {"werewolf": 1}
        assert overview["reflection_rate"] == 1.0
        assert overview["illegal_evaluation_rate"] == 0.5
    finally:
        await database.dispose()


async def test_recovery_marks_active_records_interrupted(tmp_path):
    database, repository = await build_repository(tmp_path / "recovery.db")
    try:
        await repository.create_game(
            game_id="game-running",
            seed=1,
            max_rounds=2,
            config={},
            players={},
        )
        await repository.mark_game_running("game-running")
        await repository.create_training_job(
            job_id="job-running",
            kind="self_play",
            config={},
            output_path="output.jsonl",
            log_path="job.log",
        )
        await repository.update_training_job("job-running", status="running")

        recovered = await repository.recover_interrupted()

        assert recovered == {"games": 1, "jobs": 1}
        assert (await repository.get_game("game-running"))["status"] == "interrupted"
        assert (await repository.get_training_job("job-running"))["status"] == "interrupted"
    finally:
        await database.dispose()


async def test_builtin_agent_cannot_be_deleted(tmp_path):
    database, repository = await build_repository(tmp_path / "agents.db")
    try:
        agents = await repository.list_agents()
        assert [agent["id"] for agent in agents] == ["heuristic"]
        with pytest.raises(ConflictError):
            await repository.delete_agent("heuristic")
    finally:
        await database.dispose()
