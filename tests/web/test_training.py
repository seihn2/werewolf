import asyncio
import sys
from pathlib import Path

import pytest

from wolfplay.web.commands import JobCommand, build_training_command
from wolfplay.web.config import WebSettings
from wolfplay.web.database import Database
from wolfplay.web.realtime import RealtimeHub
from wolfplay.web.repository import StudioRepository
from wolfplay.web.training import TrainingManager


def settings_for(tmp_path: Path) -> WebSettings:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return WebSettings(
        data_dir=tmp_path,
        artifact_dir=artifact_dir,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'studio.db'}",
        frontend_dist=tmp_path / "dist",
    )


def test_command_builder_restricts_inputs_to_artifact_root(tmp_path):
    settings = settings_for(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inside the configured artifact"):
        build_training_command(
            kind="latent",
            config={"input_path": "../outside.jsonl"},
            job_dir=settings.artifact_dir / "jobs" / "job-test",
            settings=settings,
        )


async def test_training_manager_streams_logs_and_persists_completion(tmp_path):
    settings = settings_for(tmp_path)
    database = Database(settings.database_url)
    await database.initialize()
    repository = StudioRepository(database.sessions)
    await repository.initialize_defaults()
    hub = RealtimeHub(queue_size=32)

    def command_builder(**kwargs):
        job_dir = kwargs["job_dir"]
        output = job_dir / "result.json"
        code = (
            "import json, pathlib; "
            f"pathlib.Path({str(output)!r}).write_text('ok'); "
            "print(json.dumps({'records': 3}))"
        )
        return JobCommand((sys.executable, "-c", code), output, "test stage")

    manager = TrainingManager(
        repository=repository,
        hub=hub,
        settings=settings,
        command_builder=command_builder,
    )
    try:
        job = await manager.create_job(kind="self_play", config={})
        async with hub.subscribe(f"training:{job['id']}") as queue:
            await manager.wait(job["id"])
            messages = []
            while not queue.empty():
                messages.append(queue.get_nowait())

        completed = await repository.get_training_job(job["id"])
        logs = await manager.read_logs(job["id"])

        assert completed["status"] == "completed"
        assert completed["progress"] == 1.0
        assert completed["metrics"] == {"records": 3}
        assert await asyncio.to_thread(Path(completed["output_path"]).read_text) == "ok"
        assert logs["lines"] == ['{"records": 3}']
        assert any(message["type"] == "log" for message in messages)
        assert messages[-1]["type"] == "status"
    finally:
        await manager.shutdown()
        await database.dispose()


async def test_training_manager_records_failed_process(tmp_path):
    settings = settings_for(tmp_path)
    database = Database(settings.database_url)
    await database.initialize()
    repository = StudioRepository(database.sessions)
    await repository.initialize_defaults()

    def command_builder(**kwargs):
        output = kwargs["job_dir"] / "none"
        return JobCommand(
            (sys.executable, "-c", "import sys; print('broken'); sys.exit(7)"),
            output,
            "test failure",
        )

    manager = TrainingManager(
        repository=repository,
        hub=RealtimeHub(),
        settings=settings,
        command_builder=command_builder,
    )
    try:
        job = await manager.create_job(kind="self_play", config={})
        await manager.wait(job["id"])
        failed = await repository.get_training_job(job["id"])

        assert failed["status"] == "failed"
        assert failed["exit_code"] == 7
        assert "broken" in failed["error"]
    finally:
        await manager.shutdown()
        await database.dispose()
