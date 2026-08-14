from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .commands import JobCommand, build_training_command
from .config import WebSettings
from .realtime import RealtimeHub
from .repository import ConflictError, StudioRepository
from .tables import utc_now

CommandBuilder = Callable[..., JobCommand]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _job_channel(job_id: str) -> str:
    return f"training:{job_id}"


class TrainingManager:
    def __init__(
        self,
        *,
        repository: StudioRepository,
        hub: RealtimeHub,
        settings: WebSettings,
        command_builder: CommandBuilder = build_training_command,
    ) -> None:
        self.repository = repository
        self.hub = hub
        self.settings = settings
        self.command_builder = command_builder
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, *, kind: str, config: dict[str, Any]) -> dict[str, Any]:
        job_id = self._new_job_id()
        job_dir = self.settings.resolve_artifact_path(Path("jobs") / job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        command = self.command_builder(
            kind=kind,
            config=config,
            job_dir=job_dir,
            settings=self.settings,
        )
        log_path = job_dir / "job.log"
        job = await self.repository.create_training_job(
            job_id=job_id,
            kind=kind,
            config=config,
            output_path=str(command.output_path),
            log_path=str(log_path),
        )
        task = asyncio.create_task(
            self._run_job(job_id, command, log_path), name=f"training:{job_id}"
        )
        async with self._lock:
            self._tasks[job_id] = task
        task.add_done_callback(
            lambda completed, current=job_id: self._task_finished(current, completed)
        )
        return job

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = await self.repository.get_training_job(job_id)
        if job["status"] not in {"queued", "running", "cancelling"}:
            raise ConflictError(f"training job {job_id} is not active")
        async with self._lock:
            task = self._tasks.get(job_id)
            process = self._processes.get(job_id)
        if process is None:
            if task is not None:
                task.cancel()
            cancelled = await self.repository.update_training_job(
                job_id,
                status="cancelled",
                stage="cancelled",
                progress=0.0,
                error="Cancelled by user.",
                completed_at=utc_now(),
            )
        else:
            cancelled = await self.repository.update_training_job(
                job_id,
                status="cancelling",
                stage="stopping process",
            )
            await self._terminate_process(process)
        await self.hub.publish(
            _job_channel(job_id),
            {"type": "status", "job": cancelled},
        )
        return cancelled

    async def read_logs(self, job_id: str, *, offset: int = 0, limit: int = 500) -> dict[str, Any]:
        job = await self.repository.get_training_job(job_id)
        path = Path(job["log_path"] or "")
        if not await asyncio.to_thread(path.is_file):
            return {"lines": [], "offset": offset, "next_offset": offset, "complete": True}
        content = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        lines = content.splitlines()
        selected = lines[offset : offset + limit]
        next_offset = offset + len(selected)
        return {
            "lines": selected,
            "offset": offset,
            "next_offset": next_offset,
            "complete": next_offset >= len(lines),
        }

    async def wait(self, job_id: str) -> None:
        async with self._lock:
            task = self._tasks.get(job_id)
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def shutdown(self) -> None:
        async with self._lock:
            tasks = tuple(self._tasks.values())
            processes = tuple(self._processes.values())
        for process in processes:
            await self._terminate_process(process)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_job(self, job_id: str, command: JobCommand, log_path: Path) -> None:
        try:
            async with self._semaphore:
                current = await self.repository.get_training_job(job_id)
                if current["status"] == "cancelled":
                    return
                running = await self.repository.update_training_job(
                    job_id,
                    status="running",
                    stage=command.stage,
                    progress=0.08,
                    command_json=list(command.argv),
                    started_at=utc_now(),
                    error=None,
                )
                await self.hub.publish(
                    _job_channel(job_id),
                    {"type": "status", "job": running},
                )
                process = await asyncio.create_subprocess_exec(
                    *command.argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=PROJECT_ROOT,
                    start_new_session=os.name != "nt",
                )
                async with self._lock:
                    self._processes[job_id] = process
                await self.repository.update_training_job(job_id, pid=process.pid)
                output = await self._stream_output(job_id, process, log_path)
                exit_code = await process.wait()
                current = await self.repository.get_training_job(job_id)
                if current["status"] == "cancelling":
                    cancelled = await self.repository.update_training_job(
                        job_id,
                        status="cancelled",
                        stage="cancelled",
                        progress=current["progress"],
                        exit_code=exit_code,
                        pid=None,
                        error="Cancelled by user.",
                        completed_at=utc_now(),
                    )
                    await self.hub.publish(
                        _job_channel(job_id), {"type": "status", "job": cancelled}
                    )
                    return
                if exit_code != 0:
                    error = self._failure_summary(output, exit_code)
                    failed = await self.repository.update_training_job(
                        job_id,
                        status="failed",
                        stage="failed",
                        exit_code=exit_code,
                        pid=None,
                        error=error,
                        completed_at=utc_now(),
                    )
                    await self.hub.publish(
                        _job_channel(job_id), {"type": "error", "message": error}
                    )
                    await self.hub.publish(_job_channel(job_id), {"type": "status", "job": failed})
                    return
                metrics = self._parse_metrics(output)
                completed = await self.repository.update_training_job(
                    job_id,
                    status="completed",
                    stage="completed",
                    progress=1.0,
                    exit_code=exit_code,
                    pid=None,
                    metrics_json=metrics,
                    completed_at=utc_now(),
                )
                await self.hub.publish(
                    _job_channel(job_id),
                    {"type": "artifact", "path": completed["output_path"]},
                )
                await self.hub.publish(_job_channel(job_id), {"type": "status", "job": completed})
        except asyncio.CancelledError:
            current = await self.repository.get_training_job(job_id)
            if current["status"] in {"queued", "running", "cancelling"}:
                await self.repository.update_training_job(
                    job_id,
                    status="cancelled",
                    stage="cancelled",
                    pid=None,
                    error="Cancelled by user.",
                    completed_at=utc_now(),
                )
            raise
        except Exception as error:
            failed = await self.repository.update_training_job(
                job_id,
                status="failed",
                stage="failed",
                pid=None,
                error=str(error),
                completed_at=utc_now(),
            )
            await self.hub.publish(_job_channel(job_id), {"type": "error", "message": str(error)})
            await self.hub.publish(_job_channel(job_id), {"type": "status", "job": failed})
        finally:
            async with self._lock:
                self._processes.pop(job_id, None)

    async def _stream_output(
        self,
        job_id: str,
        process: asyncio.subprocess.Process,
        log_path: Path,
    ) -> str:
        if process.stdout is None:
            return ""
        chunks: list[str] = []
        line_count = 0
        with log_path.open("a", encoding="utf-8") as log_file:
            while line := await process.stdout.readline():
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                chunks.append(text)
                log_file.write(text + "\n")
                log_file.flush()
                line_count += 1
                await self.hub.publish(
                    _job_channel(job_id),
                    {"type": "log", "line": text, "line_number": line_count},
                )
                if line_count % 10 == 0:
                    current = await self.repository.get_training_job(job_id)
                    progress = min(0.9, max(current["progress"], 0.08) + 0.02)
                    await self.repository.update_training_job(job_id, progress=progress)
        return "\n".join(chunks)

    @staticmethod
    async def _terminate_process(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5)
        except ProcessLookupError:
            return
        except TimeoutError:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            await process.wait()

    @staticmethod
    def _parse_metrics(output: str) -> dict[str, Any] | None:
        stripped = output.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else {"result": parsed}

    @staticmethod
    def _failure_summary(output: str, exit_code: int) -> str:
        lines = [line for line in output.splitlines() if line.strip()]
        tail = "\n".join(lines[-20:])
        return f"Training process exited with code {exit_code}." + (f"\n{tail}" if tail else "")

    def _task_finished(self, job_id: str, task: asyncio.Task[None]) -> None:
        del task
        asyncio.create_task(self._remove_task(job_id))

    async def _remove_task(self, job_id: str) -> None:
        async with self._lock:
            self._tasks.pop(job_id, None)

    @staticmethod
    def _new_job_id() -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"job-{timestamp}-{uuid4().hex[:8]}"
