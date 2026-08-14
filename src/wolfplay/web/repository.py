from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from wolfplay.models import GameEvent

from .tables import AgentProfileRow, GameEventRow, GameRow, TrainingJobRow, utc_now

ACTIVE_GAME_STATUSES = ("queued", "running")
ACTIVE_JOB_STATUSES = ("queued", "running", "cancelling")


class NotFoundError(LookupError):
    pass


class ConflictError(RuntimeError):
    pass


class StudioRepository:
    def __init__(self, sessions: async_sessionmaker) -> None:
        self.sessions = sessions

    async def initialize_defaults(self) -> None:
        async with self.sessions() as session, session.begin():
            heuristic = await session.get(AgentProfileRow, "heuristic")
            if heuristic is None:
                session.add(
                    AgentProfileRow(
                        id="heuristic",
                        name="Heuristic Core",
                        kind="heuristic",
                        enabled=True,
                        builtin=True,
                    )
                )

    async def recover_interrupted(self) -> dict[str, int]:
        now = utc_now()
        async with self.sessions() as session, session.begin():
            games = await session.execute(
                update(GameRow)
                .where(GameRow.status.in_(ACTIVE_GAME_STATUSES))
                .values(
                    status="interrupted",
                    error="Studio process stopped before the game finished.",
                    completed_at=now,
                )
            )
            jobs = await session.execute(
                update(TrainingJobRow)
                .where(TrainingJobRow.status.in_(ACTIVE_JOB_STATUSES))
                .values(
                    status="interrupted",
                    stage="interrupted",
                    error="Studio process stopped before the training job finished.",
                    completed_at=now,
                    pid=None,
                )
            )
        return {"games": games.rowcount or 0, "jobs": jobs.rowcount or 0}

    async def create_game(
        self,
        *,
        game_id: str,
        seed: int,
        max_rounds: int,
        config: dict[str, Any],
        players: dict[str, Any],
    ) -> dict[str, Any]:
        row = GameRow(
            id=game_id,
            seed=seed,
            max_rounds=max_rounds,
            config_json=deepcopy(config),
            players_json=deepcopy(players),
        )
        async with self.sessions() as session, session.begin():
            session.add(row)
        return self._game_dict(row, include_result=True)

    async def mark_game_running(self, game_id: str) -> dict[str, Any]:
        async with self.sessions() as session, session.begin():
            row = await self._require_game(session, game_id)
            if row.status != "queued":
                raise ConflictError(f"game {game_id} is not queued")
            row.status = "running"
            row.started_at = utc_now()
        return self._game_dict(row, include_result=True)

    async def append_game_event(self, game_id: str, event: GameEvent) -> dict[str, Any]:
        event_data = event.to_dict()
        async with self.sessions() as session, session.begin():
            game = await self._require_game(session, game_id)
            session.add(
                GameEventRow(
                    game_id=game_id,
                    logical_time=event.logical_time,
                    topic=event.topic,
                    round_no=event.round_no,
                    phase=event.phase.value,
                    sender=event.sender,
                    payload_json=deepcopy(event.payload),
                    audience_json=list(event.audience) if event.audience is not None else None,
                    is_public=event.audience is None,
                )
            )
            game.current_round = event.round_no
            game.current_phase = event.phase.value
            game.event_count += 1
            players = deepcopy(game.players_json)
            eliminated_id = None
            if event.topic == "night_result":
                eliminated_id = event.payload.get("victim_id")
            elif event.topic == "vote_result":
                eliminated_id = event.payload.get("eliminated_id")
            if eliminated_id and eliminated_id in players:
                players[eliminated_id]["alive"] = False
                game.players_json = players
            if event.topic == "game_over":
                game.winner = event.payload.get("winner")
                game.termination_reason = event.payload.get("reason")
        return event_data

    async def complete_game(self, game_id: str, result: dict[str, Any]) -> dict[str, Any]:
        async with self.sessions() as session, session.begin():
            row = await self._require_game(session, game_id)
            row.status = "completed"
            row.winner = result["winner"]
            row.termination_reason = result["termination_reason"]
            row.rounds = int(result["rounds"])
            row.current_round = int(result["rounds"])
            row.current_phase = "game_over"
            row.players_json = deepcopy(result["players"])
            row.result_json = deepcopy(result)
            row.completed_at = utc_now()
            row.error = None
        return self._game_dict(row, include_result=True)

    async def fail_game(self, game_id: str, error: str, *, status: str = "failed") -> None:
        async with self.sessions() as session, session.begin():
            row = await self._require_game(session, game_id)
            if row.status == "completed":
                return
            row.status = status
            row.error = error
            row.completed_at = utc_now()

    async def get_game(self, game_id: str, *, include_result: bool = True) -> dict[str, Any]:
        async with self.sessions() as session:
            row = await self._require_game(session, game_id)
            return self._game_dict(row, include_result=include_result)

    async def list_games(
        self,
        *,
        limit: int = 30,
        offset: int = 0,
        status: str | None = None,
        winner: str | None = None,
    ) -> dict[str, Any]:
        conditions = []
        if status:
            conditions.append(GameRow.status == status)
        if winner:
            conditions.append(GameRow.winner == winner)
        async with self.sessions() as session:
            total = await session.scalar(select(func.count(GameRow.id)).where(*conditions))
            rows = (
                await session.scalars(
                    select(GameRow)
                    .where(*conditions)
                    .order_by(GameRow.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
            ).all()
        return {
            "items": [self._game_dict(row, include_result=False) for row in rows],
            "total": total or 0,
            "limit": limit,
            "offset": offset,
        }

    async def list_game_events(
        self,
        game_id: str,
        *,
        public_only: bool,
        after: int = 0,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        async with self.sessions() as session:
            game = await session.get(GameRow, game_id)
            if game is None:
                raise NotFoundError(f"game not found: {game_id}")
            conditions = [
                GameEventRow.game_id == game_id,
                GameEventRow.logical_time > after,
            ]
            if public_only:
                conditions.append(GameEventRow.is_public.is_(True))
            rows = (
                await session.scalars(
                    select(GameEventRow)
                    .where(*conditions)
                    .order_by(GameEventRow.logical_time)
                    .limit(limit)
                )
            ).all()
        return [self._event_dict(row) for row in rows]

    async def create_agent(self, values: dict[str, Any]) -> dict[str, Any]:
        row = AgentProfileRow(id=f"agent-{uuid4().hex[:12]}", **deepcopy(values))
        async with self.sessions() as session, session.begin():
            session.add(row)
        return self._agent_dict(row)

    async def get_agent(self, agent_id: str) -> dict[str, Any]:
        async with self.sessions() as session:
            row = await session.get(AgentProfileRow, agent_id)
            if row is None:
                raise NotFoundError(f"agent not found: {agent_id}")
            return self._agent_dict(row)

    async def list_agents(self) -> list[dict[str, Any]]:
        async with self.sessions() as session:
            rows = (
                await session.scalars(
                    select(AgentProfileRow).order_by(
                        AgentProfileRow.builtin.desc(), AgentProfileRow.name
                    )
                )
            ).all()
        return [self._agent_dict(row) for row in rows]

    async def update_agent(self, agent_id: str, values: dict[str, Any]) -> dict[str, Any]:
        async with self.sessions() as session, session.begin():
            row = await session.get(AgentProfileRow, agent_id)
            if row is None:
                raise NotFoundError(f"agent not found: {agent_id}")
            if row.builtin:
                allowed = {"enabled"}
                if set(values) - allowed:
                    raise ConflictError("built-in agent fields cannot be changed")
            for name, value in values.items():
                setattr(row, name, value)
            row.updated_at = utc_now()
        return self._agent_dict(row)

    async def delete_agent(self, agent_id: str) -> None:
        async with self.sessions() as session, session.begin():
            row = await session.get(AgentProfileRow, agent_id)
            if row is None:
                raise NotFoundError(f"agent not found: {agent_id}")
            if row.builtin:
                raise ConflictError("built-in agents cannot be deleted")
            await session.delete(row)

    async def create_training_job(
        self,
        *,
        job_id: str,
        kind: str,
        config: dict[str, Any],
        output_path: str,
        log_path: str,
    ) -> dict[str, Any]:
        row = TrainingJobRow(
            id=job_id,
            kind=kind,
            config_json=deepcopy(config),
            output_path=output_path,
            log_path=log_path,
        )
        async with self.sessions() as session, session.begin():
            session.add(row)
        return self._job_dict(row)

    async def update_training_job(self, job_id: str, **values: Any) -> dict[str, Any]:
        async with self.sessions() as session, session.begin():
            row = await self._require_job(session, job_id)
            for name, value in values.items():
                if not hasattr(row, name):
                    raise ValueError(f"unknown training job field: {name}")
                setattr(row, name, deepcopy(value))
        return self._job_dict(row)

    async def get_training_job(self, job_id: str) -> dict[str, Any]:
        async with self.sessions() as session:
            row = await self._require_job(session, job_id)
            return self._job_dict(row)

    async def list_training_jobs(
        self,
        *,
        limit: int = 30,
        offset: int = 0,
        status: str | None = None,
        kind: str | None = None,
    ) -> dict[str, Any]:
        conditions = []
        if status:
            conditions.append(TrainingJobRow.status == status)
        if kind:
            conditions.append(TrainingJobRow.kind == kind)
        async with self.sessions() as session:
            total = await session.scalar(select(func.count(TrainingJobRow.id)).where(*conditions))
            rows = (
                await session.scalars(
                    select(TrainingJobRow)
                    .where(*conditions)
                    .order_by(TrainingJobRow.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
            ).all()
        return {
            "items": [self._job_dict(row) for row in rows],
            "total": total or 0,
            "limit": limit,
            "offset": offset,
        }

    async def analytics_overview(self) -> dict[str, Any]:
        async with self.sessions() as session:
            games = (
                await session.scalars(
                    select(GameRow)
                    .where(GameRow.status == "completed")
                    .order_by(GameRow.created_at)
                )
            ).all()
            active_games = await session.scalar(
                select(func.count(GameRow.id)).where(GameRow.status.in_(ACTIVE_GAME_STATUSES))
            )
            active_jobs = await session.scalar(
                select(func.count(TrainingJobRow.id)).where(
                    TrainingJobRow.status.in_(ACTIVE_JOB_STATUSES)
                )
            )
        winners = Counter(row.winner or "unknown" for row in games)
        reasons = Counter(row.termination_reason or "unknown" for row in games)
        rounds = [row.rounds or row.current_round for row in games]
        strategies: Counter[str] = Counter()
        role_games: dict[str, int] = defaultdict(int)
        role_survivals: dict[str, int] = defaultdict(int)
        reflections = 0
        decisions = 0
        illegal_evaluations = 0
        evaluations = 0
        for row in games:
            result = row.result_json or {}
            for player in result.get("players", {}).values():
                role = player.get("role", "unknown")
                role_games[role] += 1
                if player.get("alive"):
                    role_survivals[role] += 1
            for trace in result.get("decision_traces", []):
                decisions += 1
                if trace.get("reflection"):
                    reflections += 1
                action = trace.get("action", {})
                if action.get("strategy"):
                    strategies[action["strategy"]] += 1
                for evaluation in trace.get("evaluations", []):
                    evaluations += 1
                    if not evaluation.get("legal", True):
                        illegal_evaluations += 1
        role_stats = [
            {
                "role": role,
                "appearances": count,
                "survivals": role_survivals[role],
                "survival_rate": role_survivals[role] / count if count else 0.0,
            }
            for role, count in sorted(role_games.items())
        ]
        return {
            "completed_games": len(games),
            "active_games": active_games or 0,
            "active_jobs": active_jobs or 0,
            "average_rounds": sum(rounds) / len(rounds) if rounds else 0.0,
            "winner_counts": dict(winners),
            "termination_reasons": dict(reasons),
            "reflection_rate": reflections / decisions if decisions else 0.0,
            "illegal_evaluation_rate": illegal_evaluations / evaluations if evaluations else 0.0,
            "role_stats": role_stats,
            "strategy_distribution": [
                {"strategy": name, "count": count} for name, count in strategies.most_common(12)
            ],
        }

    async def analytics_timeseries(self, *, days: int = 30) -> list[dict[str, Any]]:
        async with self.sessions() as session:
            rows = (
                await session.scalars(
                    select(GameRow)
                    .where(GameRow.status == "completed")
                    .order_by(GameRow.completed_at.desc())
                )
            ).all()
        buckets: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            timestamp = _as_utc(row.completed_at or row.created_at)
            day = timestamp.astimezone(UTC).date().isoformat()
            bucket = buckets[day]
            bucket["games"] += 1
            bucket[row.winner or "unknown"] += 1
            bucket["rounds"] += row.rounds or 0
        return [
            {
                "date": day,
                "games": values["games"],
                "werewolf": values["werewolf"],
                "village": values["village"],
                "draw": values["draw"],
                "average_rounds": values["rounds"] / values["games"],
            }
            for day, values in sorted(buckets.items())[-days:]
        ]

    async def clear_all(self) -> None:
        async with self.sessions() as session, session.begin():
            await session.execute(delete(GameEventRow))
            await session.execute(delete(GameRow))
            await session.execute(delete(TrainingJobRow))

    @staticmethod
    async def _require_game(session, game_id: str) -> GameRow:
        row = await session.get(GameRow, game_id)
        if row is None:
            raise NotFoundError(f"game not found: {game_id}")
        return row

    @staticmethod
    async def _require_job(session, job_id: str) -> TrainingJobRow:
        row = await session.get(TrainingJobRow, job_id)
        if row is None:
            raise NotFoundError(f"training job not found: {job_id}")
        return row

    @staticmethod
    def _game_dict(row: GameRow, *, include_result: bool) -> dict[str, Any]:
        record = {
            "id": row.id,
            "seed": row.seed,
            "max_rounds": row.max_rounds,
            "status": row.status,
            "winner": row.winner,
            "termination_reason": row.termination_reason,
            "rounds": row.rounds,
            "current_round": row.current_round,
            "current_phase": row.current_phase,
            "event_count": row.event_count,
            "config": deepcopy(row.config_json),
            "players": deepcopy(row.players_json),
            "error": row.error,
            "created_at": _as_utc(row.created_at),
            "started_at": _as_utc(row.started_at),
            "completed_at": _as_utc(row.completed_at),
        }
        if include_result:
            record["result"] = deepcopy(row.result_json)
        return record

    @staticmethod
    def _event_dict(row: GameEventRow) -> dict[str, Any]:
        return {
            "logical_time": row.logical_time,
            "topic": row.topic,
            "round_no": row.round_no,
            "phase": row.phase,
            "payload": deepcopy(row.payload_json),
            "sender": row.sender,
            "audience": deepcopy(row.audience_json),
            "is_public": row.is_public,
            "created_at": _as_utc(row.created_at),
        }

    @staticmethod
    def _agent_dict(row: AgentProfileRow) -> dict[str, Any]:
        return {
            "id": row.id,
            "name": row.name,
            "kind": row.kind,
            "model": row.model,
            "base_url": row.base_url,
            "env_prefix": row.env_prefix,
            "temperature": row.temperature,
            "timeout_seconds": row.timeout_seconds,
            "enabled": row.enabled,
            "builtin": row.builtin,
            "created_at": _as_utc(row.created_at),
            "updated_at": _as_utc(row.updated_at),
        }

    @staticmethod
    def _job_dict(row: TrainingJobRow) -> dict[str, Any]:
        return {
            "id": row.id,
            "kind": row.kind,
            "status": row.status,
            "stage": row.stage,
            "progress": row.progress,
            "config": deepcopy(row.config_json),
            "command": deepcopy(row.command_json),
            "metrics": deepcopy(row.metrics_json),
            "output_path": row.output_path,
            "log_path": row.log_path,
            "pid": row.pid,
            "exit_code": row.exit_code,
            "error": row.error,
            "created_at": _as_utc(row.created_at),
            "started_at": _as_utc(row.started_at),
            "completed_at": _as_utc(row.completed_at),
        }


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
