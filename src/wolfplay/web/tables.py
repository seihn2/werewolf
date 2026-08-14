from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class GameRow(Base):
    __tablename__ = "games"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    max_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="queued", index=True)
    winner: Mapped[str | None] = mapped_column(String(24), index=True)
    termination_reason: Mapped[str | None] = mapped_column(String(80))
    rounds: Mapped[int | None] = mapped_column(Integer)
    current_round: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    current_phase: Mapped[str] = mapped_column(String(40), nullable=False, default="setup")
    event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    players_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)


class GameEventRow(Base):
    __tablename__ = "game_events"
    __table_args__ = (
        UniqueConstraint("game_id", "logical_time", name="uq_game_event_logical_time"),
        Index("ix_game_events_game_public_time", "game_id", "is_public", "logical_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("games.id", ondelete="CASCADE"), nullable=False, index=True
    )
    logical_time: Mapped[int] = mapped_column(Integer, nullable=False)
    topic: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    round_no: Mapped[int] = mapped_column(Integer, nullable=False)
    phase: Mapped[str] = mapped_column(String(40), nullable=False)
    sender: Mapped[str | None] = mapped_column(String(64))
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    audience_json: Mapped[list[str] | None] = mapped_column(JSON)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class AgentProfileRow(Base):
    __tablename__ = "agent_profiles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    model: Mapped[str | None] = mapped_column(String(200))
    base_url: Mapped[str | None] = mapped_column(String(500))
    env_prefix: Mapped[str | None] = mapped_column(String(100))
    temperature: Mapped[float] = mapped_column(Float, nullable=False, default=0.7)
    timeout_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=90.0)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    builtin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class TrainingJobRow(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(80), nullable=False, default="queued")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    command_json: Mapped[list[str] | None] = mapped_column(JSON)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    output_path: Mapped[str | None] = mapped_column(String(1000))
    log_path: Mapped[str | None] = mapped_column(String(1000))
    pid: Mapped[int | None] = mapped_column(Integer)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
