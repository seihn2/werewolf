from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from .tables import Base


class Database:
    def __init__(self, url: str) -> None:
        options: dict = {"pool_pre_ping": True}
        if url.endswith(":memory:"):
            options["poolclass"] = StaticPool
        self.engine: AsyncEngine = create_async_engine(url, **options)
        self.sessions = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        if url.startswith("sqlite"):
            self._configure_sqlite()

    def _configure_sqlite(self) -> None:
        @event.listens_for(self.engine.sync_engine, "connect")
        def set_sqlite_pragmas(dbapi_connection, connection_record) -> None:
            del connection_record
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
            except Exception:
                pass
            cursor.close()

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()
