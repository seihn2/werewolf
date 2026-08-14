from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError

from .config import WebSettings
from .database import Database
from .game_manager import GameManager
from .realtime import RealtimeHub
from .repository import ConflictError, NotFoundError, StudioRepository
from .routes import agents, analytics, artifacts, games, realtime, system, training
from .training import TrainingManager


def create_app(settings: WebSettings | None = None) -> FastAPI:
    resolved_settings = settings or WebSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_settings.ensure_directories()
        database = Database(resolved_settings.database_url)
        await database.initialize()
        repository = StudioRepository(database.sessions)
        await repository.initialize_defaults()
        await repository.recover_interrupted()
        hub = RealtimeHub(queue_size=resolved_settings.realtime_queue_size)
        game_manager = GameManager(
            repository=repository,
            hub=hub,
            settings=resolved_settings,
        )
        training_manager = TrainingManager(
            repository=repository,
            hub=hub,
            settings=resolved_settings,
        )
        app.state.settings = resolved_settings
        app.state.database = database
        app.state.repository = repository
        app.state.realtime_hub = hub
        app.state.game_manager = game_manager
        app.state.training_manager = training_manager
        try:
            yield
        finally:
            await training_manager.shutdown()
            await game_manager.shutdown()
            await database.dispose()

    app = FastAPI(
        title="WolfPlay Studio API",
        version="1.0.0",
        description="Realtime multi-agent Werewolf games, replay, training and analytics.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(system.router)
    app.include_router(games.router)
    app.include_router(agents.router)
    app.include_router(training.router)
    app.include_router(artifacts.router)
    app.include_router(analytics.router)
    app.include_router(realtime.router)
    _register_error_handlers(app)
    _mount_frontend(app, resolved_settings.frontend_dist)
    return app


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, error: NotFoundError) -> JSONResponse:
        del request
        return _error(status.HTTP_404_NOT_FOUND, "not_found", str(error))

    @app.exception_handler(ConflictError)
    async def conflict_handler(request: Request, error: ConflictError) -> JSONResponse:
        del request
        return _error(status.HTTP_409_CONFLICT, "conflict", str(error))

    @app.exception_handler(ValueError)
    async def value_handler(request: Request, error: ValueError) -> JSONResponse:
        del request
        return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_request", str(error))

    @app.exception_handler(IntegrityError)
    async def integrity_handler(request: Request, error: IntegrityError) -> JSONResponse:
        del request, error
        return _error(
            status.HTTP_409_CONFLICT,
            "conflict",
            "A record with the same unique value already exists.",
        )


def _error(status_code: int, code: str, message: str, details: Any = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details}},
    )


def _mount_frontend(app: FastAPI, frontend_dist: Path) -> None:
    dist = frontend_dist.resolve()
    index = dist / "index.html"
    assets = dist / "assets"
    if index.is_file() and assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def serve_spa(path: str):
            candidate = (dist / path).resolve()
            if candidate.is_relative_to(dist) and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(index)
    else:

        @app.get("/", include_in_schema=False)
        async def api_root() -> dict[str, str]:
            return {
                "name": "WolfPlay Studio API",
                "status": "frontend_not_built",
                "docs": "/docs",
            }


def main() -> None:
    settings = WebSettings.from_env()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
