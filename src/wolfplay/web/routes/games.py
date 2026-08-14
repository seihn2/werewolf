from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query, status

from ..dependencies import GameManagerDep, RepositoryDep
from ..repository import ConflictError
from ..schemas import GameCreate, GameEventResponse, GameResponse

router = APIRouter(prefix="/api/games", tags=["games"])


@router.get("")
async def list_games(
    repository: RepositoryDep,
    limit: int = Query(default=30, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    game_status: str | None = Query(default=None, alias="status"),
    winner: str | None = None,
) -> dict:
    return await repository.list_games(
        limit=limit,
        offset=offset,
        status=game_status,
        winner=winner,
    )


@router.post("", response_model=GameResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_game(
    request: GameCreate,
    manager: GameManagerDep,
) -> dict:
    return await manager.create_game(request.model_dump())


@router.get("/{game_id}", response_model=GameResponse)
async def get_game(
    game_id: str,
    repository: RepositoryDep,
) -> dict:
    return await repository.get_game(game_id)


@router.get("/{game_id}/events", response_model=list[GameEventResponse])
async def get_events(
    game_id: str,
    repository: RepositoryDep,
    view: Literal["public", "omniscient"] = "public",
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=2000, ge=1, le=10_000),
) -> list[dict]:
    game = await repository.get_game(game_id, include_result=False)
    if view == "omniscient" and game["status"] != "completed":
        raise ConflictError("omniscient events are available only after a game completes")
    return await repository.list_game_events(
        game_id,
        public_only=view == "public",
        after=after,
        limit=limit,
    )


@router.post("/{game_id}/cancel", response_model=GameResponse)
async def cancel_game(
    game_id: str,
    manager: GameManagerDep,
) -> dict:
    return await manager.cancel_game(game_id)
