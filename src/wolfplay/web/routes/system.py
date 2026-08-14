from __future__ import annotations

from importlib import metadata

from fastapi import APIRouter

from ..dependencies import RepositoryDep

router = APIRouter(tags=["system"])


@router.get("/api/health")
async def health(repository: RepositoryDep) -> dict:
    games = await repository.list_games(limit=1)
    try:
        version = metadata.version("wolfplay")
    except metadata.PackageNotFoundError:
        version = "development"
    return {
        "status": "ok",
        "service": "WolfPlay Studio",
        "version": version,
        "database": "ready",
        "games": games["total"],
    }
