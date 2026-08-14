from __future__ import annotations

from fastapi import APIRouter, Query

from ..dependencies import RepositoryDep

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/overview")
async def overview(repository: RepositoryDep) -> dict:
    return await repository.analytics_overview()


@router.get("/timeseries")
async def timeseries(
    repository: RepositoryDep,
    days: int = Query(default=30, ge=1, le=3650),
) -> list[dict]:
    return await repository.analytics_timeseries(days=days)
