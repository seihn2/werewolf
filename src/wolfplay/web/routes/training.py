from __future__ import annotations

from fastapi import APIRouter, Query, status

from ..dependencies import RepositoryDep, TrainingManagerDep
from ..schemas import TrainingJobCreate, TrainingJobResponse

router = APIRouter(prefix="/api/training/jobs", tags=["training"])


@router.get("")
async def list_jobs(
    repository: RepositoryDep,
    limit: int = Query(default=30, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    job_status: str | None = Query(default=None, alias="status"),
    kind: str | None = None,
) -> dict:
    return await repository.list_training_jobs(
        limit=limit,
        offset=offset,
        status=job_status,
        kind=kind,
    )


@router.post("", response_model=TrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    request: TrainingJobCreate,
    manager: TrainingManagerDep,
) -> dict:
    return await manager.create_job(
        kind=request.kind,
        config=request.config.model_dump(mode="json"),
    )


@router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_job(
    job_id: str,
    repository: RepositoryDep,
) -> dict:
    return await repository.get_training_job(job_id)


@router.get("/{job_id}/logs")
async def get_logs(
    job_id: str,
    manager: TrainingManagerDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    return await manager.read_logs(job_id, offset=offset, limit=limit)


@router.post("/{job_id}/cancel", response_model=TrainingJobResponse)
async def cancel_job(
    job_id: str,
    manager: TrainingManagerDep,
) -> dict:
    return await manager.cancel_job(job_id)
