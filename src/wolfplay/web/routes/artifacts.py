from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("")
async def list_artifacts(
    request: Request,
    query: str | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    root: Path = request.app.state.settings.artifact_dir.resolve()
    items = await asyncio.to_thread(_collect_artifacts, root, query, limit)
    return {"items": items, "total": len(items)}


def _collect_artifacts(root: Path, query: str | None, limit: int) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    normalized_query = query.casefold() if query else None
    items = []
    for path in sorted(root.rglob("*"), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if normalized_query and normalized_query not in relative.casefold():
            continue
        stat = path.stat()
        items.append(
            {
                "path": relative,
                "name": path.name,
                "extension": path.suffix.lower(),
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC),
                "category": _category(path),
            }
        )
        if len(items) >= limit:
            break
    return items


@router.get("/{artifact_path:path}/download")
async def download_artifact(request: Request, artifact_path: str) -> FileResponse:
    path = request.app.state.settings.resolve_artifact_path(artifact_path)
    if not await asyncio.to_thread(path.is_file):
        raise ValueError("artifact does not exist or is not a file")
    return FileResponse(path, filename=path.name)


def _category(path: Path) -> str:
    name = path.name.casefold()
    if name.endswith(".jsonl") and "dpo" in name:
        return "preference_data"
    if name.endswith(".jsonl"):
        return "self_play"
    if "latent" in name:
        return "latent_space"
    if path.suffix.lower() in {".pt", ".pth", ".safetensors"}:
        return "checkpoint"
    if "manifest" in name:
        return "manifest"
    if path.suffix.lower() == ".log":
        return "log"
    return "other"
