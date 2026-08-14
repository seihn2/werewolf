from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..repository import NotFoundError

router = APIRouter(tags=["realtime"])


@router.websocket("/ws/games/{game_id}")
async def game_socket(websocket: WebSocket, game_id: str) -> None:
    await websocket.accept()
    repository = websocket.app.state.repository
    hub = websocket.app.state.realtime_hub
    settings = websocket.app.state.settings
    try:
        async with hub.subscribe(f"game:{game_id}") as queue:
            game = await repository.get_game(game_id)
            events = await repository.list_game_events(game_id, public_only=True)
            await websocket.send_json(
                {"type": "snapshot", "game": _json_ready(game), "events": _json_ready(events)}
            )
            await _stream(websocket, queue, settings.heartbeat_seconds)
    except NotFoundError:
        await websocket.send_json({"type": "error", "message": "game not found"})
        await websocket.close(code=4404)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/training/{job_id}")
async def training_socket(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    repository = websocket.app.state.repository
    manager = websocket.app.state.training_manager
    hub = websocket.app.state.realtime_hub
    settings = websocket.app.state.settings
    try:
        async with hub.subscribe(f"training:{job_id}") as queue:
            job = await repository.get_training_job(job_id)
            logs = await manager.read_logs(job_id, offset=0, limit=200)
            await websocket.send_json(
                {"type": "snapshot", "job": _json_ready(job), "logs": logs["lines"]}
            )
            await _stream(websocket, queue, settings.heartbeat_seconds)
    except NotFoundError:
        await websocket.send_json({"type": "error", "message": "training job not found"})
        await websocket.close(code=4404)
    except WebSocketDisconnect:
        return


async def _stream(websocket: WebSocket, queue: asyncio.Queue, heartbeat_seconds: float) -> None:
    while True:
        try:
            message = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
        except TimeoutError:
            await websocket.send_json({"type": "heartbeat"})
        else:
            await websocket.send_json(_json_ready(message))


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
