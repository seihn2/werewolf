from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from .game_manager import GameManager
from .realtime import RealtimeHub
from .repository import StudioRepository
from .training import TrainingManager


def get_repository(request: Request) -> StudioRepository:
    return request.app.state.repository


def get_game_manager(request: Request) -> GameManager:
    return request.app.state.game_manager


def get_training_manager(request: Request) -> TrainingManager:
    return request.app.state.training_manager


def get_realtime_hub(request: Request) -> RealtimeHub:
    return request.app.state.realtime_hub


RepositoryDep = Annotated[StudioRepository, Depends(get_repository)]
GameManagerDep = Annotated[GameManager, Depends(get_game_manager)]
TrainingManagerDep = Annotated[TrainingManager, Depends(get_training_manager)]
RealtimeHubDep = Annotated[RealtimeHub, Depends(get_realtime_hub)]
