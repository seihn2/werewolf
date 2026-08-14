from __future__ import annotations

from fastapi import APIRouter, Response, status

from ..dependencies import RepositoryDep
from ..schemas import AgentCreate, AgentResponse, AgentUpdate

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("", response_model=list[AgentResponse])
async def list_agents(repository: RepositoryDep) -> list[dict]:
    return await repository.list_agents()


@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: AgentCreate,
    repository: RepositoryDep,
) -> dict:
    values = request.model_dump(mode="json")
    return await repository.create_agent(values)


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    request: AgentUpdate,
    repository: RepositoryDep,
) -> dict:
    values = request.model_dump(mode="json", exclude_unset=True)
    return await repository.update_agent(agent_id, values)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    repository: RepositoryDep,
) -> Response:
    await repository.delete_agent(agent_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
