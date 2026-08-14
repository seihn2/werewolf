from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GameCreate(APIModel):
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    max_rounds: int = Field(default=8, ge=1, le=50)
    pace_seconds: float = Field(default=0.35, ge=0, le=5)
    werewolf_agent_id: str = Field(default="heuristic", min_length=1, max_length=64)
    village_agent_id: str = Field(default="heuristic", min_length=1, max_length=64)
    label: str | None = Field(default=None, max_length=120)


class GameResponse(BaseModel):
    id: str
    seed: int
    max_rounds: int
    status: str
    winner: str | None
    termination_reason: str | None
    rounds: int | None
    current_round: int
    current_phase: str
    event_count: int
    config: dict[str, Any]
    players: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class GameEventResponse(BaseModel):
    logical_time: int
    topic: str
    round_no: int
    phase: str
    payload: dict[str, Any]
    sender: str | None
    audience: list[str] | None
    is_public: bool
    created_at: datetime


class AgentBase(APIModel):
    name: str = Field(min_length=2, max_length=100)
    kind: Literal["heuristic", "openai_compatible"]
    model: str | None = Field(default=None, max_length=200)
    base_url: HttpUrl | None = None
    env_prefix: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    temperature: float = Field(default=0.7, ge=0, le=2)
    timeout_seconds: float = Field(default=90.0, gt=0, le=600)
    enabled: bool = True

    @model_validator(mode="after")
    def validate_backend_configuration(self):
        if self.kind == "openai_compatible" and not self.env_prefix:
            raise ValueError("openai_compatible agents require env_prefix")
        return self


class AgentCreate(AgentBase):
    pass


class AgentUpdate(APIModel):
    name: str | None = Field(default=None, min_length=2, max_length=100)
    kind: Literal["heuristic", "openai_compatible"] | None = None
    model: str | None = Field(default=None, max_length=200)
    base_url: HttpUrl | None = None
    env_prefix: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    temperature: float | None = Field(default=None, ge=0, le=2)
    timeout_seconds: float | None = Field(default=None, gt=0, le=600)
    enabled: bool | None = None


class AgentResponse(BaseModel):
    id: str
    name: str
    kind: str
    model: str | None
    base_url: str | None
    env_prefix: str | None
    temperature: float
    timeout_seconds: float
    enabled: bool
    builtin: bool
    created_at: datetime
    updated_at: datetime


class SelfPlayConfig(APIModel):
    games: int = Field(default=20, ge=1, le=100_000)
    concurrency: int = Field(default=2, ge=1, le=128)
    seed: int = Field(default=2025, ge=0)
    max_rounds: int = Field(default=8, ge=1, le=50)
    backend: Literal["heuristic", "openai-compatible"] = "heuristic"


class LatentConfig(APIModel):
    input_path: str = Field(min_length=1)
    embedding_backend: Literal["hashing", "openai-compatible"] = "hashing"
    hash_dimensions: int = Field(default=256, ge=8, le=8192)
    werewolf_clusters: int = Field(default=3, ge=1, le=128)
    seer_clusters: int = Field(default=2, ge=1, le=128)
    doctor_clusters: int = Field(default=2, ge=1, le=128)
    villager_clusters: int = Field(default=2, ge=1, le=128)
    seed: int = Field(default=42, ge=0)


class DeepCFRJobConfig(APIModel):
    latent_space_path: str = Field(min_length=1)
    iterations: int = Field(default=10, ge=1, le=100_000)
    traversals_per_player: int = Field(default=4, ge=1, le=100_000)
    advantage_train_steps: int = Field(default=100, ge=0)
    strategy_train_steps: int = Field(default=200, ge=0)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    hidden_sizes: list[int] = Field(default_factory=lambda: [256, 256, 256], min_length=1)
    max_traversal_depth: int = Field(default=64, ge=1)
    max_rollout_steps: int = Field(default=512, ge=1)
    device: str = Field(default="auto", min_length=1, max_length=32)
    checkpoint_every: int = Field(default=1, ge=1)
    seed: int = Field(default=42, ge=0)
    max_rounds: int = Field(default=8, ge=1, le=50)
    no_save_buffers: bool = False


class CFRDPOConfig(APIModel):
    input_path: str = Field(min_length=1)
    checkpoint_path: str = Field(min_length=1)
    embedding_backend: Literal["hashing", "openai-compatible"] = "hashing"
    hash_dimensions: int = Field(default=256, ge=8, le=8192)
    device: str = Field(default="cpu", min_length=1, max_length=32)
    winning_only: bool = True


class DPOJobConfig(APIModel):
    dataset_path: str = Field(min_length=1)
    model: str = Field(min_length=1, max_length=500)
    epochs: float = Field(default=2.0, gt=0)
    learning_rate: float = Field(default=1e-6, gt=0)
    beta: float = Field(default=0.1, gt=0)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    max_length: int = Field(default=2048, ge=1)
    use_lora: bool = True
    lora_r: int = Field(default=32, ge=1)
    lora_alpha: int = Field(default=16, ge=1)


class IterativeConfig(APIModel):
    iterations: int = Field(default=2, ge=1, le=1000)
    games_per_iteration: int = Field(default=50, ge=1)
    concurrency: int = Field(default=2, ge=1, le=128)
    seed: int = Field(default=2025, ge=0)
    max_rounds: int = Field(default=8, ge=1, le=50)
    backend: Literal["heuristic", "openai-compatible"] = "heuristic"
    embedding_backend: Literal["hashing", "openai-compatible"] = "hashing"
    hash_dimensions: int = Field(default=256, ge=8, le=8192)
    clusters_added_per_iteration: int = Field(default=1, ge=0, le=128)
    cfr_iterations: int = Field(default=10, ge=1)
    cfr_traversals_per_player: int = Field(default=4, ge=1)
    cfr_advantage_train_steps: int = Field(default=100, ge=0)
    cfr_strategy_train_steps: int = Field(default=200, ge=0)
    no_resume: bool = False
    dpo_model: str | None = Field(default=None, max_length=500)


class SelfPlayJobCreate(APIModel):
    kind: Literal["self_play"]
    config: SelfPlayConfig = Field(default_factory=SelfPlayConfig)


class LatentJobCreate(APIModel):
    kind: Literal["latent"]
    config: LatentConfig


class DeepCFRJobCreate(APIModel):
    kind: Literal["deep_cfr"]
    config: DeepCFRJobConfig


class CFRDPOJobCreate(APIModel):
    kind: Literal["cfr_dpo"]
    config: CFRDPOConfig


class DPOJobCreate(APIModel):
    kind: Literal["dpo"]
    config: DPOJobConfig


class IterativeJobCreate(APIModel):
    kind: Literal["iterative"]
    config: IterativeConfig = Field(default_factory=IterativeConfig)


TrainingJobCreate = Annotated[
    SelfPlayJobCreate
    | LatentJobCreate
    | DeepCFRJobCreate
    | CFRDPOJobCreate
    | DPOJobCreate
    | IterativeJobCreate,
    Field(discriminator="kind"),
]


class TrainingJobResponse(BaseModel):
    id: str
    kind: str
    status: str
    stage: str
    progress: float
    config: dict[str, Any]
    command: list[str] | None
    metrics: dict[str, Any] | None
    output_path: str | None
    log_path: str | None
    pid: int | None
    exit_code: int | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class ErrorResponse(BaseModel):
    error: dict[str, Any]
