# ruff: noqa: ASYNC240

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .abstract_game import AbstractGameConfig
from .cfr_preference import build_cfr_dpo_dataset
from .latent import LatentStrategySpace, TextEmbedder, build_latent_strategy_file
from .llm import ChatBackend
from .models import Role
from .self_play import run_self_play
from .training.deep_cfr import DeepCFRConfig, train_deep_cfr
from .training.dpo import DPOTrainingConfig, train_dpo

BackendFactory = Callable[[int, Path | None], ChatBackend | None]
DPOConfigFactory = Callable[[int, Path, Path, Path | None], DPOTrainingConfig]


@dataclass(frozen=True, slots=True)
class IterativePolicyConfig:
    iterations: int = 1
    games_per_iteration: int = 100
    concurrency: int = 1
    seed: int = 2025
    max_rounds: int = 8
    base_clusters: dict[Role, int] = field(
        default_factory=lambda: {
            Role.WEREWOLF: 3,
            Role.SEER: 2,
            Role.DOCTOR: 2,
            Role.VILLAGER: 2,
        }
    )
    clusters_added_per_iteration: int = 1
    deep_cfr: DeepCFRConfig = field(default_factory=DeepCFRConfig)
    winning_only_dpo: bool = False
    resume: bool = True

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.games_per_iteration <= 0:
            raise ValueError("games_per_iteration must be positive")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.clusters_added_per_iteration < 0:
            raise ValueError("clusters_added_per_iteration must be non-negative")
        if any(count <= 0 for count in self.base_clusters.values()):
            raise ValueError("base_clusters values must be positive")


@dataclass(frozen=True, slots=True)
class IterationArtifacts:
    iteration: int
    directory: Path
    self_play: Path
    latent_space: Path
    deep_cfr_checkpoint: Path
    dpo_dataset: Path
    dpo_checkpoint: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "directory": str(self.directory),
            "self_play": str(self.self_play),
            "latent_space": str(self.latent_space),
            "deep_cfr_checkpoint": str(self.deep_cfr_checkpoint),
            "dpo_dataset": str(self.dpo_dataset),
            "dpo_checkpoint": str(self.dpo_checkpoint) if self.dpo_checkpoint else None,
        }


class IterativePolicyOptimizer:
    """Runs repeated sampling, clustering, Deep CFR, DPO data and optional DPO training."""

    def __init__(
        self,
        *,
        config: IterativePolicyConfig,
        embedder: TextEmbedder,
        backend_factory: BackendFactory,
        dpo_config_factory: DPOConfigFactory | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.backend_factory = backend_factory
        self.dpo_config_factory = dpo_config_factory

    async def run(self, output_dir: Path) -> list[IterationArtifacts]:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[IterationArtifacts] = []
        previous_dpo_checkpoint: Path | None = None
        for iteration in range(1, self.config.iterations + 1):
            iteration_dir = output_dir / f"iteration_{iteration:02d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            self_play_path = iteration_dir / "self_play.jsonl"
            latent_path = iteration_dir / "latent_space.json"
            deep_cfr_dir = iteration_dir / "deep_cfr"
            deep_cfr_checkpoint = deep_cfr_dir / "deep_cfr.pt"
            dpo_path = iteration_dir / "dpo_cfr.jsonl"
            dpo_checkpoint = iteration_dir / "dpo" / "final"

            if not (self.config.resume and self_play_path.is_file()):
                backend = self.backend_factory(iteration, previous_dpo_checkpoint)
                try:
                    await run_self_play(
                        games=self.config.games_per_iteration,
                        seed=self.config.seed + (iteration - 1) * 100_000,
                        max_rounds=self.config.max_rounds,
                        output_path=self_play_path,
                        backend=backend,
                        concurrency=self.config.concurrency,
                    )
                finally:
                    close = getattr(backend, "aclose", None)
                    if close is not None:
                        await close()

            clusters = {
                role: base + (iteration - 1) * self.config.clusters_added_per_iteration
                for role, base in self.config.base_clusters.items()
            }
            if not (self.config.resume and latent_path.is_file()):
                build_latent_strategy_file(
                    input_path=self_play_path,
                    output_path=latent_path,
                    embedder=self.embedder,
                    clusters_by_role=clusters,
                    seed=self.config.seed + iteration,
                )
            latent_space = LatentStrategySpace.load(latent_path)

            if not (self.config.resume and deep_cfr_checkpoint.is_file()):
                cfr_config = replace(
                    self.config.deep_cfr,
                    seed=self.config.deep_cfr.seed + iteration - 1,
                )
                train_deep_cfr(
                    latent_space=latent_space,
                    output_dir=deep_cfr_dir,
                    config=cfr_config,
                    game_config=AbstractGameConfig(
                        max_rounds=self.config.max_rounds,
                        role_assignment_seed=self.config.seed + iteration,
                    ),
                )

            if not (self.config.resume and dpo_path.is_file()):
                build_cfr_dpo_dataset(
                    input_path=self_play_path,
                    checkpoint=deep_cfr_checkpoint,
                    output_path=dpo_path,
                    embedder=self.embedder,
                    winning_only=self.config.winning_only_dpo,
                )

            resolved_dpo_checkpoint: Path | None = None
            if self.dpo_config_factory is not None:
                if not (self.config.resume and dpo_checkpoint.is_dir()):
                    dpo_config = self.dpo_config_factory(
                        iteration,
                        dpo_path,
                        iteration_dir / "dpo",
                        previous_dpo_checkpoint,
                    )
                    train_dpo(dpo_config)
                resolved_dpo_checkpoint = dpo_checkpoint
                previous_dpo_checkpoint = dpo_checkpoint

            current = IterationArtifacts(
                iteration=iteration,
                directory=iteration_dir,
                self_play=self_play_path,
                latent_space=latent_path,
                deep_cfr_checkpoint=deep_cfr_checkpoint,
                dpo_dataset=dpo_path,
                dpo_checkpoint=resolved_dpo_checkpoint,
            )
            artifacts.append(current)
            _atomic_write_json(
                iteration_dir / "iteration_manifest.json",
                {
                    "config": _config_to_dict(self.config),
                    "clusters": {role.value: count for role, count in clusters.items()},
                    "artifacts": current.to_dict(),
                },
            )
        _atomic_write_json(
            output_dir / "iterations_manifest.json",
            {"iterations": [artifact.to_dict() for artifact in artifacts]},
        )
        return artifacts


def _config_to_dict(config: IterativePolicyConfig) -> dict[str, Any]:
    record = asdict(config)
    record["base_clusters"] = {
        role.value if isinstance(role, Role) else str(role): value
        for role, value in config.base_clusters.items()
    }
    record["deep_cfr"] = config.deep_cfr.to_dict()
    return record


def _atomic_write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
