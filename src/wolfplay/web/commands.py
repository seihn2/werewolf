from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import WebSettings

SUPPORTED_TRAINING_KINDS = (
    "self_play",
    "latent",
    "deep_cfr",
    "cfr_dpo",
    "dpo",
    "iterative",
)


@dataclass(frozen=True, slots=True)
class JobCommand:
    argv: tuple[str, ...]
    output_path: Path
    stage: str


def build_training_command(
    *,
    kind: str,
    config: dict[str, Any],
    job_dir: Path,
    settings: WebSettings,
) -> JobCommand:
    if kind not in SUPPORTED_TRAINING_KINDS:
        raise ValueError(f"unsupported training job kind: {kind}")
    job_dir.mkdir(parents=True, exist_ok=True)
    prefix = [sys.executable, "-m", "wolfplay"]
    if kind == "self_play":
        output = job_dir / "self_play.jsonl"
        argv = [
            *prefix,
            "self-play",
            "--games",
            str(_integer(config, "games", 20, minimum=1)),
            "--concurrency",
            str(_integer(config, "concurrency", 2, minimum=1)),
            "--seed",
            str(_integer(config, "seed", 2025, minimum=0)),
            "--max-rounds",
            str(_integer(config, "max_rounds", 8, minimum=1)),
            "--backend",
            _choice(config, "backend", "heuristic", {"heuristic", "openai-compatible"}),
            "--output",
            str(output),
        ]
        return JobCommand(tuple(argv), output, "self-play sampling")

    if kind == "latent":
        input_path = _artifact_input(settings, config, "input_path")
        output = job_dir / "latent_space.json"
        argv = [
            *prefix,
            "build-latent",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--embedding-backend",
            _choice(config, "embedding_backend", "hashing", {"hashing", "openai-compatible"}),
            "--hash-dimensions",
            str(_integer(config, "hash_dimensions", 256, minimum=8)),
            "--werewolf-clusters",
            str(_integer(config, "werewolf_clusters", 3, minimum=1)),
            "--seer-clusters",
            str(_integer(config, "seer_clusters", 2, minimum=1)),
            "--doctor-clusters",
            str(_integer(config, "doctor_clusters", 2, minimum=1)),
            "--villager-clusters",
            str(_integer(config, "villager_clusters", 2, minimum=1)),
            "--seed",
            str(_integer(config, "seed", 42, minimum=0)),
        ]
        return JobCommand(tuple(argv), output, "strategy clustering")

    if kind == "deep_cfr":
        latent_space = _artifact_input(settings, config, "latent_space_path")
        output = job_dir / "deep_cfr"
        argv = [
            *prefix,
            "train-deep-cfr",
            "--latent-space",
            str(latent_space),
            "--output-dir",
            str(output),
            "--iterations",
            str(_integer(config, "iterations", 10, minimum=1)),
            "--traversals-per-player",
            str(_integer(config, "traversals_per_player", 4, minimum=1)),
            "--advantage-train-steps",
            str(_integer(config, "advantage_train_steps", 100, minimum=0)),
            "--strategy-train-steps",
            str(_integer(config, "strategy_train_steps", 200, minimum=0)),
            "--batch-size",
            str(_integer(config, "batch_size", 128, minimum=1)),
            "--learning-rate",
            str(_number(config, "learning_rate", 1e-3, minimum=0.0, strict=True)),
            "--max-traversal-depth",
            str(_integer(config, "max_traversal_depth", 64, minimum=1)),
            "--max-rollout-steps",
            str(_integer(config, "max_rollout_steps", 512, minimum=1)),
            "--device",
            str(config.get("device", "auto")),
            "--checkpoint-every",
            str(_integer(config, "checkpoint_every", 1, minimum=1)),
            "--seed",
            str(_integer(config, "seed", 42, minimum=0)),
            "--max-rounds",
            str(_integer(config, "max_rounds", 8, minimum=1)),
        ]
        hidden_sizes = config.get("hidden_sizes", [256, 256, 256])
        if not isinstance(hidden_sizes, list) or not hidden_sizes:
            raise ValueError("hidden_sizes must be a non-empty list")
        argv.extend(["--hidden-sizes", *(str(int(value)) for value in hidden_sizes)])
        if config.get("no_save_buffers"):
            argv.append("--no-save-buffers")
        return JobCommand(tuple(argv), output, "Deep CFR training")

    if kind == "cfr_dpo":
        input_path = _artifact_input(settings, config, "input_path")
        checkpoint = _artifact_input(settings, config, "checkpoint_path")
        output = job_dir / "dpo_cfr.jsonl"
        argv = [
            *prefix,
            "build-cfr-dpo",
            "--input",
            str(input_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--embedding-backend",
            _choice(config, "embedding_backend", "hashing", {"hashing", "openai-compatible"}),
            "--hash-dimensions",
            str(_integer(config, "hash_dimensions", 256, minimum=8)),
            "--device",
            str(config.get("device", "cpu")),
        ]
        if config.get("winning_only", True):
            argv.append("--winning-only")
        return JobCommand(tuple(argv), output, "CFR preference construction")

    if kind == "dpo":
        dataset = _artifact_input(settings, config, "dataset_path")
        model = str(config.get("model", "")).strip()
        if not model:
            raise ValueError("model is required for DPO training")
        output = job_dir / "dpo"
        argv = [
            *prefix,
            "train-dpo",
            "--dataset",
            str(dataset),
            "--model",
            model,
            "--output-dir",
            str(output),
            "--epochs",
            str(_number(config, "epochs", 2.0, minimum=0.0, strict=True)),
            "--learning-rate",
            str(_number(config, "learning_rate", 1e-6, minimum=0.0, strict=True)),
            "--beta",
            str(_number(config, "beta", 0.1, minimum=0.0, strict=True)),
            "--batch-size",
            str(_integer(config, "batch_size", 1, minimum=1)),
            "--gradient-accumulation-steps",
            str(_integer(config, "gradient_accumulation_steps", 16, minimum=1)),
            "--max-length",
            str(_integer(config, "max_length", 2048, minimum=1)),
            "--lora-r",
            str(_integer(config, "lora_r", 32, minimum=1)),
            "--lora-alpha",
            str(_integer(config, "lora_alpha", 16, minimum=1)),
        ]
        if config.get("use_lora", True) is False:
            argv.append("--no-lora")
        return JobCommand(tuple(argv), output, "DPO alignment")

    output = job_dir / "iterations"
    argv = [
        *prefix,
        "iterate-policy",
        "--output-dir",
        str(output),
        "--iterations",
        str(_integer(config, "iterations", 2, minimum=1)),
        "--games-per-iteration",
        str(_integer(config, "games_per_iteration", 50, minimum=1)),
        "--concurrency",
        str(_integer(config, "concurrency", 2, minimum=1)),
        "--seed",
        str(_integer(config, "seed", 2025, minimum=0)),
        "--max-rounds",
        str(_integer(config, "max_rounds", 8, minimum=1)),
        "--backend",
        _choice(config, "backend", "heuristic", {"heuristic", "openai-compatible"}),
        "--embedding-backend",
        _choice(config, "embedding_backend", "hashing", {"hashing", "openai-compatible"}),
        "--hash-dimensions",
        str(_integer(config, "hash_dimensions", 256, minimum=8)),
        "--clusters-added-per-iteration",
        str(_integer(config, "clusters_added_per_iteration", 1, minimum=0)),
        "--cfr-iterations",
        str(_integer(config, "cfr_iterations", 10, minimum=1)),
        "--cfr-traversals-per-player",
        str(_integer(config, "cfr_traversals_per_player", 4, minimum=1)),
        "--cfr-advantage-train-steps",
        str(_integer(config, "cfr_advantage_train_steps", 100, minimum=0)),
        "--cfr-strategy-train-steps",
        str(_integer(config, "cfr_strategy_train_steps", 200, minimum=0)),
    ]
    if config.get("no_resume"):
        argv.append("--no-resume")
    dpo_model = str(config.get("dpo_model", "")).strip()
    if dpo_model:
        argv.extend(["--dpo-model", dpo_model])
    return JobCommand(tuple(argv), output, "iterative policy optimization")


def _artifact_input(settings: WebSettings, config: dict[str, Any], name: str) -> Path:
    raw = config.get(name)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{name} is required")
    path = settings.resolve_artifact_path(raw)
    if not path.exists():
        raise ValueError(f"artifact does not exist: {raw}")
    return path


def _choice(config: dict[str, Any], name: str, default: str, values: set[str]) -> str:
    value = str(config.get(name, default))
    if value not in values:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(values))}")
    return value


def _integer(config: dict[str, Any], name: str, default: int, *, minimum: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _number(
    config: dict[str, Any],
    name: str,
    default: float,
    *,
    minimum: float,
    strict: bool = False,
) -> float:
    value = config.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    parsed = float(value)
    if parsed < minimum or (strict and parsed == minimum):
        qualifier = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {qualifier} {minimum}")
    return parsed
