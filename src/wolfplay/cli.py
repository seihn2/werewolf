from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .abstract_game import AbstractGameConfig
from .cfr_preference import build_cfr_dpo_dataset
from .engine import GameRuntime
from .evaluation import run_head_to_head
from .iterative import IterativePolicyConfig, IterativePolicyOptimizer
from .latent import (
    EmbeddingModelConfig,
    HashingTextEmbedder,
    OpenAICompatibleEmbedder,
    build_latent_strategy_file,
)
from .llm import ChatModelConfig, OpenAICompatibleBackend
from .models import Role
from .preference import build_dpo_dataset
from .self_play import run_self_play, summarize_jsonl
from .training.deep_cfr import DeepCFRConfig, train_deep_cfr
from .training.dpo import DPOTrainingConfig, train_dpo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wolfplay")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="run one complete game")
    _add_runtime_args(play)
    play.add_argument("--output", type=Path)

    self_play = subparsers.add_parser("self-play", help="generate self-play trajectories")
    _add_runtime_args(self_play)
    self_play.add_argument("--games", type=int, default=10)
    self_play.add_argument("--concurrency", type=int, default=1)
    self_play.add_argument("--output", type=Path, required=True)

    build_dpo = subparsers.add_parser("build-dpo", help="build prompt/chosen/rejected JSONL")
    build_dpo.add_argument("--input", type=Path, required=True)
    build_dpo.add_argument("--output", type=Path, required=True)
    build_dpo.add_argument("--winning-only", action="store_true")
    build_dpo.add_argument("--outcome-bonus", type=float, default=0.25)

    build_latent = subparsers.add_parser(
        "build-latent",
        help="embed discussion candidates and build role-specific K-Means strategy spaces",
    )
    build_latent.add_argument("--input", type=Path, required=True)
    build_latent.add_argument("--output", type=Path, required=True)
    _add_embedding_args(build_latent)
    _add_cluster_args(build_latent)
    build_latent.add_argument("--seed", type=int, default=42)

    train_cfr = subparsers.add_parser(
        "train-deep-cfr",
        help="train external-sampling Deep CFR in the abstract game",
    )
    train_cfr.add_argument("--latent-space", type=Path, required=True)
    train_cfr.add_argument("--output-dir", type=Path, required=True)
    _add_deep_cfr_args(train_cfr)
    train_cfr.add_argument("--seed", type=int, default=42)
    train_cfr.add_argument("--max-rounds", type=int, default=8)
    train_cfr.add_argument("--role-assignment-limit", type=int)

    build_cfr_dpo = subparsers.add_parser(
        "build-cfr-dpo",
        help="build DPO pairs ranked by a Deep CFR advantage checkpoint",
    )
    build_cfr_dpo.add_argument("--input", type=Path, required=True)
    build_cfr_dpo.add_argument("--checkpoint", type=Path, required=True)
    build_cfr_dpo.add_argument("--output", type=Path, required=True)
    build_cfr_dpo.add_argument("--winning-only", action="store_true")
    build_cfr_dpo.add_argument("--device", default="cpu")
    _add_embedding_args(build_cfr_dpo)

    evaluate = subparsers.add_parser("evaluate", help="summarize an existing self-play JSONL")
    evaluate.add_argument("--input", type=Path, required=True)

    head_to_head = subparsers.add_parser(
        "head-to-head",
        help="compare challenger and baseline models on both factions",
    )
    head_to_head.add_argument("--games-per-side", type=int, default=10)
    head_to_head.add_argument("--seed", type=int, default=42)
    head_to_head.add_argument("--max-rounds", type=int, default=8)
    head_to_head.add_argument("--output", type=Path)
    head_to_head.add_argument(
        "--challenger-backend",
        choices=("heuristic", "openai-compatible"),
        default="openai-compatible",
    )
    head_to_head.add_argument(
        "--baseline-backend",
        choices=("heuristic", "openai-compatible"),
        default="heuristic",
    )

    train = subparsers.add_parser("train-dpo", help="run TRL DPO training")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=float, default=2.0)
    train.add_argument("--learning-rate", type=float, default=1e-6)
    train.add_argument("--beta", type=float, default=0.1)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=16)
    train.add_argument("--max-length", type=int, default=2048)
    train.add_argument("--no-lora", action="store_true")
    train.add_argument("--lora-r", type=int, default=32)
    train.add_argument("--lora-alpha", type=int, default=16)

    iterate = subparsers.add_parser(
        "iterate-policy",
        help="run repeated self-play, clustering, Deep CFR and optional DPO training",
    )
    iterate.add_argument("--output-dir", type=Path, required=True)
    iterate.add_argument("--iterations", type=int, default=1)
    iterate.add_argument("--games-per-iteration", type=int, default=100)
    iterate.add_argument("--concurrency", type=int, default=1)
    iterate.add_argument("--seed", type=int, default=2025)
    iterate.add_argument("--max-rounds", type=int, default=8)
    iterate.add_argument(
        "--backend",
        choices=("heuristic", "openai-compatible"),
        default="heuristic",
    )
    iterate.add_argument("--clusters-added-per-iteration", type=int, default=1)
    iterate.add_argument("--no-resume", action="store_true")
    _add_embedding_args(iterate)
    _add_cluster_args(iterate)
    _add_deep_cfr_args(iterate, prefix="cfr-")
    iterate.add_argument("--dpo-model")
    iterate.add_argument("--continue-dpo-from-previous", action="store_true")
    iterate.add_argument("--dpo-epochs", type=float, default=2.0)
    iterate.add_argument("--dpo-learning-rate", type=float, default=1e-6)
    iterate.add_argument("--dpo-beta", type=float, default=0.1)
    iterate.add_argument("--dpo-batch-size", type=int, default=1)
    iterate.add_argument("--dpo-gradient-accumulation-steps", type=int, default=16)
    iterate.add_argument("--dpo-no-lora", action="store_true")
    return parser


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument(
        "--backend",
        choices=("heuristic", "openai-compatible"),
        default="heuristic",
    )


def _add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-backend",
        choices=("hashing", "openai-compatible"),
        default="hashing",
    )
    parser.add_argument("--hash-dimensions", type=int, default=256)


def _add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--werewolf-clusters", type=int, default=3)
    parser.add_argument("--seer-clusters", type=int, default=2)
    parser.add_argument("--doctor-clusters", type=int, default=2)
    parser.add_argument("--villager-clusters", type=int, default=2)


def _add_deep_cfr_args(parser: argparse.ArgumentParser, *, prefix: str = "") -> None:
    option = f"--{prefix}"
    parser.add_argument(f"{option}iterations", type=int, default=10)
    parser.add_argument(f"{option}traversals-per-player", type=int, default=4)
    parser.add_argument(f"{option}advantage-train-steps", type=int, default=100)
    parser.add_argument(f"{option}strategy-train-steps", type=int, default=200)
    parser.add_argument(f"{option}batch-size", type=int, default=128)
    parser.add_argument(f"{option}learning-rate", type=float, default=1e-3)
    parser.add_argument(f"{option}advantage-buffer-capacity", type=int, default=100_000)
    parser.add_argument(f"{option}strategy-buffer-capacity", type=int, default=100_000)
    parser.add_argument(
        f"{option}hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 256, 256],
    )
    parser.add_argument(f"{option}max-traversal-depth", type=int, default=64)
    parser.add_argument(f"{option}max-rollout-steps", type=int, default=512)
    parser.add_argument(f"{option}device", default="auto")
    parser.add_argument(f"{option}checkpoint-every", type=int, default=1)
    parser.add_argument(f"{option}no-save-buffers", action="store_true")


def _backend(name: str, *, environment_prefix: str = "WOLFPLAY"):
    if name == "heuristic":
        return None
    return OpenAICompatibleBackend(ChatModelConfig.from_env(environment_prefix))


def _embedder(args: argparse.Namespace):
    if args.embedding_backend == "hashing":
        return HashingTextEmbedder(args.hash_dimensions)
    return OpenAICompatibleEmbedder(EmbeddingModelConfig.from_env())


def _cluster_counts(args: argparse.Namespace) -> dict[Role, int]:
    return {
        Role.WEREWOLF: args.werewolf_clusters,
        Role.SEER: args.seer_clusters,
        Role.DOCTOR: args.doctor_clusters,
        Role.VILLAGER: args.villager_clusters,
    }


def _deep_cfr_config(args: argparse.Namespace, *, prefix: str = "") -> DeepCFRConfig:
    name = prefix.replace("-", "_")

    def value(suffix: str):
        return getattr(args, f"{name}{suffix}")

    return DeepCFRConfig(
        iterations=value("iterations"),
        traversals_per_player=value("traversals_per_player"),
        advantage_train_steps=value("advantage_train_steps"),
        strategy_train_steps=value("strategy_train_steps"),
        batch_size=value("batch_size"),
        learning_rate=value("learning_rate"),
        advantage_buffer_capacity=value("advantage_buffer_capacity"),
        strategy_buffer_capacity=value("strategy_buffer_capacity"),
        hidden_sizes=tuple(value("hidden_sizes")),
        max_traversal_depth=value("max_traversal_depth"),
        max_rollout_steps=value("max_rollout_steps"),
        seed=args.seed,
        device=value("device"),
        checkpoint_every=value("checkpoint_every"),
        save_buffers=not value("no_save_buffers"),
    )


async def _run_async(args: argparse.Namespace) -> None:
    backend = _backend(getattr(args, "backend", "heuristic"))
    try:
        if args.command == "play":
            runtime = GameRuntime(seed=args.seed, max_rounds=args.max_rounds, backend=backend)
            result = await runtime.play()
            _print_transcript(result.to_dict())
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        elif args.command == "self-play":
            _, summary = await run_self_play(
                games=args.games,
                seed=args.seed,
                max_rounds=args.max_rounds,
                output_path=args.output,
                backend=backend,
                concurrency=args.concurrency,
            )
            print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    finally:
        if backend is not None:
            await backend.aclose()


async def _run_head_to_head(args: argparse.Namespace) -> None:
    challenger = _backend(
        args.challenger_backend,
        environment_prefix="WOLFPLAY_CHALLENGER",
    )
    baseline = _backend(
        args.baseline_backend,
        environment_prefix="WOLFPLAY_BASELINE",
    )
    try:
        results, summary = await run_head_to_head(
            games_per_side=args.games_per_side,
            seed=args.seed,
            max_rounds=args.max_rounds,
            challenger=challenger,
            baseline=baseline,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = args.output.with_name(f".{args.output.name}.tmp")
            temporary_path.write_text(
                json.dumps(
                    {
                        "summary": summary.to_dict(),
                        "games": [result.to_dict() for result in results],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary_path.replace(args.output)
    finally:
        for backend in {challenger, baseline}:
            if backend is not None:
                await backend.aclose()


async def _run_iterative(args: argparse.Namespace) -> None:
    def backend_factory(iteration: int, previous_checkpoint: Path | None):
        del previous_checkpoint
        if args.backend == "heuristic":
            return None
        prefix = "WOLFPLAY" if iteration == 1 else f"WOLFPLAY_ITERATION_{iteration}"
        return _backend(args.backend, environment_prefix=prefix)

    dpo_factory = None
    if args.dpo_model:

        def dpo_factory(
            iteration: int,
            dataset: Path,
            output_dir: Path,
            previous_checkpoint: Path | None,
        ) -> DPOTrainingConfig:
            del iteration
            model = (
                str(previous_checkpoint)
                if args.continue_dpo_from_previous and previous_checkpoint is not None
                else args.dpo_model
            )
            return DPOTrainingConfig(
                dataset=dataset,
                model=model,
                output_dir=output_dir,
                epochs=args.dpo_epochs,
                learning_rate=args.dpo_learning_rate,
                beta=args.dpo_beta,
                batch_size=args.dpo_batch_size,
                gradient_accumulation_steps=args.dpo_gradient_accumulation_steps,
                seed=args.seed,
                use_lora=not args.dpo_no_lora,
            )

    optimizer = IterativePolicyOptimizer(
        config=IterativePolicyConfig(
            iterations=args.iterations,
            games_per_iteration=args.games_per_iteration,
            concurrency=args.concurrency,
            seed=args.seed,
            max_rounds=args.max_rounds,
            base_clusters=_cluster_counts(args),
            clusters_added_per_iteration=args.clusters_added_per_iteration,
            deep_cfr=_deep_cfr_config(args, prefix="cfr_"),
            resume=not args.no_resume,
        ),
        embedder=_embedder(args),
        backend_factory=backend_factory,
        dpo_config_factory=dpo_factory,
    )
    artifacts = await optimizer.run(args.output_dir)
    print(
        json.dumps(
            {"iterations": [artifact.to_dict() for artifact in artifacts]},
            ensure_ascii=False,
            indent=2,
        )
    )


def _print_transcript(record: dict) -> None:
    for event in record["events"]:
        if event["topic"] == "speech":
            print(f"[{event['logical_time']}] {event['sender']}: {event['payload']['message']}")
        elif event["topic"] in {"night_result", "vote_result", "game_over"}:
            print(f"[{event['logical_time']}] {event['topic']}: {event['payload']}")
    print(
        json.dumps(
            {
                "game_id": record["game_id"],
                "winner": record["winner"],
                "rounds": record["rounds"],
                "termination_reason": record["termination_reason"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"play", "self-play"}:
        asyncio.run(_run_async(args))
        return
    if args.command == "head-to-head":
        asyncio.run(_run_head_to_head(args))
        return
    if args.command == "iterate-policy":
        asyncio.run(_run_iterative(args))
        return
    if args.command == "build-dpo":
        count = build_dpo_dataset(
            input_path=args.input,
            output_path=args.output,
            winning_only=args.winning_only,
            outcome_bonus=args.outcome_bonus,
        )
        print(json.dumps({"pairs": count, "output": str(args.output)}, indent=2))
        return
    if args.command == "build-latent":
        latent_space = build_latent_strategy_file(
            input_path=args.input,
            output_path=args.output,
            embedder=_embedder(args),
            clusters_by_role=_cluster_counts(args),
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "embedding_model": latent_space.embedding_model,
                    "roles": {
                        role.value: len(role_space.clusters)
                        for role, role_space in latent_space.roles.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "train-deep-cfr":
        summary = train_deep_cfr(
            latent_space=args.latent_space,
            output_dir=args.output_dir,
            config=_deep_cfr_config(args),
            game_config=AbstractGameConfig(
                max_rounds=args.max_rounds,
                role_assignment_limit=args.role_assignment_limit,
                role_assignment_seed=args.seed,
            ),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "build-cfr-dpo":
        count = build_cfr_dpo_dataset(
            input_path=args.input,
            checkpoint=args.checkpoint,
            output_path=args.output,
            embedder=_embedder(args),
            winning_only=args.winning_only,
            device=args.device,
        )
        print(json.dumps({"pairs": count, "output": str(args.output)}, indent=2))
        return
    if args.command == "evaluate":
        print(json.dumps(summarize_jsonl(args.input).to_dict(), ensure_ascii=False, indent=2))
        return
    if args.command == "train-dpo":
        train_dpo(
            DPOTrainingConfig(
                dataset=args.dataset,
                model=args.model,
                output_dir=args.output_dir,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                beta=args.beta,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_length=args.max_length,
                use_lora=not args.no_lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
        )


if __name__ == "__main__":
    main()
