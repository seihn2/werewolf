from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .engine import GameRuntime
from .evaluation import run_head_to_head
from .llm import ChatModelConfig, OpenAICompatibleBackend
from .preference import build_dpo_dataset
from .self_play import run_self_play, summarize_jsonl
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
    return parser


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument(
        "--backend",
        choices=("heuristic", "openai-compatible"),
        default="heuristic",
    )


def _backend(name: str, *, environment_prefix: str = "WOLFPLAY"):
    if name == "heuristic":
        return None
    return OpenAICompatibleBackend(ChatModelConfig.from_env(environment_prefix))


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
    if args.command == "build-dpo":
        count = build_dpo_dataset(
            input_path=args.input,
            output_path=args.output,
            winning_only=args.winning_only,
            outcome_bonus=args.outcome_bonus,
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
