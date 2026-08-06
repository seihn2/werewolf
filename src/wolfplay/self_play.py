from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine import GameRuntime
from .llm import ChatBackend
from .models import GameResult, Winner


@dataclass(frozen=True, slots=True)
class SelfPlaySummary:
    games: int
    werewolf_wins: int
    village_wins: int
    draws: int
    average_rounds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "werewolf_wins": self.werewolf_wins,
            "village_wins": self.village_wins,
            "draws": self.draws,
            "average_rounds": self.average_rounds,
            "werewolf_win_rate": self.werewolf_wins / self.games if self.games else 0.0,
            "village_win_rate": self.village_wins / self.games if self.games else 0.0,
        }


async def run_self_play(
    *,
    games: int,
    seed: int,
    max_rounds: int,
    output_path: Path | None = None,
    backend: ChatBackend | None = None,
    concurrency: int = 1,
) -> tuple[list[GameResult], SelfPlaySummary]:
    if games <= 0:
        raise ValueError("games must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def run_game(game_index: int) -> GameResult:
        async with semaphore:
            runtime = GameRuntime(
                seed=seed + game_index,
                max_rounds=max_rounds,
                backend=backend,
            )
            return await runtime.play()

    results = list(await asyncio.gather(*(run_game(index) for index in range(games))))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for result in results:
                output_file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)

    summary = summarize_results(results)
    return results, summary


def summarize_results(results: list[GameResult]) -> SelfPlaySummary:
    werewolf_wins = sum(result.winner is Winner.WEREWOLF for result in results)
    village_wins = sum(result.winner is Winner.VILLAGE for result in results)
    draws = sum(result.winner is Winner.DRAW for result in results)
    average_rounds = sum(result.rounds for result in results) / len(results) if results else 0.0
    return SelfPlaySummary(
        games=len(results),
        werewolf_wins=werewolf_wins,
        village_wins=village_wins,
        draws=draws,
        average_rounds=average_rounds,
    )


def summarize_jsonl(path: Path) -> SelfPlaySummary:
    records = load_jsonl(path)
    if not records:
        return SelfPlaySummary(0, 0, 0, 0, 0.0)
    winners = [Winner(record["winner"]) for record in records]
    rounds = [int(record["rounds"]) for record in records]
    return SelfPlaySummary(
        games=len(records),
        werewolf_wins=sum(winner is Winner.WEREWOLF for winner in winners),
        village_wins=sum(winner is Winner.VILLAGE for winner in winners),
        draws=sum(winner is Winner.DRAW for winner in winners),
        average_rounds=sum(rounds) / len(rounds),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {path}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must be an object: {path}")
            records.append(record)
    return records
