from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .engine import GameRuntime
from .llm import ChatBackend
from .models import Faction, GameResult, Winner


@dataclass(frozen=True, slots=True)
class HeadToHeadSummary:
    games_per_side: int
    challenger_as_werewolf_wins: int
    challenger_as_village_wins: int
    draws: int

    @property
    def total_games(self) -> int:
        return self.games_per_side * 2

    def to_dict(self) -> dict[str, Any]:
        challenger_wins = self.challenger_as_werewolf_wins + self.challenger_as_village_wins
        baseline_wins = self.total_games - challenger_wins - self.draws
        return {
            "games_per_side": self.games_per_side,
            "total_games": self.total_games,
            "challenger_as_werewolf_wins": self.challenger_as_werewolf_wins,
            "challenger_as_village_wins": self.challenger_as_village_wins,
            "challenger_wins": challenger_wins,
            "baseline_wins": baseline_wins,
            "draws": self.draws,
            "challenger_werewolf_win_rate": (
                self.challenger_as_werewolf_wins / self.games_per_side
            ),
            "challenger_village_win_rate": (self.challenger_as_village_wins / self.games_per_side),
            "challenger_overall_win_rate": (challenger_wins) / self.total_games,
        }


async def run_head_to_head(
    *,
    games_per_side: int,
    seed: int,
    max_rounds: int,
    challenger: ChatBackend | None,
    baseline: ChatBackend | None,
) -> tuple[list[GameResult], HeadToHeadSummary]:
    if games_per_side <= 0:
        raise ValueError("games_per_side must be positive")

    results: list[GameResult] = []
    challenger_as_werewolf_wins = 0
    challenger_as_village_wins = 0
    draws = 0

    for game_index in range(games_per_side):
        result = await GameRuntime(
            seed=seed + game_index,
            max_rounds=max_rounds,
            backend_by_faction={
                Faction.WEREWOLF: challenger,
                Faction.VILLAGE: baseline,
            },
        ).play()
        results.append(result)
        challenger_as_werewolf_wins += result.winner is Winner.WEREWOLF
        draws += result.winner is Winner.DRAW

    for game_index in range(games_per_side):
        result = await GameRuntime(
            seed=seed + games_per_side + game_index,
            max_rounds=max_rounds,
            backend_by_faction={
                Faction.WEREWOLF: baseline,
                Faction.VILLAGE: challenger,
            },
        ).play()
        results.append(result)
        challenger_as_village_wins += result.winner is Winner.VILLAGE
        draws += result.winner is Winner.DRAW

    return results, HeadToHeadSummary(
        games_per_side=games_per_side,
        challenger_as_werewolf_wins=challenger_as_werewolf_wins,
        challenger_as_village_wins=challenger_as_village_wins,
        draws=draws,
    )
