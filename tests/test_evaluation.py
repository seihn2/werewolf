from wolfplay.evaluation import run_head_to_head


async def test_head_to_head_runs_both_factions():
    results, summary = await run_head_to_head(
        games_per_side=2,
        seed=500,
        max_rounds=3,
        challenger=None,
        baseline=None,
    )

    assert len(results) == 4
    assert summary.total_games == 4
    assert (
        summary.challenger_as_werewolf_wins + summary.challenger_as_village_wins + summary.draws
        <= 4
    )
    assert 0.0 <= summary.to_dict()["challenger_overall_win_rate"] <= 1.0
    assert (
        summary.to_dict()["challenger_wins"]
        + summary.to_dict()["baseline_wins"]
        + summary.to_dict()["draws"]
        == 4
    )
