from collections import Counter

from wolfplay.engine import GameRuntime
from wolfplay.models import Role, Winner


async def test_complete_game_runs_offline():
    runtime = GameRuntime(seed=7, max_rounds=4)
    result = await runtime.play()

    role_counts = Counter(player.role for player in result.players.values())
    assert role_counts == Counter(
        {
            Role.WEREWOLF: 2,
            Role.SEER: 1,
            Role.DOCTOR: 1,
            Role.VILLAGER: 3,
        }
    )
    assert result.winner in {Winner.WEREWOLF, Winner.VILLAGE, Winner.DRAW}
    assert result.decision_traces
    assert any(event.topic == "speech" for event in result.events)
    assert result.events[-1].topic == "game_over"


async def test_role_assignment_is_not_visible_to_other_players():
    runtime = GameRuntime(seed=11, max_rounds=2)
    await runtime.play()

    player_0_events = runtime.bus.events_for("player_0")
    foreign_roles = [
        event
        for event in player_0_events
        if event.topic == "role_assignment" and event.payload["player_id"] != "player_0"
    ]
    assert foreign_roles == []


async def test_same_seed_is_reproducible():
    first = await GameRuntime(seed=23, max_rounds=4).play()
    second = await GameRuntime(seed=23, max_rounds=4).play()

    assert first.to_dict() == second.to_dict()


async def test_runtime_cannot_be_reused():
    runtime = GameRuntime(seed=31, max_rounds=2)
    await runtime.play()

    try:
        await runtime.play()
    except RuntimeError as error:
        assert "only run one game" in str(error)
    else:
        raise AssertionError("expected GameRuntime reuse to fail")
