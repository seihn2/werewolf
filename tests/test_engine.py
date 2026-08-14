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


async def test_runtime_observer_receives_events_in_logical_order():
    observed = []

    async def observe(event):
        observed.append(event)

    result = await GameRuntime(seed=13, max_rounds=3, event_observer=observe).play()

    assert observed
    assert [event.logical_time for event in observed] == sorted(
        event.logical_time for event in observed
    )
    assert [event.to_dict() for event in observed if event.audience is None] == [
        event.to_dict() for event in result.events
    ]


def test_runtime_rejects_negative_public_event_delay():
    try:
        GameRuntime(public_event_delay_seconds=-0.1)
    except ValueError as error:
        assert "must not be negative" in str(error)
    else:
        raise AssertionError("expected negative delay to fail")
