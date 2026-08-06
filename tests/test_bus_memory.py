import asyncio

import pytest

from wolfplay.bus import AsyncMessageBus, LamportClock
from wolfplay.memory import HierarchicalMemory, MemoryStore
from wolfplay.models import GameEvent, Phase, Role


async def test_lamport_clock_merges_remote_time_and_rejects_invalid_values():
    clock = LamportClock()

    assert await clock.tick() == 1
    assert await clock.tick(received_time=7) == 8
    assert await clock.tick(received_time=3) == 9

    with pytest.raises(ValueError, match="non-negative"):
        await clock.tick(received_time=-1)
    with pytest.raises(TypeError, match="integer"):
        await clock.tick(received_time=True)
    assert clock.value == 9


async def test_private_events_are_isolated_and_public_history_cannot_leak_them():
    bus = AsyncMessageBus()
    bus.register(["player_0", "player_1"])
    memories = MemoryStore(["player_0", "player_1"])

    private = await bus.publish(
        topic="seer_result",
        round_no=1,
        phase=Phase.NIGHT_SEER,
        payload={"target_id": "player_1", "is_werewolf": True},
        audience=["player_0"],
    )
    memories.observe(private)
    public = await bus.publish(
        topic="night_result",
        round_no=1,
        phase=Phase.NIGHT_RESOLUTION,
        payload={"victim_id": None},
    )
    memories.observe(public)

    assert private.logical_time == 1
    assert public.logical_time == 2
    assert [event.topic for event in bus.events] == ["night_result"]
    assert private in bus.events_for("player_0")
    assert private not in bus.events_for("player_1")
    assert [event.topic for event in await bus.drain("player_0")] == [
        "seer_result",
        "night_result",
    ]
    assert [event.topic for event in await bus.drain("player_1")] == ["night_result"]
    assert memories["player_0"].role_belief("player_1", Role.WEREWOLF) == 1.0
    assert not memories["player_1"].semantic_beliefs


async def test_publish_rejects_unsafe_private_audiences_without_advancing_clock():
    bus = AsyncMessageBus()
    bus.register(["player_0"])

    with pytest.raises(ValueError, match="explicit audience"):
        await bus.publish(
            topic="role_assignment",
            round_no=1,
            phase=Phase.SETUP,
            payload={"player_id": "player_0", "role": Role.SEER.value},
        )
    with pytest.raises(ValueError, match="at least one"):
        await bus.publish(
            topic="seer_result",
            round_no=1,
            phase=Phase.NIGHT_SEER,
            payload={"target_id": "player_0", "is_werewolf": False},
            audience=[],
        )
    with pytest.raises(KeyError, match="player_404"):
        await bus.publish(
            topic="seer_result",
            round_no=1,
            phase=Phase.NIGHT_SEER,
            payload={"target_id": "player_0", "is_werewolf": False},
            audience=["player_404"],
        )

    assert bus.clock.value == 0
    assert bus.events_for("player_0") == ()
    assert await bus.drain("player_0") == []


async def test_payloads_are_isolated_from_publishers_history_and_other_queues():
    bus = AsyncMessageBus()
    bus.register(["player_0", "player_1"])
    payload = {"members": ["player_0", "player_1"], "plan": {"target": "player_2"}}

    published = await bus.publish(
        topic="werewolf_team",
        round_no=1,
        phase=Phase.SETUP,
        payload=payload,
        audience=["player_0", "player_1"],
    )
    payload["members"].clear()
    published.payload["plan"]["target"] = "tampered"

    player_0_event = (await bus.drain("player_0"))[0]
    player_1_event = (await bus.drain("player_1"))[0]
    player_0_event.payload["members"].append("intruder")
    player_0_event.payload["plan"]["target"] = "player_6"

    assert player_1_event.payload == {
        "members": ["player_0", "player_1"],
        "plan": {"target": "player_2"},
    }
    assert bus.events_for("player_1")[0].payload == player_1_event.payload
    assert bus.events == ()


async def test_concurrent_publications_preserve_history_and_queue_order():
    bus = AsyncMessageBus()
    bus.register(["player_0", "player_1"])

    await asyncio.gather(
        *(
            bus.publish(
                topic="speech",
                round_no=1,
                phase=Phase.DAY_DISCUSSION,
                payload={"index": index},
                sender=f"player_{index % 2}",
            )
            for index in range(40)
        )
    )

    history_times = [event.logical_time for event in bus.events]
    assert history_times == list(range(1, 41))
    assert [event.logical_time for event in await bus.drain("player_0")] == history_times
    assert [event.logical_time for event in await bus.drain("player_1")] == history_times
    assert await bus.drain("player_0") == []


def test_memory_tiers_are_bounded_deduplicated_and_recalled_by_metadata():
    memory = HierarchicalMemory(
        "player_0",
        working_limit=2,
        episodic_limit=2,
        reflection_limit=1,
        known_player_ids=["player_0", "player_1"],
    )
    events = [
        GameEvent(
            logical_time=index,
            topic="speech",
            round_no=1,
            phase=Phase.DAY_DISCUSSION,
            payload={"message": f"claim {index}"},
            sender="player_1",
        )
        for index in range(1, 4)
    ]
    for event in events:
        memory.observe(event)
    memory.observe(events[-1])
    memory.add_reflection(round_no=1, text="first reflection")
    memory.add_reflection(round_no=1, text="latest reflection")

    recalled = memory.recall(query="day discussion", limit=3)

    assert [entry.logical_time for entry in memory.working] == [2, 3]
    assert [entry.logical_time for entry in memory.episodic] == [2, 3]
    assert len(memory.reflections) == 1
    assert memory.reflections[0].logical_time == 3
    assert recalled.count("speech:") == 2
    assert "claim 1" not in recalled
    assert "latest reflection" in recalled
    assert memory.recall(limit=0) == ""
    with pytest.raises(ValueError, match="non-negative"):
        memory.recall(limit=-1)


def test_semantic_beliefs_ignore_stale_or_malformed_evidence():
    memory = HierarchicalMemory(
        "player_0",
        known_player_ids=["player_0", "player_1"],
    )
    newer = GameEvent(
        logical_time=5,
        topic="seer_result",
        round_no=2,
        phase=Phase.NIGHT_SEER,
        payload={"target_id": "player_1", "is_werewolf": True},
        audience=("player_0",),
    )
    stale = GameEvent(
        logical_time=4,
        topic="seer_result",
        round_no=1,
        phase=Phase.NIGHT_SEER,
        payload={"target_id": "player_1", "is_werewolf": False},
        audience=("player_0",),
    )
    malformed = GameEvent(
        logical_time=6,
        topic="seer_result",
        round_no=2,
        phase=Phase.NIGHT_SEER,
        payload={"target_id": "player_1", "is_werewolf": "false"},
        audience=("player_0",),
    )

    memory.observe(newer)
    memory.observe(stale)
    memory.observe(malformed)

    assert memory.role_belief("player_1", Role.WEREWOLF) == 1.0
    assert memory.role_belief("player_1", Role.SEER) == 0.0


def test_exact_role_evidence_resets_conflicts_and_belief_views_are_defensive():
    memory = HierarchicalMemory(
        "player_0",
        known_player_ids=["player_0", "player_1"],
    )
    memory.set_role_belief("player_1", Role.WEREWOLF, 0.8, logical_time=1)
    memory.observe(
        GameEvent(
            logical_time=2,
            topic="role_assignment",
            round_no=1,
            phase=Phase.SETUP,
            payload={"player_id": "player_1", "role": Role.SEER.value},
            audience=("player_0",),
        )
    )

    snapshot = memory.semantic_beliefs
    snapshot["player_1"][Role.SEER] = 0.0

    assert memory.role_belief("player_1", Role.SEER) == 1.0
    assert memory.role_belief("player_1", Role.WEREWOLF) == 0.0
    with pytest.raises(ValueError, match="finite"):
        memory.set_role_belief("player_1", Role.WEREWOLF, float("nan"))
    with pytest.raises(KeyError, match="player_404"):
        memory.set_role_belief("player_404", Role.WEREWOLF, 0.5)
