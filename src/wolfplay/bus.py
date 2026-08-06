from __future__ import annotations

import asyncio
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from .models import GameEvent, Phase

_PRIVATE_TOPICS = frozenset(
    {
        "doctor_choice",
        "role_assignment",
        "seer_result",
        "werewolf_proposal",
        "werewolf_team",
    }
)


class LamportClock:
    def __init__(self) -> None:
        self._value = 0
        self._lock = asyncio.Lock()

    @property
    def value(self) -> int:
        return self._value

    async def tick(self, received_time: int | None = None) -> int:
        if received_time is not None:
            if isinstance(received_time, bool) or not isinstance(received_time, int):
                raise TypeError("received_time must be an integer")
            if received_time < 0:
                raise ValueError("received_time must be non-negative")
        async with self._lock:
            if received_time is not None:
                self._value = max(self._value, received_time)
            self._value += 1
            return self._value


class AsyncMessageBus:
    """Centralized event bus with audience-based view isolation."""

    def __init__(self) -> None:
        self.clock = LamportClock()
        self._queues: dict[str, asyncio.Queue[GameEvent]] = {}
        self._events: list[GameEvent] = []
        self._known_players: set[str] = set()
        self._publish_lock = asyncio.Lock()

    def register(self, player_ids: Iterable[str]) -> None:
        if isinstance(player_ids, str):
            raise TypeError("player_ids must be an iterable of player IDs, not a string")
        normalized_ids = tuple(player_ids)
        for player_id in normalized_ids:
            self._validate_player_id(player_id)
        for player_id in normalized_ids:
            self._known_players.add(player_id)
            self._queues.setdefault(player_id, asyncio.Queue())

    async def publish(
        self,
        *,
        topic: str,
        round_no: int,
        phase: Phase,
        payload: dict[str, Any],
        sender: str | None = None,
        audience: Iterable[str] | None = None,
        received_time: int | None = None,
    ) -> GameEvent:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")
        normalized_audience = self._normalize_audience(topic, audience)
        stored_payload = deepcopy(payload)

        async with self._publish_lock:
            event = GameEvent(
                logical_time=await self.clock.tick(received_time),
                topic=topic,
                round_no=round_no,
                phase=phase,
                payload=stored_payload,
                sender=sender,
                audience=normalized_audience,
            )
            self._events.append(event)

            recipients = (
                tuple(sorted(self._known_players))
                if normalized_audience is None
                else normalized_audience
            )
            for player_id in recipients:
                self._queues[player_id].put_nowait(self._copy_event(event))
        return self._copy_event(event)

    def events_for(self, player_id: str) -> tuple[GameEvent, ...]:
        if player_id not in self._known_players:
            raise KeyError(f"unknown player: {player_id}")
        return tuple(
            self._copy_event(event) for event in self._events if event.visible_to(player_id)
        )

    async def drain(self, player_id: str) -> list[GameEvent]:
        if player_id not in self._known_players:
            raise KeyError(f"unknown player: {player_id}")
        queue = self._queues[player_id]
        events: list[GameEvent] = []
        while True:
            try:
                events.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    @property
    def events(self) -> tuple[GameEvent, ...]:
        return tuple(self._copy_event(event) for event in self._events if event.audience is None)

    def _normalize_audience(
        self, topic: str, audience: Iterable[str] | None
    ) -> tuple[str, ...] | None:
        if audience is None:
            if topic in _PRIVATE_TOPICS:
                raise ValueError(f"private topic requires an explicit audience: {topic}")
            return None
        if isinstance(audience, str):
            raise TypeError("audience must be an iterable of player IDs, not a string")

        audience_ids = tuple(audience)
        if not audience_ids:
            raise ValueError("private audience must contain at least one player")
        for player_id in audience_ids:
            self._validate_player_id(player_id)

        normalized = tuple(sorted(set(audience_ids)))
        unknown_players = set(normalized) - self._known_players
        if unknown_players:
            unknown = ", ".join(sorted(unknown_players))
            raise KeyError(f"unknown audience player(s): {unknown}")
        return normalized

    @staticmethod
    def _validate_player_id(player_id: str) -> None:
        if not isinstance(player_id, str):
            raise TypeError("player ID must be a string")
        if not player_id.strip():
            raise ValueError("player ID must not be empty")

    @staticmethod
    def _copy_event(event: GameEvent) -> GameEvent:
        return GameEvent(
            logical_time=event.logical_time,
            topic=event.topic,
            round_no=event.round_no,
            phase=event.phase,
            payload=deepcopy(event.payload),
            sender=event.sender,
            audience=event.audience,
        )
