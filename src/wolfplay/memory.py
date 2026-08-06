from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .models import GameEvent, Role

_EPISODIC_TOPICS = frozenset({"night_result", "seer_result", "speech", "vote_result"})
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


class MemoryTier(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    REFLECTION = "reflection"


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    tier: MemoryTier
    logical_time: int
    round_no: int
    text: str
    metadata: dict[str, Any]


class HierarchicalMemory:
    def __init__(
        self,
        player_id: str,
        working_limit: int = 24,
        episodic_limit: int = 256,
        reflection_limit: int = 64,
        known_player_ids: Iterable[str] | None = None,
    ) -> None:
        self._validate_player_id(player_id)
        self._validate_limit("working_limit", working_limit)
        self._validate_limit("episodic_limit", episodic_limit)
        self._validate_limit("reflection_limit", reflection_limit)

        known_players = None
        if known_player_ids is not None:
            if isinstance(known_player_ids, str):
                raise TypeError("known_player_ids must be an iterable of player IDs")
            known_players = frozenset(known_player_ids)
            for known_player_id in known_players:
                self._validate_player_id(known_player_id)
            if player_id not in known_players:
                raise ValueError("player_id must be included in known_player_ids")

        self.player_id = player_id
        self.working: deque[MemoryEntry] = deque(maxlen=working_limit)
        self.episodic: list[MemoryEntry] = []
        self.reflections: list[MemoryEntry] = []
        self._episodic_limit = episodic_limit
        self._reflection_limit = reflection_limit
        self._known_player_ids = known_players
        self._semantic_beliefs: dict[str, dict[Role, float]] = {}
        self._belief_times: dict[str, dict[Role, int]] = {}
        self._seen_events: set[tuple[Any, ...]] = set()
        self._last_logical_time = 0

    def observe(self, event: GameEvent) -> None:
        if not event.visible_to(self.player_id):
            return
        event_key = (
            event.logical_time,
            event.topic,
            event.round_no,
            event.phase,
            event.sender,
            event.audience,
            repr(event.payload),
        )
        if event_key in self._seen_events:
            return
        self._seen_events.add(event_key)
        self._last_logical_time = max(self._last_logical_time, event.logical_time)

        entry = MemoryEntry(
            tier=MemoryTier.WORKING,
            logical_time=event.logical_time,
            round_no=event.round_no,
            text=f"{event.topic}: {event.payload}",
            metadata={"phase": event.phase.value, "sender": event.sender},
        )
        self.working.append(entry)

        if event.topic in _EPISODIC_TOPICS:
            self._append_bounded(
                self.episodic,
                MemoryEntry(
                    tier=MemoryTier.EPISODIC,
                    logical_time=event.logical_time,
                    round_no=event.round_no,
                    text=entry.text,
                    metadata=dict(entry.metadata),
                ),
                self._episodic_limit,
            )

        if event.topic == "seer_result":
            target_id = event.payload.get("target_id")
            is_werewolf = event.payload.get("is_werewolf")
            if self._can_track_player(target_id) and type(is_werewolf) is bool:
                self.set_role_belief(
                    target_id,
                    Role.WEREWOLF,
                    1.0 if is_werewolf else 0.0,
                    logical_time=event.logical_time,
                    exclusive=is_werewolf,
                )

        if event.topic == "role_assignment":
            target_id = event.payload.get("player_id")
            try:
                role = Role(event.payload.get("role"))
            except (TypeError, ValueError):
                return
            if self._can_track_player(target_id):
                self.set_role_belief(
                    target_id,
                    role,
                    1.0,
                    logical_time=event.logical_time,
                    exclusive=True,
                )

    @property
    def semantic_beliefs(self) -> dict[str, dict[Role, float]]:
        return {
            player_id: dict(role_scores)
            for player_id, role_scores in self._semantic_beliefs.items()
        }

    def set_role_belief(
        self,
        player_id: str,
        role: Role,
        probability: float,
        *,
        logical_time: int | None = None,
        exclusive: bool = False,
    ) -> None:
        self._validate_player_id(player_id)
        if self._known_player_ids is not None and player_id not in self._known_player_ids:
            raise KeyError(f"unknown player: {player_id}")
        role = Role(role)
        probability = self._normalize_probability(probability)
        evidence_time = (
            self._last_logical_time
            if logical_time is None
            else self._normalize_logical_time(logical_time)
        )

        belief_times = self._belief_times.setdefault(player_id, {})
        if exclusive:
            if evidence_time < max(belief_times.values(), default=-1):
                return
            self._semantic_beliefs[player_id] = {
                candidate_role: 1.0 if candidate_role is role else 0.0 for candidate_role in Role
            }
            self._belief_times[player_id] = {
                candidate_role: evidence_time for candidate_role in Role
            }
            return

        if evidence_time < belief_times.get(role, -1):
            return
        beliefs = self._semantic_beliefs.setdefault(player_id, {})
        beliefs[role] = probability
        belief_times[role] = evidence_time

    def role_belief(self, player_id: str, role: Role) -> float:
        return self._semantic_beliefs.get(player_id, {}).get(Role(role), 0.0)

    def add_reflection(self, round_no: int, text: str, logical_time: int | None = None) -> None:
        if not isinstance(text, str):
            raise TypeError("reflection text must be a string")
        text = text.strip()
        if not text:
            return
        reflection_time = (
            self._last_logical_time
            if logical_time is None
            else self._normalize_logical_time(logical_time)
        )
        self._append_bounded(
            self.reflections,
            MemoryEntry(
                tier=MemoryTier.REFLECTION,
                logical_time=reflection_time,
                round_no=round_no,
                text=text,
                metadata={},
            ),
            self._reflection_limit,
        )

    def recall(self, query: str = "", limit: int = 16) -> str:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit < 0:
            raise ValueError("limit must be non-negative")
        query_tokens = self._tokens(query)

        deduplicated: dict[tuple[int, int, str], MemoryEntry] = {}
        for entry in (*self.working, *self.episodic, *self.reflections):
            key = (entry.logical_time, entry.round_no, entry.text)
            deduplicated.setdefault(key, entry)
        candidates = list(deduplicated.values())

        def rank(entry: MemoryEntry) -> tuple[int, int, int]:
            searchable_tokens = self._tokens(entry.text)
            for key, value in entry.metadata.items():
                searchable_tokens.update(self._tokens(key))
                searchable_tokens.update(self._tokens(str(value)))
            overlap = len(query_tokens & searchable_tokens)
            tier_priority = 1 if entry.tier is MemoryTier.REFLECTION else 0
            return overlap, entry.logical_time, tier_priority

        selected = sorted(candidates, key=rank, reverse=True)[:limit]
        selected.sort(key=lambda entry: (entry.logical_time, entry.round_no, entry.text))
        lines = [f"[{entry.tier.value}] {entry.text}" for entry in selected]
        if self._semantic_beliefs:
            belief_lines = []
            for player_id, role_scores in sorted(self._semantic_beliefs.items()):
                scores = ", ".join(
                    f"{role.value}={role_scores[role]:.2f}" for role in Role if role in role_scores
                )
                belief_lines.append(f"{player_id}: {scores}")
            lines.append("[semantic] " + " | ".join(belief_lines))
        return "\n".join(lines)

    def _can_track_player(self, player_id: object) -> bool:
        return (
            isinstance(player_id, str)
            and bool(player_id.strip())
            and (self._known_player_ids is None or player_id in self._known_player_ids)
        )

    @staticmethod
    def _append_bounded(entries: list[MemoryEntry], entry: MemoryEntry, limit: int) -> None:
        entries.append(entry)
        overflow = len(entries) - limit
        if overflow > 0:
            del entries[:overflow]

    @staticmethod
    def _normalize_probability(probability: float) -> float:
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise TypeError("probability must be a real number")
        normalized = float(probability)
        if not math.isfinite(normalized):
            raise ValueError("probability must be finite")
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def _normalize_logical_time(logical_time: int) -> int:
        if isinstance(logical_time, bool) or not isinstance(logical_time, int):
            raise TypeError("logical_time must be an integer")
        if logical_time < 0:
            raise ValueError("logical_time must be non-negative")
        return logical_time

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {token.casefold() for token in _TOKEN_PATTERN.findall(value)}

    @staticmethod
    def _validate_limit(name: str, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError(f"{name} must be an integer")
        if limit <= 0:
            raise ValueError(f"{name} must be positive")

    @staticmethod
    def _validate_player_id(player_id: str) -> None:
        if not isinstance(player_id, str):
            raise TypeError("player ID must be a string")
        if not player_id.strip():
            raise ValueError("player ID must not be empty")


class MemoryStore:
    def __init__(self, player_ids: list[str]) -> None:
        if isinstance(player_ids, str):
            raise TypeError("player_ids must be a list of player IDs")
        if len(player_ids) != len(set(player_ids)):
            raise ValueError("player_ids must be unique")
        for player_id in player_ids:
            HierarchicalMemory._validate_player_id(player_id)
        known_player_ids = frozenset(player_ids)
        self._memories = {
            player_id: HierarchicalMemory(
                player_id=player_id,
                known_player_ids=known_player_ids,
            )
            for player_id in player_ids
        }

    def __getitem__(self, player_id: str) -> HierarchicalMemory:
        return self._memories[player_id]

    def observe(self, event: GameEvent) -> None:
        for memory in self._memories.values():
            memory.observe(event)

    def add_reflection(
        self,
        player_id: str,
        round_no: int,
        text: str,
        logical_time: int | None = None,
    ) -> None:
        self._memories[player_id].add_reflection(
            round_no=round_no,
            text=text,
            logical_time=logical_time,
        )
