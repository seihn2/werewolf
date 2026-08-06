from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, TypedDict


class Role(StrEnum):
    WEREWOLF = "werewolf"
    SEER = "seer"
    DOCTOR = "doctor"
    VILLAGER = "villager"


class Faction(StrEnum):
    WEREWOLF = "werewolf"
    VILLAGE = "village"


class Winner(StrEnum):
    WEREWOLF = "werewolf"
    VILLAGE = "village"
    DRAW = "draw"


class Phase(StrEnum):
    SETUP = "setup"
    NIGHT_WEREWOLF = "night_werewolf"
    NIGHT_SEER = "night_seer"
    NIGHT_DOCTOR = "night_doctor"
    NIGHT_RESOLUTION = "night_resolution"
    DAY_ANNOUNCEMENT = "day_announcement"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTE = "day_vote"
    VOTE_RESOLUTION = "vote_resolution"
    GAME_OVER = "game_over"


class ActionType(StrEnum):
    KILL = "kill"
    CHECK = "check"
    PROTECT = "protect"
    SPEAK = "speak"
    VOTE = "vote"
    ABSTAIN = "abstain"


@dataclass(slots=True)
class PlayerState:
    player_id: str
    name: str
    role: Role
    alive: bool = True

    @property
    def faction(self) -> Faction:
        if self.role is Role.WEREWOLF:
            return Faction.WEREWOLF
        return Faction.VILLAGE

    def public_dict(self) -> dict[str, Any]:
        return {"player_id": self.player_id, "name": self.name, "alive": self.alive}


@dataclass(frozen=True, slots=True)
class GameEvent:
    logical_time: int
    topic: str
    round_no: int
    phase: Phase
    payload: dict[str, Any]
    sender: str | None = None
    audience: tuple[str, ...] | None = None

    def visible_to(self, player_id: str) -> bool:
        return self.audience is None or player_id in self.audience

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["phase"] = self.phase.value
        return data


@dataclass(frozen=True, slots=True)
class CandidateAction:
    action_type: ActionType
    strategy: str
    target_id: str | None = None
    message: str = ""
    rationale: str = ""

    def response_text(self) -> str:
        if self.action_type is ActionType.SPEAK:
            return self.message
        target = self.target_id or "abstain"
        return f'{{"action":"{self.action_type.value}","target":"{target}"}}'

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action_type"] = self.action_type.value
        return data


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    score: float
    legal: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GameAction:
    actor_id: str
    action_type: ActionType
    target_id: str | None = None
    message: str = ""
    strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action_type"] = self.action_type.value
        return data


@dataclass(frozen=True, slots=True)
class AgentObservation:
    game_id: str
    player_id: str
    player_name: str
    role: Role
    teammate_ids: tuple[str, ...]
    alive_players: tuple[dict[str, Any], ...]
    round_no: int
    phase: Phase
    legal_targets: tuple[str, ...]
    events: tuple[GameEvent, ...]
    memory_context: str
    role_beliefs: dict[str, dict[str, float]]

    def prompt(self) -> str:
        event_lines = [
            f"t={event.logical_time} {event.topic}: {event.payload}" for event in self.events[-24:]
        ]
        alive = ", ".join(player["player_id"] for player in self.alive_players)
        targets = ", ".join(self.legal_targets) or "none"
        teammates = ", ".join(self.teammate_ids) or "unknown"
        return "\n".join(
            [
                f"You are {self.player_id} ({self.player_name}), role={self.role.value}.",
                f"Round={self.round_no}, phase={self.phase.value}.",
                f"Alive players: {alive}.",
                f"Known teammates: {teammates}.",
                f"Legal targets: {targets}.",
                "Visible events:",
                *event_lines,
                "Private memory:",
                self.memory_context or "(empty)",
                f"Role beliefs: {self.role_beliefs}",
            ]
        )


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    player_id: str
    role: Role
    round_no: int
    phase: Phase
    observation_prompt: str
    candidates: tuple[CandidateAction, ...]
    evaluations: tuple[CandidateEvaluation, ...]
    selected_index: int
    action: GameAction
    reflection: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "role": self.role.value,
            "round_no": self.round_no,
            "phase": self.phase.value,
            "observation_prompt": self.observation_prompt,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "selected_index": self.selected_index,
            "action": self.action.to_dict(),
            "reflection": self.reflection,
        }


class GameState(TypedDict):
    game_id: str
    seed: int
    round_no: int
    max_rounds: int
    phase: Phase
    players: dict[str, PlayerState]
    pending_actions: dict[str, GameAction]
    night_victim: str | None
    protected_player: str | None
    eliminated_today: str | None
    checkpoint: str
    winner: Winner | None
    termination_reason: str | None
    decision_traces: list[DecisionTrace]
    metadata: dict[str, Any]


@dataclass(slots=True)
class GameResult:
    game_id: str
    seed: int
    rounds: int
    winner: Winner
    termination_reason: str
    players: dict[str, PlayerState]
    events: list[GameEvent] = field(default_factory=list)
    decision_traces: list[DecisionTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "rounds": self.rounds,
            "winner": self.winner.value,
            "termination_reason": self.termination_reason,
            "players": {
                player_id: {
                    "player_id": player.player_id,
                    "name": player.name,
                    "role": player.role.value,
                    "alive": player.alive,
                }
                for player_id, player in self.players.items()
            },
            "events": [event.to_dict() for event in self.events],
            "decision_traces": [trace.to_dict() for trace in self.decision_traces],
        }
