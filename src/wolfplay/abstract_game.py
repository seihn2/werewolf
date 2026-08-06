from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass, field
from typing import Any, Literal

from .latent import LatentStrategySpace, TextEmbedder
from .models import ActionType, Faction, Phase, Role, Winner

PLAYER_COUNT = 7
PLAYER_IDS = tuple(f"player_{index}" for index in range(PLAYER_COUNT))
STANDARD_7P_ROLES = (
    Role.WEREWOLF,
    Role.WEREWOLF,
    Role.SEER,
    Role.DOCTOR,
    Role.VILLAGER,
    Role.VILLAGER,
    Role.VILLAGER,
)
_DECISION_PHASES = (
    Phase.NIGHT_WEREWOLF,
    Phase.NIGHT_SEER,
    Phase.NIGHT_DOCTOR,
    Phase.DAY_DISCUSSION,
    Phase.DAY_VOTE,
)


@dataclass(frozen=True, slots=True)
class AbstractAction:
    action_id: int
    kind: Literal["target", "abstain", "speech"]
    label: str
    target_index: int | None = None
    role: Role | None = None
    cluster_id: int | None = None
    representative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "label": self.label,
            "target_index": self.target_index,
            "role": self.role.value if self.role is not None else None,
            "cluster_id": self.cluster_id,
            "representative": self.representative,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> AbstractAction:
        raw_role = record.get("role")
        return cls(
            action_id=int(record["action_id"]),
            kind=record["kind"],
            label=str(record["label"]),
            target_index=(
                int(record["target_index"]) if record.get("target_index") is not None else None
            ),
            role=Role(raw_role) if raw_role is not None else None,
            cluster_id=(
                int(record["cluster_id"]) if record.get("cluster_id") is not None else None
            ),
            representative=str(record.get("representative", "")),
        )


@dataclass(frozen=True, slots=True)
class ActionCatalog:
    actions: tuple[AbstractAction, ...]

    @classmethod
    def from_latent_space(cls, latent_space: LatentStrategySpace) -> ActionCatalog:
        actions: list[AbstractAction] = []
        for target_index, player_id in enumerate(PLAYER_IDS):
            actions.append(
                AbstractAction(
                    action_id=len(actions),
                    kind="target",
                    label=f"target:{player_id}",
                    target_index=target_index,
                )
            )
        actions.append(
            AbstractAction(action_id=len(actions), kind="abstain", label="target:abstain")
        )
        for role in Role:
            role_space = latent_space.roles[role]
            for cluster in sorted(role_space.clusters, key=lambda item: item.cluster_id):
                actions.append(
                    AbstractAction(
                        action_id=len(actions),
                        kind="speech",
                        label=f"speech:{role.value}:{cluster.cluster_id}",
                        role=role,
                        cluster_id=cluster.cluster_id,
                        representative=cluster.representative,
                    )
                )
        return cls(actions=tuple(actions))

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> ActionCatalog:
        actions = tuple(AbstractAction.from_dict(item) for item in record["actions"])
        if tuple(action.action_id for action in actions) != tuple(range(len(actions))):
            raise ValueError("action catalog IDs must be contiguous and ordered")
        return cls(actions=actions)

    def to_dict(self) -> dict[str, Any]:
        return {"actions": [action.to_dict() for action in self.actions]}

    @property
    def size(self) -> int:
        return len(self.actions)

    @property
    def abstain_action_id(self) -> int:
        return next(action.action_id for action in self.actions if action.kind == "abstain")

    def action(self, action_id: int) -> AbstractAction:
        if not 0 <= action_id < len(self.actions):
            raise ValueError(f"action_id {action_id} is outside the action catalog")
        return self.actions[action_id]

    def target_action_id(self, target_index: int) -> int:
        if not 0 <= target_index < PLAYER_COUNT:
            raise ValueError("target_index is outside the player range")
        return target_index

    def speech_action_ids(self, role: Role) -> tuple[int, ...]:
        return tuple(
            action.action_id
            for action in self.actions
            if action.kind == "speech" and action.role is role
        )

    def speech_action_id(self, role: Role, cluster_id: int) -> int:
        for action in self.actions:
            if action.kind == "speech" and action.role is role and action.cluster_id == cluster_id:
                return action.action_id
        raise KeyError(f"no speech action for role={role.value}, cluster={cluster_id}")

    def candidate_action_id(
        self,
        *,
        role: Role,
        candidate: dict[str, Any],
        latent_space: LatentStrategySpace,
        embedder: TextEmbedder,
    ) -> int:
        action_type = ActionType(candidate["action_type"])
        if action_type is ActionType.SPEAK:
            message = candidate.get("message")
            if not isinstance(message, str) or not message.strip():
                raise ValueError("speak candidate must contain a non-empty message")
            cluster_id = latent_space.assign(role, message, embedder)
            return self.speech_action_id(role, cluster_id)
        if action_type is ActionType.ABSTAIN:
            return self.abstain_action_id
        target_id = candidate.get("target_id")
        if not isinstance(target_id, str) or target_id not in PLAYER_IDS:
            raise ValueError("non-speech candidate must contain a valid target_id")
        return self.target_action_id(PLAYER_IDS.index(target_id))


@dataclass(frozen=True, slots=True)
class RewardConfig:
    win: float = 300.0
    survival_per_round: float = 5.0
    village_correct_vote: float = 20.0
    village_incorrect_vote: float = -20.0
    eliminated: float = -10.0
    opponent_eliminated: float = 5.0
    teammate_eliminated: float = -5.0


@dataclass(frozen=True, slots=True)
class AbstractGameConfig:
    max_rounds: int = 8
    history_window: int = 8
    role_assignment_limit: int | None = None
    role_assignment_seed: int = 42
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.history_window <= 0:
            raise ValueError("history_window must be positive")
        if self.role_assignment_limit is not None and self.role_assignment_limit <= 0:
            raise ValueError("role_assignment_limit must be positive")


class AbstractWerewolfGame:
    """Finite seven-player extensive-form game over discrete latent speech actions."""

    def __init__(
        self,
        latent_space: LatentStrategySpace,
        *,
        config: AbstractGameConfig | None = None,
        action_catalog: ActionCatalog | None = None,
    ) -> None:
        self.latent_space = latent_space
        self.config = config or AbstractGameConfig()
        self.action_catalog = action_catalog or ActionCatalog.from_latent_space(latent_space)
        assignments = sorted(set(itertools.permutations(STANDARD_7P_ROLES)))
        if (
            self.config.role_assignment_limit is not None
            and self.config.role_assignment_limit < len(assignments)
        ):
            rng = random.Random(self.config.role_assignment_seed)
            assignments = rng.sample(assignments, self.config.role_assignment_limit)
            assignments.sort()
        self.role_assignments = tuple(assignments)

    @property
    def num_players(self) -> int:
        return PLAYER_COUNT

    @property
    def num_actions(self) -> int:
        return self.action_catalog.size

    @property
    def information_state_size(self) -> int:
        return (
            len(Role)
            + PLAYER_COUNT
            + len(_DECISION_PHASES)
            + 1
            + PLAYER_COUNT
            + PLAYER_COUNT
            + PLAYER_COUNT
            + (PLAYER_COUNT + 1)
            + (PLAYER_COUNT + 1)
            + (PLAYER_COUNT + 1)
            + (PLAYER_COUNT + 1)
            + self.config.history_window * self.num_actions
        )

    def new_initial_state(self) -> AbstractWerewolfState:
        return AbstractWerewolfState(game=self)

    def new_state_with_roles(self, roles: tuple[Role, ...]) -> AbstractWerewolfState:
        if len(roles) != PLAYER_COUNT or sorted(roles) != sorted(STANDARD_7P_ROLES):
            raise ValueError("roles must contain the standard seven-player role multiset")
        state = AbstractWerewolfState(
            game=self,
            roles=roles,
            phase=Phase.NIGHT_WEREWOLF,
            chance_kind=None,
        )
        state._advance_forced()
        return state


@dataclass(slots=True)
class AbstractWerewolfState:
    game: AbstractWerewolfGame
    roles: tuple[Role, ...] | None = None
    alive: list[bool] = field(default_factory=lambda: [True] * PLAYER_COUNT)
    round_no: int = 1
    phase: Phase = Phase.SETUP
    actor_cursor: int = 0
    pending_kill: int | None = None
    pending_protect: int | None = None
    pending_votes: dict[int, int | None] = field(default_factory=dict)
    seer_knowledge: list[dict[int, bool]] = field(
        default_factory=lambda: [dict() for _ in range(PLAYER_COUNT)]
    )
    public_history: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=lambda: [0.0] * PLAYER_COUNT)
    last_night_victim: int | None = None
    last_eliminated: int | None = None
    terminal_winner: Winner | None = None
    chance_kind: Literal["roles", "vote_tie"] | None = "roles"
    tie_candidates: tuple[int, ...] = ()

    def clone(self) -> AbstractWerewolfState:
        return AbstractWerewolfState(
            game=self.game,
            roles=self.roles,
            alive=list(self.alive),
            round_no=self.round_no,
            phase=self.phase,
            actor_cursor=self.actor_cursor,
            pending_kill=self.pending_kill,
            pending_protect=self.pending_protect,
            pending_votes=dict(self.pending_votes),
            seer_knowledge=[dict(knowledge) for knowledge in self.seer_knowledge],
            public_history=list(self.public_history),
            rewards=list(self.rewards),
            last_night_victim=self.last_night_victim,
            last_eliminated=self.last_eliminated,
            terminal_winner=self.terminal_winner,
            chance_kind=self.chance_kind,
            tie_candidates=self.tie_candidates,
        )

    @property
    def is_terminal(self) -> bool:
        return self.terminal_winner is not None

    @property
    def is_chance_node(self) -> bool:
        return self.chance_kind is not None and not self.is_terminal

    @property
    def is_decision_node(self) -> bool:
        return not self.is_terminal and not self.is_chance_node

    @property
    def current_player(self) -> int:
        if not self.is_decision_node:
            raise RuntimeError("current_player is only available at decision nodes")
        actors = self._phase_actors()
        if not 0 <= self.actor_cursor < len(actors):
            raise RuntimeError("decision state has no current actor")
        return actors[self.actor_cursor]

    @property
    def current_role(self) -> Role:
        if self.roles is None:
            raise RuntimeError("roles have not been assigned")
        return self.roles[self.current_player]

    def chance_outcomes(self) -> tuple[tuple[int, float], ...]:
        if not self.is_chance_node:
            raise RuntimeError("chance_outcomes is only available at chance nodes")
        if self.chance_kind == "roles":
            probability = 1.0 / len(self.game.role_assignments)
            return tuple((index, probability) for index in range(len(self.game.role_assignments)))
        probability = 1.0 / len(self.tie_candidates)
        return tuple((candidate, probability) for candidate in self.tie_candidates)

    def legal_actions(self) -> tuple[int, ...]:
        if not self.is_decision_node:
            return ()
        actor = self.current_player
        role = self.current_role
        catalog = self.game.action_catalog
        if self.phase is Phase.NIGHT_WEREWOLF:
            return tuple(
                catalog.target_action_id(index)
                for index in range(PLAYER_COUNT)
                if self.alive[index] and self.roles[index] is not Role.WEREWOLF
            )
        if self.phase is Phase.NIGHT_SEER:
            return tuple(
                catalog.target_action_id(index)
                for index in range(PLAYER_COUNT)
                if self.alive[index] and index != actor
            )
        if self.phase is Phase.NIGHT_DOCTOR:
            return tuple(
                catalog.target_action_id(index)
                for index in range(PLAYER_COUNT)
                if self.alive[index]
            )
        if self.phase is Phase.DAY_DISCUSSION:
            return catalog.speech_action_ids(role)
        if self.phase is Phase.DAY_VOTE:
            targets = [
                catalog.target_action_id(index)
                for index in range(PLAYER_COUNT)
                if self.alive[index] and index != actor
            ]
            return (*targets, catalog.abstain_action_id)
        raise RuntimeError(f"unsupported decision phase: {self.phase.value}")

    def apply_action(self, action_id: int) -> None:
        if self.is_terminal:
            raise RuntimeError("cannot apply actions to a terminal state")
        if self.is_chance_node:
            self._apply_chance_action(action_id)
            return
        legal_actions = self.legal_actions()
        if action_id not in legal_actions:
            raise ValueError(f"action {action_id} is not legal in phase {self.phase.value}")
        actor = self.current_player
        action = self.game.action_catalog.action(action_id)
        if self.phase is Phase.NIGHT_WEREWOLF:
            self.pending_kill = action.target_index
        elif self.phase is Phase.NIGHT_SEER:
            target = _required_target(action)
            self.seer_knowledge[actor][target] = self.roles[target] is Role.WEREWOLF
        elif self.phase is Phase.NIGHT_DOCTOR:
            self.pending_protect = _required_target(action)
        elif self.phase is Phase.DAY_DISCUSSION:
            self.public_history.append(action_id)
        elif self.phase is Phase.DAY_VOTE:
            target = action.target_index if action.kind == "target" else None
            self.pending_votes[actor] = target
            self._apply_vote_reward(actor, target)
        else:
            raise RuntimeError(f"unsupported decision phase: {self.phase.value}")
        self.actor_cursor += 1
        self._advance_forced()

    def returns(self) -> tuple[float, ...]:
        if not self.is_terminal:
            raise RuntimeError("returns are only available at terminal states")
        return tuple(self.rewards)

    def information_state_tensor(self, player_index: int | None = None) -> tuple[float, ...]:
        if self.roles is None:
            raise RuntimeError("information state is unavailable before role assignment")
        player = self.current_player if player_index is None else player_index
        if not 0 <= player < PLAYER_COUNT:
            raise ValueError("player_index is outside the player range")
        role = self.roles[player]
        vector: list[float] = []
        vector.extend(_one_hot(list(Role).index(role), len(Role)))
        vector.extend(_one_hot(player, PLAYER_COUNT))
        vector.extend(
            _one_hot(_DECISION_PHASES.index(self.phase), len(_DECISION_PHASES))
            if self.phase in _DECISION_PHASES
            else [0.0] * len(_DECISION_PHASES)
        )
        vector.append(self.round_no / self.game.config.max_rounds)
        vector.extend(1.0 if alive else 0.0 for alive in self.alive)

        known_wolf = [0.0] * PLAYER_COUNT
        known_not_wolf = [0.0] * PLAYER_COUNT
        if role is Role.WEREWOLF:
            for index, other_role in enumerate(self.roles):
                if other_role is Role.WEREWOLF:
                    known_wolf[index] = 1.0
        for target, is_wolf in self.seer_knowledge[player].items():
            if is_wolf:
                known_wolf[target] = 1.0
            else:
                known_not_wolf[target] = 1.0
        vector.extend(known_wolf)
        vector.extend(known_not_wolf)
        vector.extend(_optional_player_one_hot(self.last_night_victim))
        vector.extend(_optional_player_one_hot(self.last_eliminated))
        vector.extend(
            _optional_player_one_hot(self.pending_kill if role is Role.WEREWOLF else None)
        )
        vector.extend(
            _optional_player_one_hot(self.pending_protect if role is Role.DOCTOR else None)
        )

        recent_history = self.public_history[-self.game.config.history_window :]
        padding = self.game.config.history_window - len(recent_history)
        vector.extend([0.0] * (padding * self.game.num_actions))
        for action_id in recent_history:
            vector.extend(_one_hot(action_id, self.game.num_actions))
        if len(vector) != self.game.information_state_size:
            raise RuntimeError(
                f"information state has length {len(vector)}, "
                f"expected {self.game.information_state_size}"
            )
        return tuple(vector)

    def information_state_key(self, player_index: int | None = None) -> str:
        vector = self.information_state_tensor(player_index)
        payload = json.dumps(vector, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _apply_chance_action(self, action_id: int) -> None:
        outcomes = {outcome for outcome, _ in self.chance_outcomes()}
        if action_id not in outcomes:
            raise ValueError(f"invalid chance outcome: {action_id}")
        if self.chance_kind == "roles":
            self.roles = self.game.role_assignments[action_id]
            self.chance_kind = None
            self.phase = Phase.NIGHT_WEREWOLF
            self.actor_cursor = 0
            self._advance_forced()
            return
        eliminated = action_id
        self.chance_kind = None
        self.tie_candidates = ()
        self._finish_vote(eliminated)

    def _advance_forced(self) -> None:
        while not self.is_terminal and not self.is_chance_node:
            actors = self._phase_actors()
            if self.actor_cursor < len(actors):
                return
            self.actor_cursor = 0
            if self.phase is Phase.NIGHT_WEREWOLF:
                self.phase = Phase.NIGHT_SEER
            elif self.phase is Phase.NIGHT_SEER:
                self.phase = Phase.NIGHT_DOCTOR
            elif self.phase is Phase.NIGHT_DOCTOR:
                self._resolve_night()
            elif self.phase is Phase.DAY_DISCUSSION:
                self.phase = Phase.DAY_VOTE
                self.pending_votes = {}
            elif self.phase is Phase.DAY_VOTE:
                self._resolve_vote()
            else:
                raise RuntimeError(f"cannot advance unsupported phase {self.phase.value}")

    def _phase_actors(self) -> tuple[int, ...]:
        if self.roles is None:
            return ()
        if self.phase is Phase.NIGHT_WEREWOLF:
            return tuple(
                index
                for index, role in enumerate(self.roles)
                if self.alive[index] and role is Role.WEREWOLF
            )
        if self.phase is Phase.NIGHT_SEER:
            return tuple(
                index
                for index, role in enumerate(self.roles)
                if self.alive[index] and role is Role.SEER
            )
        if self.phase is Phase.NIGHT_DOCTOR:
            return tuple(
                index
                for index, role in enumerate(self.roles)
                if self.alive[index] and role is Role.DOCTOR
            )
        if self.phase in {Phase.DAY_DISCUSSION, Phase.DAY_VOTE}:
            return tuple(index for index, alive in enumerate(self.alive) if alive)
        return ()

    def _resolve_night(self) -> None:
        victim = (
            self.pending_kill
            if self.pending_kill is not None and self.pending_kill != self.pending_protect
            else None
        )
        self.last_night_victim = victim
        self.last_eliminated = None
        if victim is not None and self.alive[victim]:
            self._eliminate(victim)
        self.pending_kill = None
        self.pending_protect = None
        if self._finish_if_terminal():
            return
        self.phase = Phase.DAY_DISCUSSION

    def _resolve_vote(self) -> None:
        tally: dict[int, int] = {}
        for target in self.pending_votes.values():
            if target is not None:
                tally[target] = tally.get(target, 0) + 1
        if not tally:
            self._finish_vote(None)
            return
        max_votes = max(tally.values())
        tied = tuple(sorted(target for target, votes in tally.items() if votes == max_votes))
        if len(tied) > 1:
            self.phase = Phase.VOTE_RESOLUTION
            self.chance_kind = "vote_tie"
            self.tie_candidates = tied
            return
        self._finish_vote(tied[0])

    def _finish_vote(self, eliminated: int | None) -> None:
        self.last_eliminated = eliminated
        if eliminated is not None and self.alive[eliminated]:
            self._eliminate(eliminated)
        rewards = self.game.config.rewards
        for index, alive in enumerate(self.alive):
            if alive:
                self.rewards[index] += rewards.survival_per_round
        self.pending_votes = {}
        if self._finish_if_terminal():
            return
        if self.round_no >= self.game.config.max_rounds:
            self._set_terminal(Winner.DRAW)
            return
        self.round_no += 1
        self.phase = Phase.NIGHT_WEREWOLF
        self.actor_cursor = 0
        self._advance_forced()

    def _apply_vote_reward(self, actor: int, target: int | None) -> None:
        if (
            self.roles is None
            or _role_faction(self.roles[actor]) is Faction.WEREWOLF
            or target is None
        ):
            return
        rewards = self.game.config.rewards
        if self.roles[target] is Role.WEREWOLF:
            self.rewards[actor] += rewards.village_correct_vote
        else:
            self.rewards[actor] += rewards.village_incorrect_vote

    def _eliminate(self, player: int) -> None:
        if self.roles is None:
            raise RuntimeError("cannot eliminate a player before role assignment")
        self.alive[player] = False
        rewards = self.game.config.rewards
        self.rewards[player] += rewards.eliminated
        eliminated_faction = _role_faction(self.roles[player])
        for index, role in enumerate(self.roles):
            if index == player:
                continue
            if _role_faction(role) is eliminated_faction:
                self.rewards[index] += rewards.teammate_eliminated
            else:
                self.rewards[index] += rewards.opponent_eliminated

    def _finish_if_terminal(self) -> bool:
        if self.roles is None:
            return False
        alive_wolves = sum(
            self.alive[index] and role is Role.WEREWOLF for index, role in enumerate(self.roles)
        )
        alive_village = sum(
            self.alive[index] and role is not Role.WEREWOLF for index, role in enumerate(self.roles)
        )
        if alive_wolves == 0:
            self._set_terminal(Winner.VILLAGE)
            return True
        if alive_wolves >= alive_village:
            self._set_terminal(Winner.WEREWOLF)
            return True
        return False

    def _set_terminal(self, winner: Winner) -> None:
        if self.roles is None:
            raise RuntimeError("cannot end the game before role assignment")
        self.terminal_winner = winner
        self.phase = Phase.GAME_OVER
        self.chance_kind = None
        if winner is Winner.DRAW:
            return
        rewards = self.game.config.rewards
        for index, role in enumerate(self.roles):
            won = (role is Role.WEREWOLF and winner is Winner.WEREWOLF) or (
                role is not Role.WEREWOLF and winner is Winner.VILLAGE
            )
            self.rewards[index] += rewards.win if won else -rewards.win


def _required_target(action: AbstractAction) -> int:
    if action.kind != "target" or action.target_index is None:
        raise ValueError("the selected action does not contain a target")
    return action.target_index


def _one_hot(index: int, size: int) -> list[float]:
    if not 0 <= index < size:
        raise ValueError("one-hot index is outside the vector")
    vector = [0.0] * size
    vector[index] = 1.0
    return vector


def _optional_player_one_hot(player: int | None) -> list[float]:
    index = PLAYER_COUNT if player is None else player
    return _one_hot(index, PLAYER_COUNT + 1)


def _role_faction(role: Role) -> Faction:
    return Faction.WEREWOLF if role is Role.WEREWOLF else Faction.VILLAGE
