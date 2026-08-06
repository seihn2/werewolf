from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from langgraph.graph import END, START, StateGraph

from .bus import AsyncMessageBus
from .cognition import CognitiveAgent, build_agent
from .llm import ChatBackend
from .memory import MemoryStore
from .models import (
    ActionType,
    AgentObservation,
    DecisionTrace,
    Faction,
    GameAction,
    GameResult,
    GameState,
    Phase,
    PlayerState,
    Role,
    Winner,
)

STANDARD_7P_ROLES = [
    Role.WEREWOLF,
    Role.WEREWOLF,
    Role.SEER,
    Role.DOCTOR,
    Role.VILLAGER,
    Role.VILLAGER,
    Role.VILLAGER,
]


class GameRuntime:
    """A single seven-player Werewolf game executed by a LangGraph state machine."""

    def __init__(
        self,
        *,
        seed: int = 42,
        max_rounds: int = 8,
        backend: ChatBackend | None = None,
        backend_by_faction: Mapping[Faction, ChatBackend | None] | None = None,
    ) -> None:
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        self.seed = seed
        self.max_rounds = max_rounds
        self.backend = backend
        self.backend_by_faction = dict(backend_by_faction or {})
        self.rng = random.Random(seed)
        self._has_run = False
        self.bus = AsyncMessageBus()
        self.initial_state = self._new_game_state()
        player_ids = list(self.initial_state["players"])
        self.bus.register(player_ids)
        self.memories = MemoryStore(player_ids)
        self.agents: dict[str, CognitiveAgent] = {}
        for index, player_id in enumerate(player_ids):
            faction = self.initial_state["players"][player_id].faction
            if faction in self.backend_by_faction:
                player_backend = self.backend_by_faction[faction]
            else:
                player_backend = backend
            self.agents[player_id] = build_agent(
                seed=seed + index * 1009,
                backend=player_backend,
            )
        self.graph = self._build_graph()

    def _new_game_state(self) -> GameState:
        roles = STANDARD_7P_ROLES.copy()
        self.rng.shuffle(roles)
        players = {
            f"player_{index}": PlayerState(
                player_id=f"player_{index}",
                name=f"Player {index}",
                role=role,
            )
            for index, role in enumerate(roles)
        }
        return GameState(
            game_id=f"wolfplay-{self.seed:08x}",
            seed=self.seed,
            round_no=1,
            max_rounds=self.max_rounds,
            phase=Phase.SETUP,
            players=players,
            pending_actions={},
            night_victim=None,
            protected_player=None,
            eliminated_today=None,
            checkpoint="setup",
            winner=None,
            termination_reason=None,
            decision_traces=[],
            metadata={"ruleset": "standard_7p_v1"},
        )

    def _build_graph(self):
        workflow = StateGraph(GameState)
        workflow.add_node("setup", self._setup)
        workflow.add_node("night_werewolves", self._night_werewolves)
        workflow.add_node("night_seer", self._night_seer)
        workflow.add_node("night_doctor", self._night_doctor)
        workflow.add_node("resolve_night", self._resolve_night)
        workflow.add_node("day_announcement", self._day_announcement)
        workflow.add_node("day_discussion", self._day_discussion)
        workflow.add_node("day_vote", self._day_vote)
        workflow.add_node("resolve_vote", self._resolve_vote)
        workflow.add_node("check_outcome", self._check_outcome)
        workflow.add_node("advance_round", self._advance_round)

        workflow.add_edge(START, "setup")
        workflow.add_edge("setup", "night_werewolves")
        workflow.add_edge("night_werewolves", "night_seer")
        workflow.add_edge("night_seer", "night_doctor")
        workflow.add_edge("night_doctor", "resolve_night")
        workflow.add_edge("resolve_night", "check_outcome")
        workflow.add_conditional_edges(
            "check_outcome",
            self._route_after_outcome,
            {
                "end": END,
                "day": "day_announcement",
                "next_round": "advance_round",
            },
        )
        workflow.add_edge("day_announcement", "day_discussion")
        workflow.add_edge("day_discussion", "day_vote")
        workflow.add_edge("day_vote", "resolve_vote")
        workflow.add_edge("resolve_vote", "check_outcome")
        workflow.add_edge("advance_round", "night_werewolves")
        return workflow.compile()

    async def play(self) -> GameResult:
        if self._has_run:
            raise RuntimeError("a GameRuntime instance can only run one game")
        self._has_run = True
        final_state = await self.graph.ainvoke(
            self.initial_state,
            config={"recursion_limit": max(100, self.max_rounds * 16)},
        )
        winner = final_state["winner"] or Winner.DRAW
        return GameResult(
            game_id=final_state["game_id"],
            seed=final_state["seed"],
            rounds=final_state["round_no"],
            winner=winner,
            termination_reason=final_state["termination_reason"] or "unknown",
            players=final_state["players"],
            events=list(self.bus.events),
            decision_traces=final_state["decision_traces"],
        )

    async def _emit(
        self,
        *,
        state: GameState,
        topic: str,
        payload: dict[str, Any],
        phase: Phase | None = None,
        sender: str | None = None,
        audience: list[str] | tuple[str, ...] | None = None,
    ):
        event = await self.bus.publish(
            topic=topic,
            round_no=state["round_no"],
            phase=phase or state["phase"],
            payload=payload,
            sender=sender,
            audience=audience,
        )
        self.memories.observe(event)
        return event

    async def _setup(self, state: GameState) -> dict[str, Any]:
        player_ids = sorted(state["players"])
        await self._emit(
            state=state,
            topic="game_started",
            phase=Phase.SETUP,
            payload={"game_id": state["game_id"], "players": player_ids},
        )
        wolves = self._alive_ids_by_role(state, Role.WEREWOLF)
        for player_id, player in state["players"].items():
            await self._emit(
                state=state,
                topic="role_assignment",
                phase=Phase.SETUP,
                sender="moderator",
                audience=[player_id],
                payload={"player_id": player_id, "role": player.role.value},
            )
        await self._emit(
            state=state,
            topic="werewolf_team",
            phase=Phase.SETUP,
            sender="moderator",
            audience=wolves,
            payload={"members": wolves},
        )
        return {"phase": Phase.NIGHT_WEREWOLF, "checkpoint": "setup"}

    async def _night_werewolves(self, state: GameState) -> dict[str, Any]:
        wolves = self._alive_ids_by_role(state, Role.WEREWOLF)
        traces = list(state["decision_traces"])
        final_action: GameAction | None = None
        for wolf_id in wolves:
            observation = self._observation(state, wolf_id, Phase.NIGHT_WEREWOLF)
            trace = await self.agents[wolf_id].decide(observation)
            traces.append(trace)
            self._remember_reflection(trace)
            final_action = trace.action
            await self._emit(
                state=state,
                topic="werewolf_proposal",
                phase=Phase.NIGHT_WEREWOLF,
                sender=wolf_id,
                audience=wolves,
                payload={"target_id": trace.action.target_id},
            )

        pending = dict(state["pending_actions"])
        if final_action is not None:
            pending["werewolf_kill"] = final_action
        return {
            "phase": Phase.NIGHT_SEER,
            "pending_actions": pending,
            "decision_traces": traces,
        }

    async def _night_seer(self, state: GameState) -> dict[str, Any]:
        seers = self._alive_ids_by_role(state, Role.SEER)
        if not seers:
            return {"phase": Phase.NIGHT_DOCTOR}
        seer_id = seers[0]
        observation = self._observation(state, seer_id, Phase.NIGHT_SEER)
        trace = await self.agents[seer_id].decide(observation)
        self._remember_reflection(trace)
        target_id = trace.action.target_id
        is_werewolf = bool(target_id and state["players"][target_id].role is Role.WEREWOLF)
        await self._emit(
            state=state,
            topic="seer_result",
            phase=Phase.NIGHT_SEER,
            sender="moderator",
            audience=[seer_id],
            payload={"target_id": target_id, "is_werewolf": is_werewolf},
        )
        pending = dict(state["pending_actions"])
        pending["seer_check"] = trace.action
        return {
            "phase": Phase.NIGHT_DOCTOR,
            "pending_actions": pending,
            "decision_traces": [*state["decision_traces"], trace],
        }

    async def _night_doctor(self, state: GameState) -> dict[str, Any]:
        doctors = self._alive_ids_by_role(state, Role.DOCTOR)
        if not doctors:
            return {"phase": Phase.NIGHT_RESOLUTION}
        doctor_id = doctors[0]
        observation = self._observation(state, doctor_id, Phase.NIGHT_DOCTOR)
        trace = await self.agents[doctor_id].decide(observation)
        self._remember_reflection(trace)
        await self._emit(
            state=state,
            topic="doctor_choice",
            phase=Phase.NIGHT_DOCTOR,
            sender="moderator",
            audience=[doctor_id],
            payload={"target_id": trace.action.target_id},
        )
        pending = dict(state["pending_actions"])
        pending["doctor_protect"] = trace.action
        return {
            "phase": Phase.NIGHT_RESOLUTION,
            "pending_actions": pending,
            "decision_traces": [*state["decision_traces"], trace],
        }

    async def _resolve_night(self, state: GameState) -> dict[str, Any]:
        kill_action = state["pending_actions"].get("werewolf_kill")
        protect_action = state["pending_actions"].get("doctor_protect")
        target_id = kill_action.target_id if kill_action else None
        protected_id = protect_action.target_id if protect_action else None
        victim_id = target_id if target_id and target_id != protected_id else None
        players = self._copy_players(state)
        if victim_id and players[victim_id].alive:
            players[victim_id].alive = False
        await self._emit(
            state=state,
            topic="night_result",
            phase=Phase.NIGHT_RESOLUTION,
            sender="moderator",
            payload={"victim_id": victim_id, "nobody_died": victim_id is None},
        )
        return {
            "phase": Phase.NIGHT_RESOLUTION,
            "players": players,
            "night_victim": victim_id,
            "protected_player": protected_id,
            "checkpoint": "night",
        }

    async def _day_announcement(self, state: GameState) -> dict[str, Any]:
        await self._emit(
            state=state,
            topic="day_started",
            phase=Phase.DAY_ANNOUNCEMENT,
            sender="moderator",
            payload={
                "round_no": state["round_no"],
                "night_victim": state["night_victim"],
                "alive_players": self._alive_ids(state),
            },
        )
        return {"phase": Phase.DAY_DISCUSSION}

    async def _day_discussion(self, state: GameState) -> dict[str, Any]:
        traces = list(state["decision_traces"])
        for player_id in self._alive_ids(state):
            observation = self._observation(state, player_id, Phase.DAY_DISCUSSION)
            trace = await self.agents[player_id].decide(observation)
            traces.append(trace)
            self._remember_reflection(trace)
            await self._emit(
                state=state,
                topic="speech",
                phase=Phase.DAY_DISCUSSION,
                sender=player_id,
                payload={
                    "message": trace.action.message,
                    "strategy": trace.action.strategy,
                },
            )
        return {"phase": Phase.DAY_VOTE, "decision_traces": traces}

    async def _day_vote(self, state: GameState) -> dict[str, Any]:
        alive_ids = self._alive_ids(state)
        observations = {
            player_id: self._observation(state, player_id, Phase.DAY_VOTE)
            for player_id in alive_ids
        }
        traces = await asyncio.gather(
            *(self.agents[player_id].decide(observations[player_id]) for player_id in alive_ids)
        )
        pending = dict(state["pending_actions"])
        for trace in traces:
            self._remember_reflection(trace)
            pending[f"vote:{trace.player_id}"] = trace.action
        for trace in traces:
            await self._emit(
                state=state,
                topic="vote_cast",
                phase=Phase.DAY_VOTE,
                sender=trace.player_id,
                payload={"target_id": trace.action.target_id},
            )
        return {
            "phase": Phase.VOTE_RESOLUTION,
            "pending_actions": pending,
            "decision_traces": [*state["decision_traces"], *traces],
        }

    async def _resolve_vote(self, state: GameState) -> dict[str, Any]:
        tally: dict[str, int] = {}
        for key, action in state["pending_actions"].items():
            if not key.startswith("vote:") or action.action_type is ActionType.ABSTAIN:
                continue
            if action.target_id:
                tally[action.target_id] = tally.get(action.target_id, 0) + 1

        eliminated_id: str | None = None
        players = self._copy_players(state)
        if tally:
            max_votes = max(tally.values())
            tied = sorted(player_id for player_id, votes in tally.items() if votes == max_votes)
            eliminated_id = self.rng.choice(tied)
            players[eliminated_id].alive = False
        else:
            tied = []

        await self._emit(
            state=state,
            topic="vote_result",
            phase=Phase.VOTE_RESOLUTION,
            sender="moderator",
            payload={
                "tally": tally,
                "tie_candidates": tied if len(tied) > 1 else [],
                "eliminated_id": eliminated_id,
            },
        )
        return {
            "players": players,
            "eliminated_today": eliminated_id,
            "checkpoint": "vote",
            "phase": Phase.VOTE_RESOLUTION,
        }

    async def _check_outcome(self, state: GameState) -> dict[str, Any]:
        alive = [player for player in state["players"].values() if player.alive]
        wolves = [player for player in alive if player.faction is Faction.WEREWOLF]
        villagers = [player for player in alive if player.faction is Faction.VILLAGE]
        winner: Winner | None = None
        reason: str | None = None

        if not wolves:
            winner = Winner.VILLAGE
            reason = "all_werewolves_eliminated"
        elif len(wolves) >= len(villagers):
            winner = Winner.WEREWOLF
            reason = "werewolf_parity"
        elif state["checkpoint"] == "vote" and state["round_no"] >= state["max_rounds"]:
            winner = Winner.DRAW
            reason = "max_rounds_reached"

        if winner is not None:
            await self._emit(
                state=state,
                topic="game_over",
                phase=Phase.GAME_OVER,
                sender="moderator",
                payload={"winner": winner.value, "reason": reason},
            )
            return {
                "winner": winner,
                "termination_reason": reason,
                "phase": Phase.GAME_OVER,
            }
        return {"winner": None, "termination_reason": None}

    async def _advance_round(self, state: GameState) -> dict[str, Any]:
        next_round = state["round_no"] + 1
        await self._emit(
            state=state,
            topic="round_advanced",
            phase=Phase.NIGHT_WEREWOLF,
            sender="moderator",
            payload={"round_no": next_round},
        )
        return {
            "round_no": next_round,
            "phase": Phase.NIGHT_WEREWOLF,
            "pending_actions": {},
            "night_victim": None,
            "protected_player": None,
            "eliminated_today": None,
            "checkpoint": "next_round",
        }

    @staticmethod
    def _route_after_outcome(state: GameState) -> str:
        if state["winner"] is not None:
            return "end"
        if state["checkpoint"] == "night":
            return "day"
        return "next_round"

    def _observation(self, state: GameState, player_id: str, phase: Phase) -> AgentObservation:
        player = state["players"][player_id]
        alive_players = tuple(
            state["players"][other_id].public_dict() for other_id in self._alive_ids(state)
        )
        teammate_ids: tuple[str, ...] = ()
        if player.role is Role.WEREWOLF:
            teammate_ids = tuple(
                other_id
                for other_id in self._alive_ids_by_role(state, Role.WEREWOLF)
                if other_id != player_id
            )
        memory = self.memories[player_id]
        role_beliefs = {
            target_id: {role.value: score for role, score in beliefs.items()}
            for target_id, beliefs in memory.semantic_beliefs.items()
        }
        return AgentObservation(
            game_id=state["game_id"],
            player_id=player_id,
            player_name=player.name,
            role=player.role,
            teammate_ids=teammate_ids,
            alive_players=alive_players,
            round_no=state["round_no"],
            phase=phase,
            legal_targets=tuple(self._legal_targets(state, player_id, phase)),
            events=self.bus.events_for(player_id),
            memory_context=memory.recall(query=phase.value),
            role_beliefs=role_beliefs,
        )

    def _legal_targets(self, state: GameState, player_id: str, phase: Phase) -> list[str]:
        alive_ids = self._alive_ids(state)
        if phase is Phase.NIGHT_WEREWOLF:
            return [
                target_id
                for target_id in alive_ids
                if state["players"][target_id].role is not Role.WEREWOLF
            ]
        if phase is Phase.NIGHT_SEER:
            return [target_id for target_id in alive_ids if target_id != player_id]
        if phase is Phase.NIGHT_DOCTOR:
            return alive_ids
        if phase in {Phase.DAY_DISCUSSION, Phase.DAY_VOTE}:
            return [target_id for target_id in alive_ids if target_id != player_id]
        return []

    @staticmethod
    def _copy_players(state: GameState) -> dict[str, PlayerState]:
        return {player_id: replace(player) for player_id, player in state["players"].items()}

    @staticmethod
    def _alive_ids(state: GameState) -> list[str]:
        return sorted(player_id for player_id, player in state["players"].items() if player.alive)

    @staticmethod
    def _alive_ids_by_role(state: GameState, role: Role) -> list[str]:
        return sorted(
            player_id
            for player_id, player in state["players"].items()
            if player.alive and player.role is role
        )

    def _remember_reflection(self, trace: DecisionTrace) -> None:
        if trace.reflection:
            self.memories.add_reflection(trace.player_id, trace.round_no, trace.reflection)
