from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from .abstract_game import PLAYER_IDS, AbstractWerewolfState
from .latent import LatentStrategySpace, TextEmbedder, load_jsonl_records
from .models import ActionType, Phase, Role, Winner
from .training.deep_cfr import DeepCFRPolicy


def build_cfr_dpo_dataset(
    *,
    input_path: Path,
    checkpoint: Path,
    output_path: Path,
    embedder: TextEmbedder,
    winning_only: bool = False,
    device: str = "cpu",
) -> int:
    policy = DeepCFRPolicy.load(checkpoint, device=device)
    latent_space = policy.game.latent_space
    records = load_jsonl_records(input_path)
    pairs = [
        pair
        for record_index, record in enumerate(records, start=1)
        for pair in cfr_preference_pairs_from_game(
            record,
            policy=policy,
            latent_space=latent_space,
            embedder=embedder,
            winning_only=winning_only,
            context=f"{input_path}: record {record_index}",
        )
    ]
    if not pairs:
        raise ValueError(f"no CFR preference pairs were produced: {input_path}")
    _atomic_write_jsonl(output_path, pairs)
    return len(pairs)


def cfr_preference_pairs_from_game(
    record: dict[str, Any],
    *,
    policy: DeepCFRPolicy,
    latent_space: LatentStrategySpace,
    embedder: TextEmbedder,
    winning_only: bool = False,
    context: str = "game record",
) -> list[dict[str, Any]]:
    winner = Winner(record["winner"])
    roles = _roles_from_record(record, context=context)
    state = policy.game.new_state_with_roles(roles)
    vote_outcomes = _vote_outcomes(record)
    traces = record.get("decision_traces")
    if not isinstance(traces, list):
        raise ValueError(f"{context}.decision_traces must be a list")

    pairs: list[dict[str, Any]] = []
    for trace_index, trace in enumerate(traces):
        trace_context = f"{context}.decision_traces[{trace_index}]"
        if not isinstance(trace, dict):
            raise ValueError(f"{trace_context} must be an object")
        _resolve_recorded_chance(state, vote_outcomes, trace_context)
        if not state.is_decision_node:
            raise ValueError(f"{trace_context}: abstract replay is not at a decision node")

        player_id = trace.get("player_id")
        if player_id not in PLAYER_IDS:
            raise ValueError(f"{trace_context}.player_id is invalid")
        expected_player = PLAYER_IDS[state.current_player]
        phase = Phase(trace["phase"])
        if player_id != expected_player or phase is not state.phase:
            raise ValueError(
                f"{trace_context}: replay mismatch, expected "
                f"{expected_player}/{state.phase.value}, "
                f"got {player_id}/{phase.value}"
            )
        role = Role(trace["role"])
        if role is not state.current_role:
            raise ValueError(f"{trace_context}: replay role does not match the assigned role")

        candidates = trace.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError(f"{trace_context}.candidates must contain at least two entries")
        prompt = trace.get("observation_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{trace_context}.observation_prompt must be non-empty")

        predicted = policy.predicted_advantages(state)
        ranked: list[tuple[float, int, int, str]] = []
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise ValueError(f"{trace_context}.candidates[{candidate_index}] must be an object")
            response = _candidate_response(candidate, trace_context, candidate_index)
            try:
                action_id = policy.game.action_catalog.candidate_action_id(
                    role=role,
                    candidate=candidate,
                    latent_space=latent_space,
                    embedder=embedder,
                )
            except (KeyError, ValueError):
                action_id = -1
            advantage = predicted.get(action_id, -math.inf)
            ranked.append((advantage, -candidate_index, action_id, response))

        if not winning_only or _role_won(role, winner):
            ranked.sort(reverse=True)
            chosen = ranked[0]
            rejected = next(
                (
                    item
                    for item in reversed(ranked)
                    if item[3].strip() != chosen[3].strip() and item[2] != chosen[2]
                ),
                None,
            )
            if rejected is not None and math.isfinite(chosen[0]):
                pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen[3],
                        "rejected": rejected[3],
                        "metadata": {
                            "source": "deep_cfr_advantage",
                            "game_id": record.get("game_id"),
                            "player_id": player_id,
                            "role": role.value,
                            "round_no": int(trace["round_no"]),
                            "phase": phase.value,
                            "chosen_action_id": chosen[2],
                            "rejected_action_id": rejected[2],
                            "chosen_advantage": chosen[0],
                            "rejected_advantage": (
                                rejected[0] if math.isfinite(rejected[0]) else None
                            ),
                            "deep_cfr_iteration": policy.trainer.iteration,
                        },
                    }
                )

        action = trace.get("action")
        if not isinstance(action, dict):
            raise ValueError(f"{trace_context}.action must be an object")
        actual_action_id = policy.game.action_catalog.candidate_action_id(
            role=role,
            candidate=action,
            latent_space=latent_space,
            embedder=embedder,
        )
        if actual_action_id not in state.legal_actions():
            selected_index = trace.get("selected_index")
            if isinstance(selected_index, int) and 0 <= selected_index < len(candidates):
                actual_action_id = policy.game.action_catalog.candidate_action_id(
                    role=role,
                    candidate=candidates[selected_index],
                    latent_space=latent_space,
                    embedder=embedder,
                )
        state.apply_action(actual_action_id)

    _resolve_recorded_chance(state, vote_outcomes, context)
    return pairs


def _resolve_recorded_chance(
    state: AbstractWerewolfState,
    vote_outcomes: dict[int, int | None],
    context: str,
) -> None:
    while state.is_chance_node:
        if state.chance_kind != "vote_tie":
            raise ValueError(f"{context}: unexpected chance node {state.chance_kind}")
        eliminated = vote_outcomes.get(state.round_no)
        if eliminated is None or eliminated not in state.tie_candidates:
            raise ValueError(
                f"{context}: recorded vote outcome does not resolve round {state.round_no} tie"
            )
        state.apply_action(eliminated)


def _roles_from_record(record: dict[str, Any], *, context: str) -> tuple[Role, ...]:
    players = record.get("players")
    if not isinstance(players, dict):
        raise ValueError(f"{context}.players must be an object")
    roles: list[Role] = []
    for player_id in PLAYER_IDS:
        player = players.get(player_id)
        if not isinstance(player, dict):
            raise ValueError(f"{context}.players.{player_id} must be an object")
        roles.append(Role(player["role"]))
    return tuple(roles)


def _vote_outcomes(record: dict[str, Any]) -> dict[int, int | None]:
    outcomes: dict[int, int | None] = {}
    events = record.get("events")
    if not isinstance(events, list):
        return outcomes
    for event in events:
        if not isinstance(event, dict) or event.get("topic") != "vote_result":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        eliminated_id = payload.get("eliminated_id")
        outcomes[int(event["round_no"])] = (
            PLAYER_IDS.index(eliminated_id) if eliminated_id in PLAYER_IDS else None
        )
    return outcomes


def _candidate_response(candidate: dict[str, Any], context: str, index: int) -> str:
    action_type = ActionType(candidate["action_type"])
    if action_type is ActionType.SPEAK:
        message = candidate.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"{context}.candidates[{index}].message must be non-empty")
        return message
    target = candidate.get("target_id")
    return json.dumps(
        {"action": action_type.value, "target": target or "abstain"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _role_won(role: Role, winner: Winner) -> bool:
    if winner is Winner.DRAW:
        return False
    if role is Role.WEREWOLF:
        return winner is Winner.WEREWOLF
    return winner is Winner.VILLAGE


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
