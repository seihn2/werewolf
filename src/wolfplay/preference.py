from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from .models import ActionType, Role, Winner


def build_dpo_dataset(
    *,
    input_path: Path,
    output_path: Path,
    winning_only: bool = False,
    outcome_bonus: float = 0.25,
) -> int:
    """Build an atomic, validated DPO JSONL dataset from self-play games."""
    outcome_bonus = _validated_outcome_bonus(outcome_bonus)
    records = _load_self_play_jsonl(input_path)
    pairs = [
        pair
        for line_number, record in records
        for pair in _preference_pairs_from_game(
            record,
            winning_only=winning_only,
            outcome_bonus=outcome_bonus,
            context=f"{input_path}: line {line_number}",
        )
    ]
    if not pairs:
        reason = " after filtering to winning traces" if winning_only else ""
        raise ValueError(f"no DPO preference pairs were produced{reason}: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for pair in pairs:
                output.write(json.dumps(pair, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return len(pairs)


def preference_pairs_from_game(
    record: dict[str, Any], *, winning_only: bool, outcome_bonus: float
) -> list[dict[str, str]]:
    """Convert one game record into deterministic preference pairs."""
    return _preference_pairs_from_game(
        record,
        winning_only=winning_only,
        outcome_bonus=_validated_outcome_bonus(outcome_bonus),
        context="game record",
    )


def _preference_pairs_from_game(
    record: dict[str, Any],
    *,
    winning_only: bool,
    outcome_bonus: float,
    context: str,
) -> list[dict[str, str]]:
    if not isinstance(record, dict):
        raise ValueError(f"{context}: game record must be an object")
    winner = _parse_enum(Winner, record.get("winner"), f"{context}.winner")
    traces = record.get("decision_traces")
    if not isinstance(traces, list):
        raise ValueError(f"{context}.decision_traces must be a list")

    pairs: list[dict[str, str]] = []
    for trace_index, trace in enumerate(traces, start=1):
        trace_context = f"{context}.decision_traces[{trace_index}]"
        if not isinstance(trace, dict):
            raise ValueError(f"{trace_context} must be an object")

        role = _parse_enum(Role, trace.get("role"), f"{trace_context}.role")
        prompt = trace.get("observation_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{trace_context}.observation_prompt must be a non-empty string")

        candidates = trace.get("candidates")
        evaluations = trace.get("evaluations")
        if not isinstance(candidates, list):
            raise ValueError(f"{trace_context}.candidates must be a list")
        if not isinstance(evaluations, list):
            raise ValueError(f"{trace_context}.evaluations must be a list")
        if len(candidates) < 2:
            raise ValueError(f"{trace_context} must contain at least two candidates")
        if len(candidates) != len(evaluations):
            raise ValueError(
                f"{trace_context} has {len(candidates)} candidates but "
                f"{len(evaluations)} evaluations"
            )

        selected_index = trace.get("selected_index")
        if isinstance(selected_index, bool) or not isinstance(selected_index, int):
            raise ValueError(f"{trace_context}.selected_index must be an integer")
        if not 0 <= selected_index < len(candidates):
            raise ValueError(
                f"{trace_context}.selected_index={selected_index} is outside "
                f"[0, {len(candidates) - 1}]"
            )

        responses = [
            _candidate_response(candidate, f"{trace_context}.candidates[{index}]")
            for index, candidate in enumerate(candidates)
        ]
        scored = [
            _candidate_score(evaluation, f"{trace_context}.evaluations[{index}]")
            for index, evaluation in enumerate(evaluations)
        ]

        won = _role_won(role, winner)
        if winning_only and not won:
            continue
        if winner is not Winner.DRAW:
            selected_score, selected_legal = scored[selected_index]
            selected_score += outcome_bonus if won else -outcome_bonus
            scored[selected_index] = (selected_score, selected_legal)

        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda index: (
                not scored[index][1],
                -scored[index][0],
                index,
            ),
        )
        chosen_index = ranked_indices[0]
        chosen = responses[chosen_index]
        rejected_index = next(
            (
                index
                for index in reversed(ranked_indices)
                if responses[index].strip() != chosen.strip()
            ),
            None,
        )
        if rejected_index is None:
            continue

        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": responses[rejected_index],
            }
        )
    return pairs


def _load_self_play_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        raise ValueError(f"self-play JSONL does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"self-play JSONL is not a file: {path}")

    records: list[tuple[int, dict[str, Any]]] = []
    try:
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSONL at {path}: line {line_number}, column {error.colno}: "
                        f"{error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}: line {line_number} must be a JSON object")
                records.append((line_number, record))
    except UnicodeDecodeError as error:
        raise ValueError(f"self-play JSONL must be UTF-8 encoded: {path}") from error

    if not records:
        raise ValueError(f"self-play JSONL contains no records: {path}")
    return records


def _candidate_response(candidate: Any, context: str) -> str:
    if not isinstance(candidate, dict):
        raise ValueError(f"{context} must be an object")
    action_type = _parse_enum(ActionType, candidate.get("action_type"), f"{context}.action_type")
    if action_type is ActionType.SPEAK:
        message = candidate.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"{context}.message must be a non-empty string for speak actions")
        return message

    target = candidate.get("target_id")
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise ValueError(f"{context}.target_id must be null or a non-empty string")
    return json.dumps(
        {"action": action_type.value, "target": target or "abstain"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _candidate_score(evaluation: Any, context: str) -> tuple[float, bool]:
    if not isinstance(evaluation, dict):
        raise ValueError(f"{context} must be an object")
    legal = evaluation.get("legal")
    if not isinstance(legal, bool):
        raise ValueError(f"{context}.legal must be a boolean")
    raw_score = evaluation.get("score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise ValueError(f"{context}.score must be a finite number")
    score = float(raw_score)
    if not math.isfinite(score):
        raise ValueError(f"{context}.score must be a finite number")
    return score, legal


def _validated_outcome_bonus(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("outcome_bonus must be a finite non-negative number")
    outcome_bonus = float(value)
    if not math.isfinite(outcome_bonus) or outcome_bonus < 0:
        raise ValueError("outcome_bonus must be a finite non-negative number")
    return outcome_bonus


def _parse_enum(enum_type: type[Role] | type[Winner] | type[ActionType], value: Any, context: str):
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(
            f"{context} has invalid value {value!r}; expected one of: {allowed}"
        ) from error


def _role_won(role: Role, winner: Winner) -> bool:
    if winner is Winner.DRAW:
        return False
    if role is Role.WEREWOLF:
        return winner is Winner.WEREWOLF
    return winner is Winner.VILLAGE
