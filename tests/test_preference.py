import json

import pytest

from wolfplay.preference import build_dpo_dataset, preference_pairs_from_game


def _candidate(message: str) -> dict:
    return {"action_type": "speak", "message": message, "target_id": None}


def _trace(
    *,
    role: str = "seer",
    selected_index: int = 0,
    messages: tuple[str, ...] = ("chosen", "rejected"),
    scores: tuple[float, ...] = (0.8, 0.1),
    legal: tuple[bool, ...] | None = None,
) -> dict:
    legal = legal or tuple(True for _ in scores)
    return {
        "role": role,
        "observation_prompt": "observation",
        "selected_index": selected_index,
        "candidates": [_candidate(message) for message in messages],
        "evaluations": [
            {"score": score, "legal": is_legal}
            for score, is_legal in zip(scores, legal, strict=True)
        ],
    }


def _record(*, winner: str = "village", trace: dict | None = None) -> dict:
    return {"winner": winner, "decision_traces": [trace or _trace()]}


def test_build_dpo_dataset(tmp_path):
    input_path = tmp_path / "self_play.jsonl"
    output_path = tmp_path / "dpo.jsonl"
    input_path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    count = build_dpo_dataset(input_path=input_path, output_path=output_path)

    assert count == 1
    pair = json.loads(output_path.read_text(encoding="utf-8"))
    assert pair == {"prompt": "observation", "chosen": "chosen", "rejected": "rejected"}


def test_equal_scores_use_stable_candidate_order():
    trace = _trace(
        selected_index=1,
        messages=("first", "second", "third"),
        scores=(0.5, 0.5, 0.5),
    )

    pairs = preference_pairs_from_game(
        _record(winner="draw", trace=trace),
        winning_only=False,
        outcome_bonus=0.25,
    )

    assert pairs[0]["chosen"] == "first"
    assert pairs[0]["rejected"] == "third"


def test_game_outcome_reweights_only_the_selected_action():
    trace = _trace(selected_index=1, messages=("other", "selected"), scores=(0.5, 0.5))

    winning_pair = preference_pairs_from_game(
        _record(winner="village", trace=trace),
        winning_only=False,
        outcome_bonus=0.25,
    )[0]
    losing_pair = preference_pairs_from_game(
        _record(winner="werewolf", trace=trace),
        winning_only=False,
        outcome_bonus=0.25,
    )[0]

    assert winning_pair["chosen"] == "selected"
    assert losing_pair["rejected"] == "selected"


def test_legal_candidate_outranks_illegal_candidate_regardless_of_score():
    trace = _trace(
        messages=("legal", "illegal"),
        scores=(0.1, 100.0),
        legal=(True, False),
    )

    pair = preference_pairs_from_game(
        _record(winner="draw", trace=trace),
        winning_only=False,
        outcome_bonus=0.25,
    )[0]

    assert pair["chosen"] == "legal"
    assert pair["rejected"] == "illegal"


def test_invalid_json_reports_location_without_overwriting_output(tmp_path):
    input_path = tmp_path / "self_play.jsonl"
    output_path = tmp_path / "dpo.jsonl"
    input_path.write_text(json.dumps(_record()) + "\n{broken\n", encoding="utf-8")
    output_path.write_text("keep-me\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 2, column 2"):
        build_dpo_dataset(input_path=input_path, output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == "keep-me\n"


def test_bad_trace_reports_context(tmp_path):
    input_path = tmp_path / "self_play.jsonl"
    output_path = tmp_path / "dpo.jsonl"
    input_path.write_text(
        json.dumps(_record(trace=_trace(selected_index=5))) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"decision_traces\[1\]\.selected_index=5"):
        build_dpo_dataset(input_path=input_path, output_path=output_path)

    assert not output_path.exists()


def test_empty_or_fully_filtered_input_fails_clearly(tmp_path):
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no records"):
        build_dpo_dataset(input_path=empty_path, output_path=tmp_path / "empty-output.jsonl")

    losing_path = tmp_path / "losing.jsonl"
    losing_path.write_text(json.dumps(_record(winner="werewolf")) + "\n", encoding="utf-8")
    filtered_output = tmp_path / "filtered.jsonl"
    filtered_output.write_text("keep-me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="after filtering to winning traces"):
        build_dpo_dataset(
            input_path=losing_path,
            output_path=filtered_output,
            winning_only=True,
        )

    assert filtered_output.read_text(encoding="utf-8") == "keep-me\n"


@pytest.mark.parametrize("outcome_bonus", [-0.1, float("inf"), float("nan")])
def test_outcome_bonus_must_be_finite_and_non_negative(outcome_bonus):
    with pytest.raises(ValueError, match="outcome_bonus"):
        preference_pairs_from_game(
            _record(),
            winning_only=False,
            outcome_bonus=outcome_bonus,
        )
