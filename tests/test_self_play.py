import json

import pytest

from wolfplay.self_play import load_jsonl, run_self_play


async def test_concurrent_self_play_preserves_seed_order(tmp_path):
    output_path = tmp_path / "trajectories.jsonl"
    results, summary = await run_self_play(
        games=3,
        seed=100,
        max_rounds=3,
        concurrency=2,
        output_path=output_path,
    )

    assert [result.seed for result in results] == [100, 101, 102]
    assert summary.games == 3
    records = load_jsonl(output_path)
    assert [record["seed"] for record in records] == [100, 101, 102]
    assert not (tmp_path / ".trajectories.jsonl.tmp").exists()


def test_load_jsonl_rejects_non_object_lines(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(["not", "an", "object"]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be an object"):
        load_jsonl(path)
