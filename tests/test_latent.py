import json

import pytest

from wolfplay.latent import (
    HashingTextEmbedder,
    LatentStrategySpace,
    fit_kmeans,
    fit_latent_strategy_space,
)
from wolfplay.models import Role


def _record():
    return {
        "decision_traces": [
            {
                "role": role.value,
                "candidates": [
                    {
                        "action_type": "speak",
                        "message": f"{role.value} accuse the quiet player",
                    },
                    {
                        "action_type": "speak",
                        "message": f"{role.value} defend the claimed seer",
                    },
                    {"action_type": "vote", "target_id": "player_0"},
                ],
            }
            for role in Role
        ]
    }


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingTextEmbedder(32)

    first, second = embedder.embed_many(["悍跳预言家", "悍跳预言家"])

    assert first == second
    assert len(first) == 32
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_kmeans_is_seeded_and_separates_obvious_groups():
    vectors = [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]]

    first = fit_kmeans(vectors, clusters=2, seed=7)
    second = fit_kmeans(vectors, clusters=2, seed=7)

    assert first == second
    assert first.labels[0] == first.labels[1]
    assert first.labels[2] == first.labels[3]
    assert first.labels[0] != first.labels[2]


def test_latent_strategy_space_round_trips_and_assigns(tmp_path):
    embedder = HashingTextEmbedder(48)
    space = fit_latent_strategy_space(
        [_record()],
        embedder=embedder,
        clusters_by_role={role: 2 for role in Role},
        seed=9,
    )
    path = tmp_path / "latent.json"

    space.save(path)
    loaded = LatentStrategySpace.load(path)

    assert loaded.to_dict() == json.loads(path.read_text(encoding="utf-8"))
    cluster_id = loaded.assign(Role.WEREWOLF, "werewolf defend the claimed seer", embedder)
    assert cluster_id in {0, 1}
    assert loaded.representative(Role.WEREWOLF, cluster_id)


def test_latent_space_reduces_cluster_count_for_duplicate_vectors():
    record = {
        "decision_traces": [
            {
                "role": role.value,
                "candidates": [
                    {"action_type": "speak", "message": "same strategy"},
                    {"action_type": "speak", "message": "same strategy"},
                ],
            }
            for role in Role
        ]
    }

    space = fit_latent_strategy_space(
        [record],
        embedder=HashingTextEmbedder(16),
        clusters_by_role={role: 3 for role in Role},
    )

    assert all(len(role_space.clusters) == 1 for role_space in space.roles.values())
