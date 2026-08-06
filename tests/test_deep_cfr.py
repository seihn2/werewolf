import random

import pytest

from wolfplay.abstract_game import AbstractGameConfig, AbstractWerewolfGame
from wolfplay.latent import HashingTextEmbedder, fit_latent_strategy_space
from wolfplay.models import Role
from wolfplay.training.deep_cfr import (
    DeepCFRConfig,
    DeepCFRPolicy,
    DeepCFRTrainer,
    ReservoirBuffer,
)


def _game():
    record = {
        "decision_traces": [
            {
                "role": role.value,
                "candidates": [
                    {"action_type": "speak", "message": f"{role.value} strategy"},
                ],
            }
            for role in Role
        ]
    }
    space = fit_latent_strategy_space(
        [record],
        embedder=HashingTextEmbedder(16),
        clusters_by_role={role: 1 for role in Role},
    )
    return AbstractWerewolfGame(
        space,
        config=AbstractGameConfig(max_rounds=1, role_assignment_limit=1),
    )


def test_reservoir_buffer_caps_size_and_restores_state():
    buffer = ReservoirBuffer[int](3, seed=5)
    for value in range(20):
        buffer.add(value)

    restored = ReservoirBuffer.from_state_dict(buffer.state_dict())

    assert len(buffer) == 3
    assert buffer.seen == 20
    assert restored.items == buffer.items
    assert restored.sample(2) == buffer.sample(2)


def test_external_sampling_traversal_collects_advantage_and_strategy_samples():
    game = _game()
    trainer = object.__new__(DeepCFRTrainer)
    trainer.game = game
    trainer.config = DeepCFRConfig(
        iterations=1,
        traversals_per_player=1,
        advantage_train_steps=0,
        strategy_train_steps=0,
        max_traversal_depth=3,
        max_rollout_steps=128,
    )
    trainer.rng = random.Random(7)
    trainer.advantage_networks = {role: None for role in Role}
    trainer.advantage_buffers = {
        role: ReservoirBuffer(100, seed=10 + index) for index, role in enumerate(Role)
    }
    trainer.strategy_buffers = {
        role: ReservoirBuffer(100, seed=20 + index) for index, role in enumerate(Role)
    }
    trainer._predict = lambda network, information_state: [0.0] * game.num_actions

    value = trainer._traverse(
        game.new_initial_state(),
        traverser=0,
        depth=0,
        iteration=1,
    )

    assert isinstance(value, float)
    assert sum(len(buffer) for buffer in trainer.advantage_buffers.values()) > 0
    assert sum(len(buffer) for buffer in trainer.strategy_buffers.values()) > 0


def test_deep_cfr_config_rejects_invalid_sizes():
    with pytest.raises(ValueError, match="iterations"):
        DeepCFRConfig(iterations=0)
    with pytest.raises(ValueError, match="hidden_sizes"):
        DeepCFRConfig(hidden_sizes=())


def test_tiny_torch_training_saves_loadable_policy(tmp_path):
    pytest.importorskip("torch")
    game = _game()
    trainer = DeepCFRTrainer(
        game,
        DeepCFRConfig(
            iterations=1,
            traversals_per_player=1,
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=2,
            hidden_sizes=(8,),
            max_traversal_depth=1,
            max_rollout_steps=128,
            advantage_buffer_capacity=20,
            strategy_buffer_capacity=20,
            save_buffers=False,
        ),
    )

    trainer.train(tmp_path)
    policy = DeepCFRPolicy.load(tmp_path)
    state = policy.game.new_initial_state()
    state.apply_action(state.chance_outcomes()[0][0])
    probabilities = policy.action_probabilities(state)

    assert (tmp_path / "deep_cfr.pt").is_file()
    assert sum(probabilities.values()) == pytest.approx(1.0)
