from types import SimpleNamespace

import pytest

from wolfplay.abstract_game import AbstractWerewolfGame
from wolfplay.cfr_preference import cfr_preference_pairs_from_game
from wolfplay.engine import GameRuntime
from wolfplay.latent import HashingTextEmbedder, fit_latent_strategy_space


class _FakePolicy:
    def __init__(self, game):
        self.game = game
        self.trainer = SimpleNamespace(iteration=4)

    def predicted_advantages(self, state):
        return {action_id: float(action_id) for action_id in state.legal_actions()}


@pytest.mark.asyncio
async def test_cfr_preferences_replay_a_real_game_trace():
    result = await GameRuntime(seed=17, max_rounds=2).play()
    record = result.to_dict()
    embedder = HashingTextEmbedder(64)
    latent_space = fit_latent_strategy_space([record], embedder=embedder)
    policy = _FakePolicy(AbstractWerewolfGame(latent_space))

    pairs = cfr_preference_pairs_from_game(
        record,
        policy=policy,
        latent_space=latent_space,
        embedder=embedder,
    )

    assert pairs
    assert all(pair["chosen"] != pair["rejected"] for pair in pairs)
    assert all(pair["metadata"]["source"] == "deep_cfr_advantage" for pair in pairs)
