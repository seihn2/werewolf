import random

from wolfplay.abstract_game import AbstractGameConfig, AbstractWerewolfGame
from wolfplay.latent import HashingTextEmbedder, fit_latent_strategy_space
from wolfplay.models import Role


def _game(*, max_rounds=2, role_assignment_limit=3):
    record = {
        "decision_traces": [
            {
                "role": role.value,
                "candidates": [
                    {"action_type": "speak", "message": f"{role.value} plan alpha"},
                    {"action_type": "speak", "message": f"{role.value} plan beta"},
                ],
            }
            for role in Role
        ]
    }
    space = fit_latent_strategy_space(
        [record],
        embedder=HashingTextEmbedder(32),
        clusters_by_role={role: 2 for role in Role},
    )
    return AbstractWerewolfGame(
        space,
        config=AbstractGameConfig(
            max_rounds=max_rounds,
            role_assignment_limit=role_assignment_limit,
        ),
    )


def test_abstract_game_exposes_fixed_information_state_and_terminates():
    game = _game()
    state = game.new_initial_state()
    rng = random.Random(3)

    assert state.is_chance_node
    assert len(state.chance_outcomes()) == 3
    state.apply_action(state.chance_outcomes()[0][0])

    steps = 0
    while not state.is_terminal:
        steps += 1
        assert steps < 300
        if state.is_chance_node:
            state.apply_action(state.chance_outcomes()[0][0])
            continue
        assert len(state.information_state_tensor()) == game.information_state_size
        legal_actions = state.legal_actions()
        assert legal_actions
        state.apply_action(rng.choice(legal_actions))

    assert len(state.returns()) == game.num_players
    assert state.terminal_winner is not None


def test_action_catalog_maps_role_clusters_and_targets():
    game = _game()
    catalog = game.action_catalog

    assert catalog.target_action_id(4) == 4
    assert catalog.action(4).target_index == 4
    for role in Role:
        action_ids = catalog.speech_action_ids(role)
        assert len(action_ids) == 2
        assert all(catalog.action(action_id).role is role for action_id in action_ids)
