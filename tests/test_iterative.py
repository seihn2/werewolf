import json

import pytest

from wolfplay.iterative import IterativePolicyConfig, IterativePolicyOptimizer
from wolfplay.latent import HashingTextEmbedder, LatentStrategySpace
from wolfplay.models import Role
from wolfplay.training.deep_cfr import DeepCFRConfig
from wolfplay.training.dpo import DPOTrainingConfig


@pytest.mark.asyncio
async def test_iterative_optimizer_hands_previous_dpo_checkpoint_to_next_round(
    tmp_path, monkeypatch
):
    backend_calls = []
    trained_dpo = []

    class Backend:
        async def aclose(self):
            return None

    def backend_factory(iteration, previous_checkpoint):
        backend_calls.append((iteration, previous_checkpoint))
        return Backend()

    async def fake_self_play(**kwargs):
        kwargs["output_path"].write_text("{}\n", encoding="utf-8")
        return [], None

    def fake_build_latent(*, output_path, embedder, **kwargs):
        record = {
            "decision_traces": [
                {
                    "role": role.value,
                    "candidates": [{"action_type": "speak", "message": f"{role.value} plan"}],
                }
                for role in Role
            ]
        }
        from wolfplay.latent import fit_latent_strategy_space

        space = fit_latent_strategy_space([record], embedder=embedder)
        space.save(output_path)
        return space

    def fake_train_cfr(*, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "deep_cfr.pt").write_bytes(b"checkpoint")
        return {}

    def fake_build_cfr_dpo(*, output_path, **kwargs):
        output_path.write_text(
            json.dumps({"prompt": "p", "chosen": "c", "rejected": "r"}) + "\n",
            encoding="utf-8",
        )
        return 1

    def dpo_config_factory(iteration, dataset, output_dir, previous_checkpoint):
        return DPOTrainingConfig(
            dataset=dataset,
            model=str(previous_checkpoint or "base-model"),
            output_dir=output_dir,
        )

    def fake_train_dpo(config):
        trained_dpo.append(config.model)
        (config.output_dir / "final").mkdir(parents=True)

    monkeypatch.setattr("wolfplay.iterative.run_self_play", fake_self_play)
    monkeypatch.setattr("wolfplay.iterative.build_latent_strategy_file", fake_build_latent)
    monkeypatch.setattr("wolfplay.iterative.train_deep_cfr", fake_train_cfr)
    monkeypatch.setattr("wolfplay.iterative.build_cfr_dpo_dataset", fake_build_cfr_dpo)
    monkeypatch.setattr("wolfplay.iterative.train_dpo", fake_train_dpo)

    optimizer = IterativePolicyOptimizer(
        config=IterativePolicyConfig(
            iterations=2,
            games_per_iteration=1,
            deep_cfr=DeepCFRConfig(
                iterations=1,
                traversals_per_player=1,
                advantage_train_steps=0,
                strategy_train_steps=0,
            ),
        ),
        embedder=HashingTextEmbedder(16),
        backend_factory=backend_factory,
        dpo_config_factory=dpo_config_factory,
    )

    artifacts = await optimizer.run(tmp_path / "iterations")

    first_checkpoint = artifacts[0].dpo_checkpoint
    assert isinstance(LatentStrategySpace.load(artifacts[0].latent_space), LatentStrategySpace)
    assert backend_calls == [(1, None), (2, first_checkpoint)]
    assert trained_dpo == ["base-model", str(first_checkpoint)]
    assert artifacts[1].dpo_checkpoint is not None
