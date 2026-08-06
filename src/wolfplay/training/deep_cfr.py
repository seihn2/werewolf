from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from ..abstract_game import (
    AbstractGameConfig,
    AbstractWerewolfGame,
    AbstractWerewolfState,
    ActionCatalog,
    RewardConfig,
)
from ..latent import LatentStrategySpace
from ..models import Role

SampleT = TypeVar("SampleT")


@dataclass(frozen=True, slots=True)
class AdvantageSample:
    information_state: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    advantages: tuple[float, ...]
    iteration: int


@dataclass(frozen=True, slots=True)
class StrategySample:
    information_state: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    strategy: tuple[float, ...]
    iteration: int


class ReservoirBuffer(Generic[SampleT]):
    """Uniform reservoir sampler with deterministic, serializable state."""

    def __init__(self, capacity: int, *, seed: int = 42) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.items: list[SampleT] = []
        self.seen = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: SampleT) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        replacement = self._rng.randrange(self.seen)
        if replacement < self.capacity:
            self.items[replacement] = item

    def sample(self, batch_size: int) -> list[SampleT]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.items:
            return []
        return self._rng.sample(self.items, min(batch_size, len(self.items)))

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "items": self.items,
            "seen": self.seen,
            "rng_state": self._rng.getstate(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> ReservoirBuffer[Any]:
        buffer = cls(int(state["capacity"]))
        buffer.items = list(state["items"])
        buffer.seen = int(state["seen"])
        buffer._rng.setstate(state["rng_state"])
        return buffer


@dataclass(frozen=True, slots=True)
class DeepCFRConfig:
    iterations: int = 10
    traversals_per_player: int = 4
    advantage_train_steps: int = 100
    strategy_train_steps: int = 200
    batch_size: int = 128
    learning_rate: float = 1e-3
    advantage_buffer_capacity: int = 100_000
    strategy_buffer_capacity: int = 100_000
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    max_traversal_depth: int = 64
    max_rollout_steps: int = 512
    gradient_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    checkpoint_every: int = 1
    save_buffers: bool = True
    reinitialize_advantage_each_iteration: bool = True
    reinitialize_strategy_each_iteration: bool = True

    def __post_init__(self) -> None:
        _positive_int("iterations", self.iterations)
        _positive_int("traversals_per_player", self.traversals_per_player)
        _non_negative_int("advantage_train_steps", self.advantage_train_steps)
        _non_negative_int("strategy_train_steps", self.strategy_train_steps)
        _positive_int("batch_size", self.batch_size)
        _positive_float("learning_rate", self.learning_rate)
        _positive_int("advantage_buffer_capacity", self.advantage_buffer_capacity)
        _positive_int("strategy_buffer_capacity", self.strategy_buffer_capacity)
        if not self.hidden_sizes or any(width <= 0 for width in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        _positive_int("max_traversal_depth", self.max_traversal_depth)
        _positive_int("max_rollout_steps", self.max_rollout_steps)
        _positive_float("gradient_clip", self.gradient_clip)
        _positive_int("checkpoint_every", self.checkpoint_every)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["hidden_sizes"] = list(self.hidden_sizes)
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> DeepCFRConfig:
        values = dict(record)
        values["hidden_sizes"] = tuple(values["hidden_sizes"])
        return cls(**values)


class DeepCFRTrainer:
    """External-sampling Deep CFR for the abstract seven-player game."""

    def __init__(self, game: AbstractWerewolfGame, config: DeepCFRConfig | None = None) -> None:
        self.game = game
        self.config = config or DeepCFRConfig()
        self._torch, regret_network, strategy_network = _training_dependencies()
        self._regret_network_type = regret_network
        self._strategy_network_type = strategy_network
        self.device = _resolve_device(self._torch, self.config.device)
        self._set_global_seed(self.config.seed)
        self.rng = random.Random(self.config.seed)
        self.iteration = 0
        self.metrics: list[dict[str, Any]] = []
        self.advantage_networks = {role: self._new_advantage_network(role) for role in Role}
        self.strategy_networks = {role: self._new_strategy_network(role) for role in Role}
        self.advantage_buffers = {
            role: ReservoirBuffer[AdvantageSample](
                self.config.advantage_buffer_capacity,
                seed=self.config.seed + 10_000 + index,
            )
            for index, role in enumerate(Role)
        }
        self.strategy_buffers = {
            role: ReservoirBuffer[StrategySample](
                self.config.strategy_buffer_capacity,
                seed=self.config.seed + 20_000 + index,
            )
            for index, role in enumerate(Role)
        }

    def train(self, output_dir: Path | None = None) -> dict[str, Any]:
        self._set_global_seed(self.config.seed)
        for iteration in range(self.iteration + 1, self.config.iterations + 1):
            samples_before = sum(len(buffer) for buffer in self.advantage_buffers.values())
            for traverser in range(self.game.num_players):
                for _ in range(self.config.traversals_per_player):
                    state = self.game.new_initial_state()
                    self._traverse(state, traverser=traverser, depth=0, iteration=iteration)

            advantage_losses: dict[str, float | None] = {}
            strategy_losses: dict[str, float | None] = {}
            for role in Role:
                advantage_losses[role.value] = self._train_advantage(role, iteration)
                strategy_losses[role.value] = self._train_strategy(role, iteration)
            self.iteration = iteration
            metric = {
                "iteration": iteration,
                "advantage_samples_added": (
                    sum(len(buffer) for buffer in self.advantage_buffers.values()) - samples_before
                ),
                "advantage_buffer_sizes": {
                    role.value: len(self.advantage_buffers[role]) for role in Role
                },
                "strategy_buffer_sizes": {
                    role.value: len(self.strategy_buffers[role]) for role in Role
                },
                "advantage_losses": advantage_losses,
                "strategy_losses": strategy_losses,
            }
            self.metrics.append(metric)
            if output_dir is not None and iteration % self.config.checkpoint_every == 0:
                self.save_checkpoint(
                    output_dir / "checkpoints" / f"iteration_{iteration:04d}.pt",
                    include_buffers=self.config.save_buffers,
                )

        summary = self.summary()
        if output_dir is not None:
            self.save_checkpoint(
                output_dir / "deep_cfr.pt",
                include_buffers=self.config.save_buffers,
            )
            _atomic_write_json(output_dir / "deep_cfr_manifest.json", summary)
        return summary

    def summary(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "config": self.config.to_dict(),
            "game": {
                "players": self.game.num_players,
                "actions": self.game.num_actions,
                "information_state_size": self.game.information_state_size,
                "role_assignments": len(self.game.role_assignments),
            },
            "advantage_buffer_sizes": {
                role.value: len(self.advantage_buffers[role]) for role in Role
            },
            "strategy_buffer_sizes": {
                role.value: len(self.strategy_buffers[role]) for role in Role
            },
            "metrics": self.metrics,
        }

    def save_checkpoint(self, path: Path, *, include_buffers: bool = True) -> None:
        payload = {
            "version": 1,
            "iteration": self.iteration,
            "config": self.config.to_dict(),
            "latent_space": self.game.latent_space.to_dict(),
            "action_catalog": self.game.action_catalog.to_dict(),
            "game_config": _game_config_to_dict(self.game.config),
            "advantage_networks": {
                role.value: _cpu_state_dict(network)
                for role, network in self.advantage_networks.items()
            },
            "strategy_networks": {
                role.value: _cpu_state_dict(network)
                for role, network in self.strategy_networks.items()
            },
            "rng_state": self.rng.getstate(),
            "torch_rng_state": self._torch.random.get_rng_state(),
            "metrics": self.metrics,
        }
        if include_buffers:
            payload["advantage_buffers"] = {
                role.value: buffer.state_dict() for role, buffer in self.advantage_buffers.items()
            }
            payload["strategy_buffers"] = {
                role.value: buffer.state_dict() for role, buffer in self.strategy_buffers.items()
            }
        _atomic_torch_save(self._torch, path, payload)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        device: str | None = None,
    ) -> DeepCFRTrainer:
        torch, _, _ = _training_dependencies()
        checkpoint_path = _checkpoint_path(path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = DeepCFRConfig.from_dict(payload["config"])
        if device is not None:
            config = DeepCFRConfig.from_dict({**config.to_dict(), "device": device})
        latent_space = LatentStrategySpace.from_dict(payload["latent_space"])
        action_catalog = ActionCatalog.from_dict(payload["action_catalog"])
        game = AbstractWerewolfGame(
            latent_space,
            config=_game_config_from_dict(payload["game_config"]),
            action_catalog=action_catalog,
        )
        trainer = cls(game, config)
        trainer.iteration = int(payload["iteration"])
        trainer.metrics = list(payload.get("metrics", []))
        trainer.rng.setstate(payload["rng_state"])
        torch.random.set_rng_state(payload["torch_rng_state"])
        for role in Role:
            trainer.advantage_networks[role].load_state_dict(
                payload["advantage_networks"][role.value]
            )
            trainer.strategy_networks[role].load_state_dict(
                payload["strategy_networks"][role.value]
            )
        if "advantage_buffers" in payload:
            trainer.advantage_buffers = {
                role: ReservoirBuffer.from_state_dict(payload["advantage_buffers"][role.value])
                for role in Role
            }
        if "strategy_buffers" in payload:
            trainer.strategy_buffers = {
                role: ReservoirBuffer.from_state_dict(payload["strategy_buffers"][role.value])
                for role in Role
            }
        return trainer

    def _traverse(
        self,
        state: AbstractWerewolfState,
        *,
        traverser: int,
        depth: int,
        iteration: int,
    ) -> float:
        if state.is_terminal:
            return state.returns()[traverser]
        if depth >= self.config.max_traversal_depth:
            return self._rollout(state, traverser=traverser)
        if state.is_chance_node:
            outcome = _sample_distribution(self.rng, state.chance_outcomes())
            child = state.clone()
            child.apply_action(outcome)
            return self._traverse(
                child,
                traverser=traverser,
                depth=depth + 1,
                iteration=iteration,
            )

        role = state.current_role
        legal_actions = state.legal_actions()
        information_state = state.information_state_tensor()
        strategy = self._regret_matching_strategy(role, information_state, legal_actions)
        if state.current_player == traverser:
            action_values: dict[int, float] = {}
            for action_id in legal_actions:
                child = state.clone()
                child.apply_action(action_id)
                action_values[action_id] = self._traverse(
                    child,
                    traverser=traverser,
                    depth=depth + 1,
                    iteration=iteration,
                )
            node_value = sum(
                strategy[action_id] * action_values[action_id] for action_id in legal_actions
            )
            advantages = [0.0] * self.game.num_actions
            for action_id in legal_actions:
                advantages[action_id] = action_values[action_id] - node_value
            self.advantage_buffers[role].add(
                AdvantageSample(
                    information_state=information_state,
                    legal_mask=_legal_mask(legal_actions, self.game.num_actions),
                    advantages=tuple(advantages),
                    iteration=iteration,
                )
            )
            return node_value

        self.strategy_buffers[role].add(
            StrategySample(
                information_state=information_state,
                legal_mask=_legal_mask(legal_actions, self.game.num_actions),
                strategy=tuple(strategy),
                iteration=iteration,
            )
        )
        action_id = _sample_action(self.rng, strategy, legal_actions)
        child = state.clone()
        child.apply_action(action_id)
        return self._traverse(
            child,
            traverser=traverser,
            depth=depth + 1,
            iteration=iteration,
        )

    def _rollout(self, state: AbstractWerewolfState, *, traverser: int) -> float:
        rollout_state = state.clone()
        for _ in range(self.config.max_rollout_steps):
            if rollout_state.is_terminal:
                return rollout_state.returns()[traverser]
            if rollout_state.is_chance_node:
                outcome = _sample_distribution(self.rng, rollout_state.chance_outcomes())
                rollout_state.apply_action(outcome)
                continue
            legal_actions = rollout_state.legal_actions()
            strategy = self._regret_matching_strategy(
                rollout_state.current_role,
                rollout_state.information_state_tensor(),
                legal_actions,
            )
            rollout_state.apply_action(_sample_action(self.rng, strategy, legal_actions))
        return rollout_state.rewards[traverser]

    def _regret_matching_strategy(
        self,
        role: Role,
        information_state: tuple[float, ...],
        legal_actions: tuple[int, ...],
    ) -> list[float]:
        predicted = self._predict(self.advantage_networks[role], information_state)
        positive = {action_id: max(0.0, predicted[action_id]) for action_id in legal_actions}
        total = sum(positive.values())
        strategy = [0.0] * self.game.num_actions
        if total <= 0:
            probability = 1.0 / len(legal_actions)
            for action_id in legal_actions:
                strategy[action_id] = probability
            return strategy
        for action_id in legal_actions:
            strategy[action_id] = positive[action_id] / total
        return strategy

    def _train_advantage(self, role: Role, iteration: int) -> float | None:
        buffer = self.advantage_buffers[role]
        if not buffer or self.config.advantage_train_steps == 0:
            return None
        if self.config.reinitialize_advantage_each_iteration:
            self.advantage_networks[role] = self._new_advantage_network(role)
        network = self.advantage_networks[role]
        optimizer = self._torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
        network.train()
        losses: list[float] = []
        for _ in range(self.config.advantage_train_steps):
            batch = buffer.sample(self.config.batch_size)
            inputs, masks, targets, weights = self._advantage_batch(batch, iteration)
            predictions = network(inputs)
            per_action = (predictions - targets).pow(2) * masks
            per_sample = per_action.sum(dim=1) / masks.sum(dim=1).clamp_min(1.0)
            loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._torch.nn.utils.clip_grad_norm_(network.parameters(), self.config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return sum(losses) / len(losses)

    def _train_strategy(self, role: Role, iteration: int) -> float | None:
        buffer = self.strategy_buffers[role]
        if not buffer or self.config.strategy_train_steps == 0:
            return None
        if self.config.reinitialize_strategy_each_iteration:
            self.strategy_networks[role] = self._new_strategy_network(role)
        network = self.strategy_networks[role]
        optimizer = self._torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
        network.train()
        losses: list[float] = []
        for _ in range(self.config.strategy_train_steps):
            batch = buffer.sample(self.config.batch_size)
            inputs, masks, targets, weights = self._strategy_batch(batch, iteration)
            logits = network(inputs).masked_fill(~masks, -1e9)
            probabilities = self._torch.softmax(logits, dim=1)
            per_action = (probabilities - targets).pow(2) * masks
            per_sample = per_action.sum(dim=1) / masks.sum(dim=1).clamp_min(1.0)
            loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._torch.nn.utils.clip_grad_norm_(network.parameters(), self.config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return sum(losses) / len(losses)

    def _advantage_batch(self, batch: list[AdvantageSample], iteration: int):
        torch = self._torch
        inputs = torch.tensor(
            [sample.information_state for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.tensor(
            [sample.legal_mask for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )
        targets = torch.tensor(
            [sample.advantages for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )
        weights = torch.tensor(
            [sample.iteration / iteration for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )
        return inputs, masks, targets, weights

    def _strategy_batch(self, batch: list[StrategySample], iteration: int):
        torch = self._torch
        inputs = torch.tensor(
            [sample.information_state for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.tensor(
            [sample.legal_mask for sample in batch],
            dtype=torch.bool,
            device=self.device,
        )
        targets = torch.tensor(
            [sample.strategy for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )
        weights = torch.tensor(
            [sample.iteration / iteration for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )
        return inputs, masks, targets, weights

    def _predict(self, network, information_state: tuple[float, ...]) -> list[float]:
        torch = self._torch
        network.eval()
        with torch.no_grad():
            inputs = torch.tensor(
                [information_state],
                dtype=torch.float32,
                device=self.device,
            )
            return network(inputs)[0].detach().cpu().tolist()

    def _new_advantage_network(self, role: Role):
        del role
        return self._regret_network_type(
            self.game.information_state_size,
            self.game.num_actions,
            self.config.hidden_sizes,
        ).to(self.device)

    def _new_strategy_network(self, role: Role):
        del role
        return self._strategy_network_type(
            self.game.information_state_size,
            self.game.num_actions,
            self.config.hidden_sizes,
        ).to(self.device)

    def _set_global_seed(self, seed: int) -> None:
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)


class DeepCFRPolicy:
    """Inference wrapper for regret values and the learned average strategy."""

    def __init__(self, trainer: DeepCFRTrainer) -> None:
        self.trainer = trainer
        self.game = trainer.game

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> DeepCFRPolicy:
        return cls(DeepCFRTrainer.load_checkpoint(path, device=device))

    def predicted_advantages(self, state: AbstractWerewolfState) -> dict[int, float]:
        if not state.is_decision_node:
            raise ValueError("predicted advantages require a decision state")
        values = self.trainer._predict(
            self.trainer.advantage_networks[state.current_role],
            state.information_state_tensor(),
        )
        return {action_id: values[action_id] for action_id in state.legal_actions()}

    def action_probabilities(
        self,
        state: AbstractWerewolfState,
        *,
        average_strategy: bool = True,
    ) -> dict[int, float]:
        if not state.is_decision_node:
            raise ValueError("action probabilities require a decision state")
        legal_actions = state.legal_actions()
        if not average_strategy:
            strategy = self.trainer._regret_matching_strategy(
                state.current_role,
                state.information_state_tensor(),
                legal_actions,
            )
            return {action_id: strategy[action_id] for action_id in legal_actions}
        torch = self.trainer._torch
        network = self.trainer.strategy_networks[state.current_role]
        logits = self.trainer._predict(network, state.information_state_tensor())
        legal_logits = torch.tensor([logits[action_id] for action_id in legal_actions])
        probabilities = torch.softmax(legal_logits, dim=0).tolist()
        return dict(zip(legal_actions, probabilities, strict=True))

    def sample_action(
        self,
        state: AbstractWerewolfState,
        *,
        rng: random.Random | None = None,
        average_strategy: bool = True,
    ) -> int:
        probabilities = self.action_probabilities(
            state,
            average_strategy=average_strategy,
        )
        random_source = rng or random.Random()
        return _sample_distribution(random_source, tuple(probabilities.items()))


def train_deep_cfr(
    *,
    latent_space: LatentStrategySpace | Path,
    output_dir: Path,
    config: DeepCFRConfig | None = None,
    game_config: AbstractGameConfig | None = None,
) -> dict[str, Any]:
    if isinstance(latent_space, Path):
        resolved_space = LatentStrategySpace.load(latent_space)
    else:
        resolved_space = latent_space
    trainer = DeepCFRTrainer(
        AbstractWerewolfGame(resolved_space, config=game_config),
        config,
    )
    return trainer.train(output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wolfplay-train-deep-cfr")
    parser.add_argument("--latent-space", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--traversals-per-player", type=int, default=4)
    parser.add_argument("--advantage-train-steps", type=int, default=100)
    parser.add_argument("--strategy-train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--advantage-buffer-capacity", type=int, default=100_000)
    parser.add_argument("--strategy-buffer-capacity", type=int, default=100_000)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--max-traversal-depth", type=int, default=64)
    parser.add_argument("--max-rollout-steps", type=int, default=512)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--role-assignment-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-save-buffers", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> tuple[DeepCFRConfig, AbstractGameConfig]:
    config = DeepCFRConfig(
        iterations=args.iterations,
        traversals_per_player=args.traversals_per_player,
        advantage_train_steps=args.advantage_train_steps,
        strategy_train_steps=args.strategy_train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        advantage_buffer_capacity=args.advantage_buffer_capacity,
        strategy_buffer_capacity=args.strategy_buffer_capacity,
        hidden_sizes=tuple(args.hidden_sizes),
        max_traversal_depth=args.max_traversal_depth,
        max_rollout_steps=args.max_rollout_steps,
        seed=args.seed,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        save_buffers=not args.no_save_buffers,
    )
    game_config = AbstractGameConfig(
        max_rounds=args.max_rounds,
        role_assignment_limit=args.role_assignment_limit,
        role_assignment_seed=args.seed,
    )
    return config, game_config


def main() -> None:
    args = build_parser().parse_args()
    config, game_config = config_from_args(args)
    summary = train_deep_cfr(
        latent_space=args.latent_space,
        output_dir=args.output_dir,
        config=config,
        game_config=game_config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _training_dependencies():
    try:
        import torch

        from .torch_models import RegretNetwork, StrategyNetwork
    except ImportError as error:
        raise RuntimeError(
            "Deep CFR training requires the train dependencies: uv sync --extra train"
        ) from error
    return torch, RegretNetwork, StrategyNetwork


def _resolve_device(torch, requested: str):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _legal_mask(legal_actions: tuple[int, ...], action_count: int) -> tuple[bool, ...]:
    legal = set(legal_actions)
    return tuple(action_id in legal for action_id in range(action_count))


def _sample_action(
    rng: random.Random,
    probabilities: list[float],
    legal_actions: tuple[int, ...],
) -> int:
    return _sample_distribution(
        rng,
        tuple((action_id, probabilities[action_id]) for action_id in legal_actions),
    )


def _sample_distribution(
    rng: random.Random,
    outcomes: tuple[tuple[int, float], ...],
) -> int:
    if not outcomes:
        raise ValueError("cannot sample an empty distribution")
    if any(probability < 0 or not math.isfinite(probability) for _, probability in outcomes):
        raise ValueError("distribution probabilities must be finite and non-negative")
    total = sum(probability for _, probability in outcomes)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("distribution probabilities must have a positive finite sum")
    threshold = rng.random() * total
    cumulative = 0.0
    for outcome, probability in outcomes:
        cumulative += probability
        if probability > 0 and cumulative >= threshold:
            return outcome
    return outcomes[-1][0]


def _game_config_to_dict(config: AbstractGameConfig) -> dict[str, Any]:
    record = asdict(config)
    record["rewards"] = asdict(config.rewards)
    return record


def _game_config_from_dict(record: dict[str, Any]) -> AbstractGameConfig:
    values = dict(record)
    values["rewards"] = RewardConfig(**values["rewards"])
    return AbstractGameConfig(**values)


def _checkpoint_path(path: Path) -> Path:
    checkpoint = path / "deep_cfr.pt" if path.is_dir() else path
    if not checkpoint.is_file():
        raise ValueError(f"Deep CFR checkpoint does not exist: {checkpoint}")
    return checkpoint


def _cpu_state_dict(network) -> dict[str, Any]:
    return {name: value.detach().cpu() for name, value in network.state_dict().items()}


def _atomic_torch_save(torch, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, record: dict[str, Any]) -> None:
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
            output.write(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _positive_float(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


if __name__ == "__main__":
    main()
