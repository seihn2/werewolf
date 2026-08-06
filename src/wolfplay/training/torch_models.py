from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.normalization = nn.LayerNorm(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.normalization(inputs + torch.relu(self.linear(inputs)))


class _PolicyMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_sizes: Sequence[int],
        *,
        zero_output: bool,
    ) -> None:
        super().__init__()
        if input_size <= 0 or action_size <= 0:
            raise ValueError("input_size and action_size must be positive")
        if not hidden_sizes or any(width <= 0 for width in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        layers: list[nn.Module] = []
        previous = hidden_sizes[0]
        for width in hidden_sizes[1:]:
            if width == previous:
                layers.append(_ResidualBlock(width))
            else:
                layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.LayerNorm(width)))
            previous = width
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(previous, action_size)
        if zero_output:
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input_layer(inputs))
        hidden = self.hidden_layers(hidden)
        return self.output_layer(hidden)


class RegretNetwork(_PolicyMLP):
    """Approximates instantaneous action advantages for regret matching."""

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_sizes: Sequence[int] = (256, 256, 256),
    ) -> None:
        super().__init__(input_size, action_size, hidden_sizes, zero_output=True)


class StrategyNetwork(_PolicyMLP):
    """Approximates the iteration-weighted average strategy as action logits."""

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_sizes: Sequence[int] = (256, 256, 256),
    ) -> None:
        super().__init__(input_size, action_size, hidden_sizes, zero_output=True)
