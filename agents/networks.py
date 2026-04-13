"""Neural network modules for PPO and PDPPO agents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical


def build_mlp(input_dim: int, output_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    """Builds a two-hidden-layer MLP with Tanh activations."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


def apply_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Applies a binary action mask to logits."""
    mask = action_mask.to(dtype=torch.bool)
    invalid_fill = torch.finfo(logits.dtype).min
    return torch.where(mask, logits, torch.full_like(logits, invalid_fill))


@dataclass(slots=True)
class PolicyOutput:
    """Policy evaluation output."""

    logits: torch.Tensor
    masked_logits: torch.Tensor
    distribution: Categorical


class ActorNetwork(nn.Module):
    """Discrete actor network for masked action selection."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.model = build_mlp(obs_dim, action_dim)

    def forward(self, observations: torch.Tensor, action_mask: torch.Tensor) -> PolicyOutput:
        logits = self.model(observations)
        masked_logits = apply_action_mask(logits, action_mask)
        distribution = Categorical(logits=masked_logits)
        return PolicyOutput(logits=logits, masked_logits=masked_logits, distribution=distribution)


class CriticNetwork(nn.Module):
    """State-value network."""

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.model = build_mlp(obs_dim, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations).squeeze(-1)
