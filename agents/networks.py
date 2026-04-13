"""Neural network modules for PPO and PDPPO agents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical


def build_mlp(input_dim: int, output_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    """Build a two-hidden-layer MLP with Tanh activations.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output features.
        hidden_dim: Width of both hidden layers.

    Returns:
        A feed-forward network with two hidden layers.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


def apply_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Apply a binary action mask to policy logits.

    Args:
        logits: Unmasked action logits.
        action_mask: Binary mask where valid actions are marked with truthy values.

    Returns:
        Logits where invalid actions are replaced by the minimum finite value.
    """
    mask = action_mask.to(dtype=torch.bool)
    invalid_fill = torch.finfo(logits.dtype).min
    return torch.where(mask, logits, torch.full_like(logits, invalid_fill))


@dataclass(slots=True)
class PolicyOutput:
    """Container for policy network outputs.

    Attributes:
        logits: Raw logits before masking.
        masked_logits: Logits after invalid actions have been suppressed.
        distribution: Categorical distribution induced by the masked logits.
    """

    logits: torch.Tensor
    masked_logits: torch.Tensor
    distribution: Categorical


class ActorNetwork(nn.Module):
    """Discrete actor network for masked action selection."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        """Initialize the actor network.

        Args:
            obs_dim: Dimension of the observation vector.
            action_dim: Number of discrete actions.
        """
        super().__init__()
        self.model = build_mlp(obs_dim, action_dim)

    def forward(self, observations: torch.Tensor, action_mask: torch.Tensor) -> PolicyOutput:
        """Evaluate the policy for a batch of observations.

        Args:
            observations: Batched environment observations.
            action_mask: Batched binary action masks aligned with ``observations``.

        Returns:
            The raw logits, masked logits, and resulting categorical distribution.
        """
        logits = self.model(observations)
        masked_logits = apply_action_mask(logits, action_mask)
        distribution = Categorical(logits=masked_logits)
        return PolicyOutput(logits=logits, masked_logits=masked_logits, distribution=distribution)


class CriticNetwork(nn.Module):
    """State-value network."""

    def __init__(self, obs_dim: int) -> None:
        """Initialize the critic network.

        Args:
            obs_dim: Dimension of the observation vector.
        """
        super().__init__()
        self.model = build_mlp(obs_dim, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Estimate state values for a batch of observations.

        Args:
            observations: Batched environment observations.

        Returns:
            A one-dimensional tensor of predicted state values.
        """
        return self.model(observations).squeeze(-1)
