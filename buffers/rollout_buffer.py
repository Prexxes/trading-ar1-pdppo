"""Rollout buffer used by PPO and PDPPO agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class RolloutBuffer:
    """Stores trajectories and converts them into batched tensors."""

    observations: list[np.ndarray] = field(default_factory=list)
    next_observations: list[np.ndarray] = field(default_factory=list)
    post_observations: list[np.ndarray] = field(default_factory=list)
    action_masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    post_rewards: list[float] = field(default_factory=list)
    stochastic_rewards: list[float] = field(default_factory=list)
    state_values: list[float] = field(default_factory=list)
    post_values: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)
    truncations: list[float] = field(default_factory=list)

    def add(
        self,
        *,
        observation: np.ndarray,
        next_observation: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        state_value: float,
        done: bool,
        truncated: bool,
        post_observation: np.ndarray | None = None,
        post_reward: float | None = None,
        stochastic_reward: float | None = None,
        post_value: float | None = None,
    ) -> None:
        """Adds a transition to the buffer."""
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.next_observations.append(np.asarray(next_observation, dtype=np.float32))
        self.action_masks.append(np.asarray(action_mask, dtype=np.int8))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.state_values.append(float(state_value))
        self.dones.append(float(done))
        self.truncations.append(float(truncated))

        if post_observation is not None:
            self.post_observations.append(np.asarray(post_observation, dtype=np.float32))
        if post_reward is not None:
            self.post_rewards.append(float(post_reward))
        if stochastic_reward is not None:
            self.stochastic_rewards.append(float(stochastic_reward))
        if post_value is not None:
            self.post_values.append(float(post_value))

    def __len__(self) -> int:
        return len(self.actions)

    def as_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Converts the buffer contents into tensors on the target device."""
        tensor_map: dict[str, Any] = {
            "observations": np.asarray(self.observations, dtype=np.float32),
            "next_observations": np.asarray(self.next_observations, dtype=np.float32),
            "action_masks": np.asarray(self.action_masks, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "state_values": np.asarray(self.state_values, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "truncations": np.asarray(self.truncations, dtype=np.float32),
        }

        if self.post_observations:
            tensor_map["post_observations"] = np.asarray(self.post_observations, dtype=np.float32)
        if self.post_rewards:
            tensor_map["post_rewards"] = np.asarray(self.post_rewards, dtype=np.float32)
        if self.stochastic_rewards:
            tensor_map["stochastic_rewards"] = np.asarray(
                self.stochastic_rewards,
                dtype=np.float32,
            )
        if self.post_values:
            tensor_map["post_values"] = np.asarray(self.post_values, dtype=np.float32)

        return {key: torch.as_tensor(value, device=device) for key, value in tensor_map.items()}
