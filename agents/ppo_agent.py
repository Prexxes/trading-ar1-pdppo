"""PPO agent for the trading environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from agents.networks import ActorNetwork, CriticNetwork
from buffers.rollout_buffer import RolloutBuffer


@dataclass(slots=True)
class PPOConfig:
    """Configuration for PPO training and optimization."""

    obs_dim: int
    action_dim: int
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    rollout_steps: int = 512
    max_grad_norm: float = 0.5
    device: str = "cpu"


class PPOAgent:
    """Masked-action PPO agent with actor-critic networks."""

    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.actor = ActorNetwork(config.obs_dim, config.action_dim).to(self.device)
        self.critic = CriticNetwork(config.obs_dim).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.critic_lr)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Selects an action using the masked policy."""
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            action_mask,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            policy_output = self.actor(observation_tensor, mask_tensor)
            value = self.critic(observation_tensor)
            if deterministic:
                action = torch.argmax(policy_output.masked_logits, dim=-1)
            else:
                action = policy_output.distribution.sample()
            log_prob = policy_output.distribution.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def collect_rollout(self, env) -> tuple[RolloutBuffer, list[float]]:
        """Collects one rollout of on-policy experience."""
        buffer = RolloutBuffer()
        episode_returns: list[float] = []
        observation, info = env.reset()
        action_mask = info["action_mask"]
        running_return = 0.0

        for _ in range(self.config.rollout_steps):
            action, log_prob, value = self.select_action(observation, action_mask)
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            buffer.add(
                observation=observation,
                next_observation=next_observation,
                action_mask=action_mask,
                action=action,
                log_prob=log_prob,
                reward=reward,
                state_value=value,
                done=done,
                truncated=truncated,
            )

            running_return += reward
            observation = next_observation
            action_mask = next_info["action_mask"]

            if done:
                episode_returns.append(running_return)
                running_return = 0.0
                observation, info = env.reset()
                action_mask = info["action_mask"]

        return buffer, episode_returns

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Runs PPO updates for a collected rollout."""
        batch = buffer.as_tensors(self.device)
        advantages, returns = self._compute_gae(
            rewards=batch["rewards"],
            values=batch["state_values"],
            dones=batch["dones"],
            next_observations=batch["next_observations"],
        )
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss_value = 0.0
        critic_loss_value = 0.0
        batch_size = batch["actions"].shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                actor_loss, critic_loss = self._update_minibatch(
                    observations=batch["observations"][indices],
                    action_masks=batch["action_masks"][indices],
                    actions=batch["actions"][indices],
                    old_log_probs=batch["log_probs"][indices],
                    advantages=normalized_advantages[indices],
                    returns=returns[indices],
                )
                actor_loss_value = float(actor_loss)
                critic_loss_value = float(critic_loss)

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
        }

    def _update_minibatch(
        self,
        *,
        observations: torch.Tensor,
        action_masks: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple[float, float]:
        policy_output = self.actor(observations, action_masks)
        new_log_probs = policy_output.distribution.log_prob(actions)
        entropy = policy_output.distribution.entropy().mean()
        ratios = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratios * advantages
        clipped = torch.clamp(
            ratios,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        ) * advantages
        actor_loss = -(torch.minimum(unclipped, clipped).mean() + self.config.entropy_coef * entropy)

        predicted_values = self.critic(observations)
        critic_loss = self.config.value_coef * torch.mean((returns - predicted_values) ** 2)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        return float(actor_loss.item()), float(critic_loss.item())

    def _compute_gae(
        self,
        *,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            next_values = self.critic(next_observations)

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(1, device=self.device, dtype=torch.float32)
        for index in reversed(range(rewards.shape[0])):
            mask = 1.0 - dones[index]
            delta = rewards[index] + self.config.gamma * next_values[index] * mask - values[index]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[index] = gae

        returns = advantages + values
        return advantages, returns

    def save(self, path: str | Path) -> None:
        """Saves model and optimizer state."""
        torch.save(
            {
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            Path(path),
        )
