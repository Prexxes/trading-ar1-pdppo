"""PDPPO agent with dual critics and post-decision states."""

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
class PDPPOConfig:
    """Configuration for PDPPO training and optimization."""

    obs_dim: int
    action_dim: int
    actor_lr: float = 3e-4
    critic_state_lr: float = 1e-3
    critic_post_lr: float = 1e-3
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


class PDPPOAgent:
    """PDPPO agent with a state critic and a post-decision critic."""

    def __init__(self, config: PDPPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.actor = ActorNetwork(config.obs_dim, config.action_dim).to(self.device)
        self.critic_state = CriticNetwork(config.obs_dim).to(self.device)
        self.critic_post = CriticNetwork(config.obs_dim).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.actor_lr)
        self.state_optimizer = Adam(self.critic_state.parameters(), lr=config.critic_state_lr)
        self.post_optimizer = Adam(self.critic_post.parameters(), lr=config.critic_post_lr)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Selects an action using the masked actor."""
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
            state_value = self.critic_state(observation_tensor)
            if deterministic:
                action = torch.argmax(policy_output.masked_logits, dim=-1)
            else:
                action = policy_output.distribution.sample()
            log_prob = policy_output.distribution.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(state_value.item())

    def collect_rollout(self, env) -> tuple[RolloutBuffer, list[float]]:
        """Collects experience including post-decision states."""
        buffer = RolloutBuffer()
        episode_returns: list[float] = []
        observation, info = env.reset()
        action_mask = info["action_mask"]
        running_return = 0.0

        for _ in range(self.config.rollout_steps):
            action, log_prob, state_value = self.select_action(observation, action_mask)
            post_observation, _ = env.get_post_decision_state(action)
            with torch.no_grad():
                post_value = self.critic_post(
                    torch.as_tensor(post_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                )

            next_observation, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            buffer.add(
                observation=observation,
                next_observation=next_observation,
                post_observation=post_observation,
                action_mask=action_mask,
                action=action,
                log_prob=log_prob,
                reward=reward,
                post_reward=float(next_info["post_reward"]),
                stochastic_reward=float(next_info["stochastic_reward"]),
                state_value=state_value,
                post_value=float(post_value.item()),
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
        """Runs PDPPO updates."""
        batch = buffer.as_tensors(self.device)
        adv_state, state_returns, post_targets, adv_post = self._compute_targets(batch)
        adv_actor = torch.maximum(adv_state, adv_post)
        adv_actor = (adv_actor - adv_actor.mean()) / (adv_actor.std() + 1e-8)

        actor_loss_value = 0.0
        state_critic_loss_value = 0.0
        post_critic_loss_value = 0.0
        batch_size = batch["actions"].shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                actor_loss, state_loss, post_loss = self._update_minibatch(
                    observations=batch["observations"][indices],
                    post_observations=batch["post_observations"][indices],
                    action_masks=batch["action_masks"][indices],
                    actions=batch["actions"][indices],
                    old_log_probs=batch["log_probs"][indices],
                    actor_advantages=adv_actor[indices],
                    state_returns=state_returns[indices],
                    post_targets=post_targets[indices],
                )
                actor_loss_value = float(actor_loss)
                state_critic_loss_value = float(state_loss)
                post_critic_loss_value = float(post_loss)

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": state_critic_loss_value,
            "post_critic_loss": post_critic_loss_value,
        }

    def _compute_targets(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            next_state_values = self.critic_state(batch["next_observations"])

        adv_state = torch.zeros_like(batch["rewards"])
        gae = torch.zeros(1, device=self.device, dtype=torch.float32)
        for index in reversed(range(batch["rewards"].shape[0])):
            mask = 1.0 - batch["dones"][index]
            delta = (
                batch["rewards"][index]
                + self.config.gamma * next_state_values[index] * mask
                - batch["state_values"][index]
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            adv_state[index] = gae

        state_returns = adv_state + batch["state_values"]
        post_targets = (
            batch["stochastic_rewards"]
            + self.config.gamma * next_state_values * (1.0 - batch["dones"])
        )
        adv_post = post_targets - batch["post_values"]
        return adv_state, state_returns, post_targets, adv_post

    def _update_minibatch(
        self,
        *,
        observations: torch.Tensor,
        post_observations: torch.Tensor,
        action_masks: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        actor_advantages: torch.Tensor,
        state_returns: torch.Tensor,
        post_targets: torch.Tensor,
    ) -> tuple[float, float, float]:
        policy_output = self.actor(observations, action_masks)
        new_log_probs = policy_output.distribution.log_prob(actions)
        entropy = policy_output.distribution.entropy().mean()
        ratios = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratios * actor_advantages
        clipped = torch.clamp(
            ratios,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        ) * actor_advantages
        actor_loss = -(torch.minimum(unclipped, clipped).mean() + self.config.entropy_coef * entropy)

        state_predictions = self.critic_state(observations)
        post_predictions = self.critic_post(post_observations)
        state_loss = self.config.value_coef * torch.mean((state_returns - state_predictions) ** 2)
        post_loss = self.config.value_coef * torch.mean((post_targets - post_predictions) ** 2)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        self.state_optimizer.zero_grad()
        state_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_state.parameters(), self.config.max_grad_norm)
        self.state_optimizer.step()

        self.post_optimizer.zero_grad()
        post_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_post.parameters(), self.config.max_grad_norm)
        self.post_optimizer.step()

        return float(actor_loss.item()), float(state_loss.item()), float(post_loss.item())

    def save(self, path: str | Path) -> None:
        """Saves model and optimizer state."""
        torch.save(
            {
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "critic_state": self.critic_state.state_dict(),
                "critic_post": self.critic_post.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "state_optimizer": self.state_optimizer.state_dict(),
                "post_optimizer": self.post_optimizer.state_dict(),
            },
            Path(path),
        )
