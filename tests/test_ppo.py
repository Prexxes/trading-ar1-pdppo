"""Tests for the PPO agent."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from agents.ppo_agent import PPOAgent, PPOConfig
from buffers.rollout_buffer import RolloutBuffer
from envs.trading_ar1_env import TradingAr1Env, TradingAr1EnvConfig
from train.train_ppo import train_ppo


def make_env() -> TradingAr1Env:
    return TradingAr1Env(
        TradingAr1EnvConfig(
            episode_length=8,
            max_inventory=3,
            initial_cash=100.0,
            initial_price=10.0,
            sigma=0.01,
            seed=5,
        )
    )


def make_agent(env: TradingAr1Env) -> PPOAgent:
    return PPOAgent(
        PPOConfig(
            obs_dim=4,
            action_dim=env.action_space.n,
            rollout_steps=16,
            minibatch_size=8,
            update_epochs=2,
        )
    )


def make_output_dir(name: str) -> Path:
    output_dir = Path("test_runs") / f"{name}_{uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def test_forward_pass_shapes() -> None:
    env = make_env()
    agent = make_agent(env)
    observation, info = env.reset()
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.as_tensor(info["action_mask"], dtype=torch.float32).unsqueeze(0)
    policy_output = agent.actor(observation_tensor, mask_tensor)
    values = agent.critic(observation_tensor)

    assert policy_output.logits.shape == (1, env.action_space.n)
    assert policy_output.masked_logits.shape == (1, env.action_space.n)
    assert values.shape == (1,)


def test_masked_action_sampling_respects_invalid_actions() -> None:
    env = make_env()
    agent = make_agent(env)
    observation, _ = env.reset()
    action_mask = np.zeros(env.action_space.n, dtype=np.int8)
    action_mask[2] = 1

    samples = [agent.select_action(observation, action_mask)[0] for _ in range(25)]
    assert set(samples) == {2}


def test_rollout_buffer_stores_expected_shapes() -> None:
    buffer = RolloutBuffer()
    observation = np.zeros(4, dtype=np.float32)
    next_observation = np.ones(4, dtype=np.float32)
    action_mask = np.ones(7, dtype=np.int8)

    buffer.add(
        observation=observation,
        next_observation=next_observation,
        action_mask=action_mask,
        action=3,
        log_prob=-0.5,
        reward=1.0,
        state_value=0.2,
        done=False,
        truncated=False,
    )

    tensors = buffer.as_tensors(torch.device("cpu"))
    assert tensors["observations"].shape == (1, 4)
    assert tensors["action_masks"].shape == (1, 7)
    assert tensors["actions"].shape == (1,)


def test_ppo_update_runs_without_shape_errors() -> None:
    env = make_env()
    agent = make_agent(env)
    buffer, _ = agent.collect_rollout(env)
    losses = agent.update(buffer)

    assert "actor_loss" in losses
    assert "critic_loss" in losses


def test_ppo_smoke_training_runs() -> None:
    env_config = TradingAr1EnvConfig(
        episode_length=8,
        max_inventory=3,
        initial_cash=100.0,
        initial_price=10.0,
        sigma=0.01,
        seed=11,
    )
    agent_config = PPOConfig(
        obs_dim=4,
        action_dim=2 * env_config.max_inventory + 1,
        rollout_steps=16,
        minibatch_size=8,
        update_epochs=1,
    )

    output_dir = make_output_dir("ppo")
    metrics = train_ppo(
        env_config,
        agent_config,
        total_updates=2,
        eval_interval=1,
        eval_episodes=2,
        output_dir=output_dir,
        seed=11,
    )

    assert (output_dir / "training_log.csv").exists()
    assert (output_dir / "checkpoints" / "last.pt").exists()
    assert "actor_loss" in metrics
