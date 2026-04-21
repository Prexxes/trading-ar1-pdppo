"""Tests for the PDPPO agent."""

from __future__ import annotations

import shutil
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from agents.pdppo_agent import PDPPOAgent, PDPPOConfig
from envs.trading_ar1_env import TradingAr1Env, TradingAr1EnvConfig
from train.train_pdppo import train_pdppo


def make_env() -> TradingAr1Env:
    return TradingAr1Env(
        TradingAr1EnvConfig(
            episode_length=8,
            max_inventory=3,
            initial_cash=100.0,
            initial_price=10.0,
            sigma=0.01,
            post_reward_mode="cash_interest_and_tc",
            seed=13,
        )
    )


def make_agent(env: TradingAr1Env) -> PDPPOAgent:
    return PDPPOAgent(
        PDPPOConfig(
            obs_dim=4,
            action_dim=env.action_space.n,
            rollout_steps=16,
            minibatch_size=8,
            update_epochs=2,
        )
    )


def test_forward_pass_shapes() -> None:
    env = make_env()
    agent = make_agent(env)
    observation, info = env.reset()
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.as_tensor(info["action_mask"], dtype=torch.float32).unsqueeze(0)

    policy_output = agent.actor(observation_tensor, mask_tensor)
    state_value = agent.critic_state(observation_tensor)
    post_value = agent.critic_post(observation_tensor)

    assert policy_output.logits.shape == (1, env.action_space.n)
    assert state_value.shape == (1,)
    assert post_value.shape == (1,)


def test_post_state_handling_works() -> None:
    env = make_env()
    agent = make_agent(env)
    observation, info = env.reset()
    action, _, _ = agent.select_action(observation, info["action_mask"])
    post_observation, post_reward = env.get_post_decision_state(action)

    assert post_observation.shape == (4,)
    assert isinstance(post_reward, float)


def test_rollout_buffer_contains_post_fields() -> None:
    env = make_env()
    agent = make_agent(env)
    buffer, _ = agent.collect_rollout(env)
    tensors = buffer.as_tensors(torch.device("cpu"))

    assert tensors["post_observations"].shape[1] == 4
    assert tensors["post_rewards"].shape[0] == len(buffer)
    assert tensors["stochastic_rewards"].shape[0] == len(buffer)
    assert tensors["post_values"].shape[0] == len(buffer)


def test_pdppo_update_runs_without_shape_errors() -> None:
    env = make_env()
    agent = make_agent(env)
    buffer, _ = agent.collect_rollout(env)
    losses = agent.update(buffer)

    assert "critic_loss" in losses
    assert "post_critic_loss" in losses
    assert "actor_loss" in losses


def test_actor_advantage_max_path_runs() -> None:
    env = make_env()
    agent = make_agent(env)
    buffer, _ = agent.collect_rollout(env)
    batch = buffer.as_tensors(torch.device("cpu"))
    adv_state, _, _, adv_post = agent._compute_targets(batch)
    adv_actor = torch.maximum(adv_state, adv_post)

    assert adv_actor.shape == adv_state.shape


def test_pdppo_smoke_training_runs() -> None:
    env_config = TradingAr1EnvConfig(
        episode_length=8,
        max_inventory=3,
        initial_cash=100.0,
        initial_price=10.0,
        sigma=0.01,
        post_reward_mode="cash_interest_and_tc",
        seed=17,
    )
    agent_config = PDPPOConfig(
        obs_dim=4,
        action_dim=2 * env_config.max_inventory + 1,
        rollout_steps=16,
        minibatch_size=8,
        update_epochs=1,
    )

    base_temporary_directory = Path("test_runs")
    base_temporary_directory.mkdir(parents=True, exist_ok=True)
    output_dir = base_temporary_directory / f"pdppo_{uuid4().hex}"

    try:
        metrics = train_pdppo(
            env_config,
            agent_config,
            total_updates=2,
            eval_interval=1,
            eval_episodes=2,
            output_dir=output_dir,
            seed=17,
        )

        assert (output_dir / "training_log.csv").exists()
        metadata_path = output_dir / "run_metadata.json"
        assert metadata_path.exists()
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        assert metadata["algorithm"] == "PDPPO"
        assert metadata["environment_config"]["post_reward_mode"] == env_config.post_reward_mode
        assert metadata["agent_config"]["rollout_steps"] == agent_config.rollout_steps
        assert metadata["training_config"]["total_updates"] == 2
        assert metadata["duration"]["seconds"] >= 0.0
        assert (output_dir / "checkpoints" / "last.pt").exists()
        assert "post_critic_loss" in metrics
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
