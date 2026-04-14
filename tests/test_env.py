"""Tests for the AR(1) trading environment."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from envs.trading_ar1_env import TradingAr1Env, TradingAr1EnvConfig


def make_env(**overrides) -> TradingAr1Env:
    config_kwargs = {
        "episode_length": 5,
        "initial_cash": 100.0,
        "initial_inventory": 1,
        "max_inventory": 4,
        "initial_price": 10.0,
        "sigma": 0.0,
        "phi": 1.0,
        "seed": 123,
    }
    config_kwargs.update(overrides)
    config = TradingAr1EnvConfig(
        **config_kwargs,
    )
    return TradingAr1Env(config)


def test_reset_returns_valid_observation_and_info() -> None:
    env = make_env()
    observation, info = env.reset()

    assert observation.shape == (4,)
    assert observation.dtype == np.float32
    assert info["action_mask"].shape == (2 * env.config.max_inventory + 1,)
    assert info["price"] > 0.0


def test_step_is_gymnasium_compatible() -> None:
    env = make_env()
    check_env(env, skip_render_check=True)


def test_constraints_hold_across_steps() -> None:
    env = make_env(post_reward_mode="cash_interest_and_tc")
    observation, info = env.reset()
    action_mask = info["action_mask"]

    for _ in range(6):
        valid_action = int(np.flatnonzero(action_mask)[-1])
        observation, _, terminated, truncated, info = env.step(valid_action)
        assert observation.shape == (4,)
        assert env.cash >= 0.0
        assert env.inventory >= 0
        assert env.inventory <= env.config.max_inventory
        assert env.price > 0.0
        action_mask = info["action_mask"]
        if terminated or truncated:
            break


def test_action_mask_respects_cash_inventory_and_trade_cap() -> None:
    env = make_env(initial_cash=5.0, initial_inventory=0, max_trade_per_day=1)
    _, info = env.reset()
    mask = info["action_mask"]

    assert mask[env.config.max_inventory] == 1
    assert mask[env.config.max_inventory + 1] == 0
    assert mask[env.config.max_inventory - 1] == 0


def test_invalid_action_falls_back_to_hold() -> None:
    env = make_env(initial_cash=1.0, initial_inventory=0)
    _, info = env.reset()
    requested_action = env.config.max_inventory + 1
    _, _, _, _, next_info = env.step(requested_action)

    assert next_info["invalid_action"] is True
    assert next_info["requested_action"] == requested_action
    assert next_info["executed_action"] == env.config.max_inventory


def test_reward_matches_portfolio_change() -> None:
    env = make_env(post_reward_mode="none")
    env.reset()
    pre_value = env.cash + env.inventory * env.price
    _, reward, _, _, _ = env.step(env.config.max_inventory + 1)
    next_value = env.cash + env.inventory * env.price

    assert np.isclose(reward, next_value - pre_value)


def test_post_decision_state_matches_expected_and_does_not_mutate() -> None:
    env = make_env(post_reward_mode="cash_interest_and_tc")
    env.reset()
    original_state = (env.day, env.inventory, env.cash, env.price, env.log_price)
    action = env.config.max_inventory + 1
    post_observation, post_reward = env.get_post_decision_state(action)

    expected_trade = 1
    transaction_cost = env.config.transaction_cost_rate * expected_trade * env.price
    expected_cash = env.config.initial_cash - expected_trade * env.price - transaction_cost

    assert np.isclose(post_observation[1], (env.config.initial_inventory + 1) / env.config.max_inventory)
    assert np.isclose(post_observation[2], expected_cash / env.config.initial_cash)
    assert np.isclose(
        post_reward,
        -transaction_cost + expected_cash * env.daily_risk_free_rate,
    )
    assert original_state == (env.day, env.inventory, env.cash, env.price, env.log_price)


def test_reward_decomposition_is_exposed() -> None:
    env = make_env(post_reward_mode="cash_interest_and_tc")
    env.reset()
    _, reward, _, _, info = env.step(env.config.max_inventory + 1)

    assert np.isclose(reward, info["post_reward"] + info["stochastic_reward"])


def test_initial_price_and_mu_are_interpreted_as_price_levels() -> None:
    env = make_env(initial_price=40.0, mu=20.0, sigma=0.0, phi=0.0)
    observation, info = env.reset()

    assert np.isclose(env.log_price, np.log(40.0))
    assert np.isclose(info["price"], 40.0)

    env.step(env.config.max_inventory)
    assert np.isclose(env.log_price, np.log(20.0))
    assert np.isclose(env.price, 20.0)
