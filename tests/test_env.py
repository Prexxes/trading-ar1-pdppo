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
        "sigma": 0.0,  # No noise
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
    # Buy up to the max_inventory, then hold
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
    # Can not buy anything, just hold
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
    action = env.config.max_inventory + 1  # Buy one
    post_observation, post_reward = env.get_post_decision_state(action)

    expected_trade = 1
    transaction_cost = env.config.transaction_cost_rate * expected_trade * env.price
    expected_cash = env.config.initial_cash - expected_trade * env.price - transaction_cost
    expected_cash_with_interest = expected_cash + expected_cash * env.daily_risk_free_rate

    assert np.isclose(post_observation[1], (env.config.initial_inventory + 1) / env.config.max_inventory)
    assert np.isclose(post_observation[2], expected_cash_with_interest / env.config.initial_cash)
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


def test_price_path_is_reproducible_with_fixed_seed() -> None:
    env_a = TradingAr1Env(
        TradingAr1EnvConfig(
            episode_length=5,
            initial_cash=100.0,
            initial_inventory=0,
            max_inventory=4,
            initial_price=10.0,
            mu=10.0,
            phi=0.8,
            sigma=0.05,
            seed=123,
        )
    )
    env_b = TradingAr1Env(
        TradingAr1EnvConfig(
            episode_length=5,
            initial_cash=100.0,
            initial_inventory=0,
            max_inventory=4,
            initial_price=10.0,
            mu=10.0,
            phi=0.8,
            sigma=0.05,
            seed=123,
        )
    )

    env_a.reset()
    env_b.reset()

    prices_a = [env_a.price]
    prices_b = [env_b.price]

    hold_action = env_a.config.max_inventory
    for _ in range(4):
        env_a.step(hold_action)
        env_b.step(hold_action)
        prices_a.append(env_a.price)
        prices_b.append(env_b.price)

    assert np.allclose(prices_a, prices_b)


def test_price_changes_when_sigma_is_positive() -> None:
    env = make_env(
        initial_price=10.0,
        mu=10.0,
        phi=1.0,
        sigma=0.05,
        seed=7,
    )
    env.reset()

    initial_price = env.price
    hold_action = env.config.max_inventory
    env.step(hold_action)

    assert not np.isclose(env.price, initial_price)
    assert env.price > 0.0


def test_stochastic_reward_matches_price_change_when_sigma_positive() -> None:
    env = make_env(
        initial_price=10.0,
        mu=10.0,
        phi=0.9,
        sigma=0.05,
        initial_inventory=2,
        seed=21,
    )
    env.reset()

    old_price = env.price
    old_inventory = env.inventory
    hold_action = env.config.max_inventory

    _, _, _, _, info = env.step(hold_action)

    expected_stochastic_reward = old_inventory * (env.price - old_price)

    assert np.isclose(info["stochastic_reward"], expected_stochastic_reward)
    assert env.price > 0.0


def test_larger_sigma_creates_larger_log_price_dispersion() -> None:
    env_small = TradingAr1Env(
        TradingAr1EnvConfig(
            episode_length=200,
            initial_cash=100.0,
            initial_inventory=0,
            max_inventory=4,
            initial_price=10.0,
            mu=10.0,
            phi=1.0,
            sigma=0.01,
            seed=5,
        )
    )
    env_large = TradingAr1Env(
        TradingAr1EnvConfig(
            episode_length=200,
            initial_cash=100.0,
            initial_inventory=0,
            max_inventory=4,
            initial_price=10.0,
            mu=10.0,
            phi=1.0,
            sigma=0.10,
            seed=5,
        )
    )

    env_small.reset()
    env_large.reset()

    hold_action = env_small.config.max_inventory
    log_prices_small = []
    log_prices_large = []

    for _ in range(100):
        env_small.step(hold_action)
        env_large.step(hold_action)
        log_prices_small.append(env_small.log_price)
        log_prices_large.append(env_large.log_price)

    assert np.std(log_prices_large) > np.std(log_prices_small)
