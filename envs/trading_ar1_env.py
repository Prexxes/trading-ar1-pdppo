"""Single-asset trading environment with a log-AR(1) price process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PostRewardMode = Literal["none", "cash_interest", "cash_interest_and_tc"]


@dataclass(slots=True)
class TradingAr1EnvConfig:
    """Configuration for the trading environment."""

    episode_length: int = 365
    initial_cash: float = 1000.0
    initial_inventory: int = 0
    max_inventory: int = 20
    max_trade_per_day: int | None = None
    initial_price: float = 50.0
    initial_log_price: float | None = None
    mu: float = float(np.log(50.0))
    phi: float = 0.98
    sigma: float = 0.02
    post_reward_mode: PostRewardMode = "none"
    risk_free_rate_annual: float = 0.02
    transaction_cost_rate: float = 0.001
    seed: int | None = None


class TradingAr1Env(gym.Env[np.ndarray, int]):
    """Gymnasium environment for single-asset trading with AR(1) log-prices."""

    metadata = {"render_modes": []}

    def __init__(self, config: TradingAr1EnvConfig | None = None) -> None:
        self.config = config or TradingAr1EnvConfig()
        self._validate_config()

        action_dim = 2 * self.config.max_inventory + 1
        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, 1.0, np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.np_random, _ = gym.utils.seeding.np_random(self.config.seed)
        self.day = 0
        self.inventory = self.config.initial_inventory
        self.cash = self.config.initial_cash
        self.log_price = self._initial_log_price
        self.price = float(np.exp(self.log_price))

    @property
    def _initial_log_price(self) -> float:
        return (
            float(self.config.initial_log_price)
            if self.config.initial_log_price is not None
            else float(np.log(self.config.initial_price))
        )

    @property
    def daily_risk_free_rate(self) -> float:
        return (1.0 + self.config.risk_free_rate_annual) ** (1.0 / 365.0) - 1.0

    def _validate_config(self) -> None:
        if self.config.episode_length <= 0:
            raise ValueError("episode_length must be positive.")
        if self.config.initial_cash <= 0.0:
            raise ValueError("initial_cash must be positive.")
        if self.config.max_inventory <= 0:
            raise ValueError("max_inventory must be positive.")
        if not 0 <= self.config.initial_inventory <= self.config.max_inventory:
            raise ValueError("initial_inventory must be within [0, max_inventory].")
        if self.config.initial_price <= 0.0 and self.config.initial_log_price is None:
            raise ValueError("initial_price must be positive.")
        if self.config.max_trade_per_day is not None and self.config.max_trade_per_day < 0:
            raise ValueError("max_trade_per_day must be non-negative.")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.day = 0
        self.inventory = self.config.initial_inventory
        self.cash = self.config.initial_cash
        self.log_price = self._initial_log_price
        self.price = float(np.exp(self.log_price))

        observation = self._get_observation(
            log_price=self.log_price,
            inventory=self.inventory,
            cash=self.cash,
            day=self.day,
        )
        return observation, self._build_info(
            requested_action=None,
            executed_action=self._encode_trade(0),
            invalid_action=False,
            action_mask=self.get_action_mask(),
            post_reward=0.0,
            stochastic_reward=0.0,
            transaction_cost=0.0,
            cash_interest=0.0,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        requested_action = int(action)
        requested_trade = self._decode_action(requested_action)
        is_valid = self._is_valid_trade(requested_trade, self.inventory, self.cash, self.price)
        executed_trade = requested_trade if is_valid else 0
        invalid_action = not is_valid

        pre_value = self.cash + self.inventory * self.price
        transaction_cost = self._transaction_cost(executed_trade, self.price)
        post_cash = self._cash_after_trade(self.cash, executed_trade, self.price, transaction_cost)
        post_inventory = self.inventory + executed_trade

        next_log_price = self._sample_next_log_price(self.log_price)
        next_price = float(np.exp(next_log_price))

        cash_interest = 0.0
        if self.config.post_reward_mode in {"cash_interest", "cash_interest_and_tc"}:
            cash_interest = post_cash * self.daily_risk_free_rate
        next_cash = post_cash + cash_interest
        stochastic_reward = post_inventory * (next_price - self.price)
        post_reward = self._compute_post_reward(post_cash, transaction_cost)
        total_reward = float(post_reward + stochastic_reward)

        self.day += 1
        self.log_price = next_log_price
        self.price = next_price
        self.cash = float(next_cash)
        self.inventory = int(post_inventory)

        terminated = False
        truncated = self.day >= self.config.episode_length
        observation = self._get_observation(
            log_price=self.log_price,
            inventory=self.inventory,
            cash=self.cash,
            day=self.day,
        )

        info = self._build_info(
            requested_action=requested_action,
            executed_action=self._encode_trade(executed_trade),
            invalid_action=invalid_action,
            action_mask=self.get_action_mask(),
            post_reward=post_reward,
            stochastic_reward=stochastic_reward,
            transaction_cost=transaction_cost,
            cash_interest=cash_interest,
        )

        next_value = self.cash + self.inventory * self.price
        if not np.isclose(total_reward, next_value - pre_value, atol=1e-6):
            raise RuntimeError("Reward decomposition is inconsistent with portfolio dynamics.")

        return observation, total_reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """Returns a binary validity mask for the current action space."""
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for action_index in range(self.action_space.n):
            trade = self._decode_action(action_index)
            mask[action_index] = int(
                self._is_valid_trade(trade, self.inventory, self.cash, self.price)
            )
        return mask

    def get_post_decision_state(self, action_index: int) -> tuple[np.ndarray, float]:
        """Computes the post-decision observation and reward without mutation."""
        requested_trade = self._decode_action(int(action_index))
        is_valid = self._is_valid_trade(requested_trade, self.inventory, self.cash, self.price)
        executed_trade = requested_trade if is_valid else 0
        transaction_cost = self._transaction_cost(executed_trade, self.price)
        post_cash = self._cash_after_trade(self.cash, executed_trade, self.price, transaction_cost)
        post_inventory = self.inventory + executed_trade
        post_observation = self._get_observation(
            log_price=self.log_price,
            inventory=post_inventory,
            cash=post_cash,
            day=self.day,
        )
        return post_observation, self._compute_post_reward(post_cash, transaction_cost)

    def _sample_next_log_price(self, current_log_price: float) -> float:
        epsilon = self.np_random.normal()
        return float(
            self.config.mu
            + self.config.phi * (current_log_price - self.config.mu)
            + self.config.sigma * epsilon
        )

    def _build_info(
        self,
        *,
        requested_action: int | None,
        executed_action: int,
        invalid_action: bool,
        action_mask: np.ndarray,
        post_reward: float,
        stochastic_reward: float,
        transaction_cost: float,
        cash_interest: float,
    ) -> dict[str, Any]:
        return {
            "price": float(self.price),
            "log_price": float(self.log_price),
            "inventory": int(self.inventory),
            "cash": float(self.cash),
            "portfolio_value": float(self.cash + self.inventory * self.price),
            "requested_action": requested_action,
            "executed_action": int(executed_action),
            "invalid_action": bool(invalid_action),
            "action_mask": action_mask.copy(),
            "post_reward": float(post_reward),
            "stochastic_reward": float(stochastic_reward),
            "transaction_cost": float(transaction_cost),
            "cash_interest": float(cash_interest),
        }

    def _cash_after_trade(
        self,
        cash: float,
        trade: int,
        price: float,
        transaction_cost: float,
    ) -> float:
        return float(cash - trade * price - transaction_cost)

    def _compute_post_reward(self, post_cash: float, transaction_cost: float) -> float:
        if self.config.post_reward_mode == "none":
            return 0.0
        cash_interest = post_cash * self.daily_risk_free_rate
        if self.config.post_reward_mode == "cash_interest":
            return float(cash_interest)
        return float(-transaction_cost + cash_interest)

    def _transaction_cost(self, trade: int, price: float) -> float:
        if self.config.post_reward_mode != "cash_interest_and_tc":
            return 0.0
        return float(self.config.transaction_cost_rate * abs(trade) * price)

    def _decode_action(self, action_index: int) -> int:
        return int(action_index - self.config.max_inventory)

    def _encode_trade(self, trade: int) -> int:
        return int(trade + self.config.max_inventory)

    def _is_valid_trade(self, trade: int, inventory: int, cash: float, price: float) -> bool:
        next_inventory = inventory + trade
        if next_inventory < 0 or next_inventory > self.config.max_inventory:
            return False
        if self.config.max_trade_per_day is not None and abs(trade) > self.config.max_trade_per_day:
            return False
        transaction_cost = self._transaction_cost(trade, price)
        next_cash = self._cash_after_trade(cash, trade, price, transaction_cost)
        return next_cash >= -1e-8

    def _get_observation(
        self,
        *,
        log_price: float,
        inventory: int,
        cash: float,
        day: int,
    ) -> np.ndarray:
        day_denominator = max(self.config.episode_length - 1, 1)
        day_ratio = min(day, self.config.episode_length - 1) / day_denominator
        observation = np.array(
            [
                log_price,
                inventory / self.config.max_inventory,
                cash / self.config.initial_cash,
                day_ratio,
            ],
            dtype=np.float32,
        )
        return observation
