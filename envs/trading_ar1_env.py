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
    """Configuration for the trading environment.

    Attributes:
        episode_length: Number of trading days per episode.
        initial_cash: Initial cash balance.
        initial_inventory: Initial number of units held.
        max_inventory: Maximum allowed inventory level.
        max_trade_per_day: Optional cap on absolute daily trade size.
        initial_price: Initial asset price used to initialize the log-price state.
        mu: Long-run mean price level of the AR(1) process.
        phi: Persistence of the AR(1) log-price process.
        sigma: Innovation standard deviation of the AR(1) process.
        post_reward_mode: Reward decomposition mode.
        risk_free_rate_annual: Annualized risk-free cash return.
        transaction_cost_rate: Proportional transaction cost rate.
        seed: Optional random seed for environment sampling.
    """

    episode_length: int = 365
    initial_cash: float = 1000.0
    initial_inventory: int = 0
    max_inventory: int = 20
    max_trade_per_day: int | None = None
    initial_price: float = 50.0
    mu: float = 50.0
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
        """Initialize the trading environment.

        Args:
            config: Optional environment configuration. Defaults to
                ``TradingAr1EnvConfig()``.
        """
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
        """Return the initial log-price implied by ``initial_price``."""
        return float(np.log(self.config.initial_price))

    @property
    def _log_mu(self) -> float:
        """Return the long-run log-price implied by the configured mean price."""
        return float(np.log(self.config.mu))

    @property
    def daily_risk_free_rate(self) -> float:
        """Convert the annual risk-free rate into a daily rate."""
        return (1.0 + self.config.risk_free_rate_annual) ** (1.0 / 365.0) - 1.0

    def _validate_config(self) -> None:
        """Validate environment configuration values.

        Raises:
            ValueError: If any configuration value is inconsistent or invalid.
        """
        if self.config.episode_length <= 0:
            raise ValueError("episode_length must be positive.")
        if self.config.initial_cash <= 0.0:
            raise ValueError("initial_cash must be positive.")
        if self.config.max_inventory <= 0:
            raise ValueError("max_inventory must be positive.")
        if not 0 <= self.config.initial_inventory <= self.config.max_inventory:
            raise ValueError("initial_inventory must be within [0, max_inventory].")
        if self.config.initial_price <= 0.0:
            raise ValueError("initial_price must be positive.")
        if self.config.mu <= 0.0:
            raise ValueError("mu must be positive.")
        if self.config.max_trade_per_day is not None and self.config.max_trade_per_day < 0:
            raise ValueError("max_trade_per_day must be non-negative.")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the initial state.

        Args:
            seed: Optional seed used to reinitialize the environment RNG.
            options: Unused Gymnasium reset options.

        Returns:
            The initial observation and an info dictionary.
        """
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
        """Advance the environment by one trading step.

        Args:
            action: Encoded trade action.

        Returns:
            The next observation, total reward, termination flag, truncation
            flag, and an info dictionary.

        Raises:
            RuntimeError: If reward decomposition does not match portfolio dynamics.
        """
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
        """Return a binary validity mask for the current action space.

        Returns:
            A binary array whose entries are ``1`` for valid actions.
        """
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for action_index in range(self.action_space.n):
            trade = self._decode_action(action_index)
            mask[action_index] = int(
                self._is_valid_trade(trade, self.inventory, self.cash, self.price)
            )
        return mask

    def get_post_decision_state(self, action_index: int) -> tuple[np.ndarray, float]:
        """Compute the post-decision state without mutating the environment.

        Args:
            action_index: Encoded trade action.

        Returns:
            The post-decision observation and deterministic post-decision reward.
        """
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
        """Sample the next log-price from the AR(1) dynamics.

        Args:
            current_log_price: Current log-price state.

        Returns:
            The next sampled log-price.
        """
        epsilon = self.np_random.normal()
        log_mu = self._log_mu
        return float(
            log_mu
            + self.config.phi * (current_log_price - log_mu)
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
        """Build the Gymnasium info dictionary for the current state.

        Args:
            requested_action: Action index originally requested by the policy.
            executed_action: Action index executed after validity checks.
            invalid_action: Whether the requested action was invalid.
            action_mask: Binary validity mask for the next decision step.
            post_reward: Deterministic post-decision reward component.
            stochastic_reward: Reward contribution from the price evolution.
            transaction_cost: Transaction cost paid for the trade.
            cash_interest: Interest credited on the cash position.

        Returns:
            A dictionary with diagnostics for the transition.
        """
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
        """Compute cash after executing a trade.

        Args:
            cash: Cash balance before trading.
            trade: Signed trade quantity.
            price: Current asset price.
            transaction_cost: Transaction cost paid for the trade.

        Returns:
            The resulting cash balance.
        """
        return float(cash - trade * price - transaction_cost)

    def _compute_post_reward(self, post_cash: float, transaction_cost: float) -> float:
        """Compute the deterministic post-decision reward component.

        Args:
            post_cash: Cash balance immediately after the trade.
            transaction_cost: Transaction cost paid for the trade.

        Returns:
            The deterministic reward contribution configured for the environment.
        """
        if self.config.post_reward_mode == "none":
            return 0.0
        cash_interest = post_cash * self.daily_risk_free_rate
        if self.config.post_reward_mode == "cash_interest":
            return float(cash_interest)
        return float(-transaction_cost + cash_interest)

    def _transaction_cost(self, trade: int, price: float) -> float:
        """Compute proportional transaction costs for a trade.

        Args:
            trade: Signed trade quantity.
            price: Current asset price.

        Returns:
            The transaction cost charged for the trade.
        """
        if self.config.post_reward_mode != "cash_interest_and_tc":
            return 0.0
        return float(self.config.transaction_cost_rate * abs(trade) * price)

    def _decode_action(self, action_index: int) -> int:
        """Convert an action index into a signed trade quantity.

        Args:
            action_index: Encoded action index.

        Returns:
            The decoded signed trade.
        """
        return int(action_index - self.config.max_inventory)

    def _encode_trade(self, trade: int) -> int:
        """Convert a signed trade quantity into an action index.

        Args:
            trade: Signed trade quantity.

        Returns:
            The encoded action index.
        """
        return int(trade + self.config.max_inventory)

    def _is_valid_trade(self, trade: int, inventory: int, cash: float, price: float) -> bool:
        """Check whether a trade respects inventory and cash constraints.

        Args:
            trade: Signed trade quantity.
            inventory: Current inventory level.
            cash: Current cash balance.
            price: Current asset price.

        Returns:
            ``True`` if the trade is feasible, otherwise ``False``.
        """
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
        """Build the normalized observation vector.

        Args:
            log_price: Current log-price.
            inventory: Current inventory level.
            cash: Current cash balance.
            day: Current day index within the episode.

        Returns:
            A normalized observation vector.
        """
        day_denominator = max(self.config.episode_length - 1, 1)
        # Normalize the day index to keep observations within a stable range.
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
