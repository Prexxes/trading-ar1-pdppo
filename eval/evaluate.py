"""Evaluation helpers for PPO and PDPPO agents."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Protocol

import numpy as np


class SupportsMaskedActionSelection(Protocol):
    """Protocol for deterministic masked-action evaluation."""

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Selects an action."""


def evaluate_agent(agent: SupportsMaskedActionSelection, env, episodes: int = 100) -> dict[str, float]:
    """Evaluates an agent over multiple deterministic episodes."""
    returns: list[float] = []
    for _ in range(episodes):
        observation, info = env.reset()
        action_mask = info["action_mask"]
        done = False
        episode_return = 0.0
        while not done:
            action, _, _ = agent.select_action(observation, action_mask, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            action_mask = info["action_mask"]
            done = terminated or truncated
            episode_return += reward
        returns.append(episode_return)

    return {
        "mean_return": float(mean(returns)),
        "std_return": float(pstdev(returns)) if len(returns) > 1 else 0.0,
    }
