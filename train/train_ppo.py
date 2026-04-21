"""Training script for PPO on the AR(1) trading environment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from agents.ppo_agent import PPOAgent, PPOConfig
from envs.trading_ar1_env import TradingAr1Env, TradingAr1EnvConfig
from eval.evaluate import evaluate_agent
from utils.config import TrainingPaths
from utils.logging_utils import append_csv_row, ensure_directory, write_json
from utils.seeding import seed_everything


def create_training_paths(root: Path) -> TrainingPaths:
    """Create directories and file paths for a training run.

    Args:
        root: Root directory of the training run.

    Returns:
        A ``TrainingPaths`` object with resolved locations.
    """
    checkpoint_dir = ensure_directory(root / "checkpoints")
    return TrainingPaths(
        root=ensure_directory(root),
        checkpoint_dir=checkpoint_dir,
        log_csv=root / "training_log.csv",
        metadata_json=root / "run_metadata.json",
    )


def build_run_metadata(
    *,
    algorithm: str,
    started_at: datetime,
    finished_at: datetime,
    duration_seconds: float,
    env_config: TradingAr1EnvConfig,
    agent_config: Any,
    total_updates: int,
    eval_interval: int,
    eval_episodes: int,
    seed: int,
) -> dict[str, object]:
    """Build human-readable metadata for a completed training run.

    Args:
        algorithm: Name of the training algorithm.
        started_at: Local timestamp captured when the run started.
        finished_at: Local timestamp captured when the run finished.
        duration_seconds: Total wall-clock duration of the run.
        env_config: Environment configuration used for the run.
        agent_config: Agent configuration used for the run.
        total_updates: Number of rollout/update iterations.
        eval_interval: Frequency of evaluation in updates.
        eval_episodes: Number of evaluation episodes per evaluation phase.
        seed: Global random seed.

    Returns:
        A JSON-serializable metadata dictionary.
    """
    return {
        "algorithm": algorithm,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration": {
            "seconds": duration_seconds,
        },
        "training_config": {
            "total_updates": total_updates,
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
            "seed": seed,
        },
        "environment_config": asdict(env_config),
        "agent_config": asdict(agent_config),
    }


def train_ppo(
    env_config: TradingAr1EnvConfig,
    agent_config: PPOConfig,
    *,
    total_updates: int,
    eval_interval: int,
    eval_episodes: int,
    output_dir: Path,
    seed: int,
) -> dict[str, float]:
    """Run PPO training, evaluation, logging, and checkpointing.

    Args:
        env_config: Environment configuration used for training and evaluation.
        agent_config: PPO hyperparameters and model configuration.
        total_updates: Number of rollout/update iterations to run.
        eval_interval: Frequency of evaluation in updates.
        eval_episodes: Number of evaluation episodes per evaluation phase.
        output_dir: Directory used for logs and checkpoints.
        seed: Global random seed.

    Returns:
        Metrics collected during the final training update.
    """
    started_at = datetime.now().astimezone()
    started_time = perf_counter()
    seed_everything(seed)
    train_env = TradingAr1Env(env_config)
    eval_env = TradingAr1Env(env_config)
    agent = PPOAgent(agent_config)
    paths = create_training_paths(output_dir)
    best_eval_return = float("-inf")
    final_metrics: dict[str, float] = {}
    fieldnames = [
        "update",
        "environment_steps",
        "train_return_mean",
        "eval_return_mean",
        "eval_return_std",
        "actor_loss",
        "critic_loss",
    ]

    for update in range(1, total_updates + 1):
        buffer, episode_returns = agent.collect_rollout(train_env)
        losses = agent.update(buffer)
        environment_steps = update * agent_config.rollout_steps
        metrics = {
            "update": update,
            "environment_steps": environment_steps,
            "train_return_mean": float(sum(episode_returns) / len(episode_returns))
            if episode_returns
            else 0.0,
            "eval_return_mean": float("nan"),
            "eval_return_std": float("nan"),
            "actor_loss": losses["actor_loss"],
            "critic_loss": losses["critic_loss"],
        }

        if update % eval_interval == 0 or update == total_updates:
            # Evaluate periodically and keep the best-performing checkpoint.
            evaluation = evaluate_agent(agent, eval_env, episodes=eval_episodes)
            metrics["eval_return_mean"] = evaluation["mean_return"]
            metrics["eval_return_std"] = evaluation["std_return"]

            if evaluation["mean_return"] > best_eval_return:
                best_eval_return = evaluation["mean_return"]
                agent.save(paths.checkpoint_dir / "best.pt")

        agent.save(paths.checkpoint_dir / "last.pt")
        append_csv_row(paths.log_csv, fieldnames, metrics)
        final_metrics = metrics

    finished_at = datetime.now().astimezone()
    metadata = build_run_metadata(
        algorithm="PPO",
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=perf_counter() - started_time,
        env_config=env_config,
        agent_config=agent_config,
        total_updates=total_updates,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        seed=seed,
    )
    write_json(paths.metadata_json, metadata)

    return final_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for PPO training.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-updates", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    """Run PPO training from the command line."""
    args = parse_args()
    env_config = TradingAr1EnvConfig(seed=args.seed)
    agent_config = PPOConfig(
        obs_dim=4,
        action_dim=2 * env_config.max_inventory + 1,
        rollout_steps=args.rollout_steps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
    )
    train_ppo(
        env_config,
        agent_config,
        total_updates=args.total_updates,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
