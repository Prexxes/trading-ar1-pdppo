"""Training script for PPO on the AR(1) trading environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.ppo_agent import PPOAgent, PPOConfig
from envs.trading_ar1_env import TradingAr1Env, TradingAr1EnvConfig
from eval.evaluate import evaluate_agent
from utils.config import TrainingPaths
from utils.logging_utils import append_csv_row, ensure_directory
from utils.seeding import seed_everything


def create_training_paths(root: Path) -> TrainingPaths:
    """Creates directories for a training run."""
    checkpoint_dir = ensure_directory(root / "checkpoints")
    return TrainingPaths(
        root=ensure_directory(root),
        checkpoint_dir=checkpoint_dir,
        log_csv=root / "training_log.csv",
    )


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
    """Runs PPO training, evaluation, logging, and checkpointing."""
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
            evaluation = evaluate_agent(agent, eval_env, episodes=eval_episodes)
            metrics["eval_return_mean"] = evaluation["mean_return"]
            metrics["eval_return_std"] = evaluation["std_return"]

            if evaluation["mean_return"] > best_eval_return:
                best_eval_return = evaluation["mean_return"]
                agent.save(paths.checkpoint_dir / "best.pt")

        agent.save(paths.checkpoint_dir / "last.pt")
        append_csv_row(paths.log_csv, fieldnames, metrics)
        final_metrics = metrics

    return final_metrics


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
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
    """Runs PPO training from the command line."""
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
