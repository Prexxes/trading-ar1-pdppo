"""Training script for PDPPO on the AR(1) trading environment."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

from agents.pdppo_agent import PDPPOAgent, PDPPOConfig
from envs.trading_ar1_env import TradingAr1Env, TradingAr1EnvConfig
from eval.evaluate import evaluate_agent
from train.train_ppo import build_run_metadata, create_training_paths
from utils.logging_utils import append_csv_row, write_json
from utils.seeding import seed_everything


def train_pdppo(
    env_config: TradingAr1EnvConfig,
    agent_config: PDPPOConfig,
    *,
    total_updates: int,
    eval_interval: int,
    eval_episodes: int,
    output_dir: Path,
    seed: int,
) -> dict[str, float]:
    """Run PDPPO training, evaluation, logging, and checkpointing.

    Args:
        env_config: Environment configuration used for training and evaluation.
        agent_config: PDPPO hyperparameters and model configuration.
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
    agent = PDPPOAgent(agent_config)
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
        "post_critic_loss",
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
            "post_critic_loss": losses["post_critic_loss"],
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
        algorithm="PDPPO",
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
    """Parse command-line arguments for PDPPO training.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pdppo"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-updates", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-state-lr", type=float, default=1e-3)
    parser.add_argument("--critic-post-lr", type=float, default=1e-3)
    parser.add_argument(
        "--post-reward-mode",
        type=str,
        default="cash_interest_and_tc",
        choices=["none", "cash_interest", "cash_interest_and_tc"],
    )
    return parser.parse_args()


def main() -> None:
    """Run PDPPO training from the command line."""
    args = parse_args()
    env_config = TradingAr1EnvConfig(
        seed=args.seed,
        post_reward_mode=args.post_reward_mode,
    )
    agent_config = PDPPOConfig(
        obs_dim=4,
        action_dim=2 * env_config.max_inventory + 1,
        rollout_steps=args.rollout_steps,
        actor_lr=args.actor_lr,
        critic_state_lr=args.critic_state_lr,
        critic_post_lr=args.critic_post_lr,
    )
    train_pdppo(
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
