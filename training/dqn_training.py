from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.custom_env import NutritionEnv


def try_generate_plots() -> None:
    try:
        from training.generate_training_plots import (
            generate_and_save_convergence_plot,
            generate_and_save_plots,
        )

        summary_output = generate_and_save_plots(episodes=100)
        convergence_output = generate_and_save_convergence_plot()

        if summary_output is None:
            print(
                "[DQN] Summary plot generation skipped (required best-model artifacts are not all available yet)."
            )
        else:
            print(f"[DQN] Summary plot image ready at: {summary_output}")

        if convergence_output is None:
            print(
                "[DQN] Episodes-to-converge plot generation skipped (required experiment logs are not all available yet)."
            )
        else:
            print(f"[DQN] Episodes-to-converge plot image ready at: {convergence_output}")
    except Exception as exc:
        print(f"[DQN] Plot generation failed: {exc}")


def get_tensorboard_log_dir() -> str | None:
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("[DQN] TensorBoard not installed; continuing without tensorboard logs.")
        return None

    return os.path.join(PROJECT_ROOT, "models", "dqn", "tb_logs")


@dataclass
class ExperimentConfig:
    learning_rate: float
    gamma: float
    batch_size: int
    exploration_fraction: float
    exploration_final_eps: float


class EpisodeRewardCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info and "r" in info["episode"]:
                self.episode_rewards.append(float(info["episode"]["r"]))
        return True


def make_env() -> NutritionEnv:
    env = NutritionEnv(render_mode=None, max_steps=10)
    return Monitor(env)


def classify_convergence(episode_rewards: List[float]) -> str:
    if len(episode_rewards) < 8:
        return "insufficient-data"

    window = np.array(
        episode_rewards[-min(30, len(episode_rewards)) :], dtype=np.float32
    )
    x = np.arange(len(window), dtype=np.float32)
    slope = float(np.polyfit(x, window, 1)[0]) if len(window) >= 2 else 0.0
    variance = float(np.std(window))

    if slope > 0.05 and variance < 25.0:
        return "fast-improving"
    if slope >= 0.0 and variance < 40.0:
        return "stable"
    return "noisy-or-stagnant"


def get_experiment_grid() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(1e-3, 0.98, 32, 0.30, 0.05),
        ExperimentConfig(7e-4, 0.99, 32, 0.35, 0.03),
        ExperimentConfig(5e-4, 0.97, 64, 0.25, 0.02),
        ExperimentConfig(3e-4, 0.99, 64, 0.20, 0.01),
        ExperimentConfig(1e-4, 0.995, 64, 0.40, 0.05),
        ExperimentConfig(8e-4, 0.96, 32, 0.30, 0.02),
        ExperimentConfig(2e-4, 0.98, 128, 0.15, 0.01),
        ExperimentConfig(4e-4, 0.95, 128, 0.45, 0.06),
        ExperimentConfig(6e-4, 0.97, 64, 0.25, 0.04),
        ExperimentConfig(9e-4, 0.99, 32, 0.35, 0.01),
    ]


def run_dqn_hyperparameter_search(
    total_timesteps: int = 15000, eval_episodes: int = 6
) -> None:
    os.makedirs(os.path.join(PROJECT_ROOT, "models", "dqn"), exist_ok=True)
    tb_log_dir = get_tensorboard_log_dir()

    results: List[Dict[str, float]] = []
    best_score = -np.inf
    best_run_path = ""
    best_params: Dict[str, float] = {}

    experiment_grid = get_experiment_grid()

    for run_idx, config in enumerate(experiment_grid, start=1):
        print(f"\n[DQN] Run {run_idx}/{len(experiment_grid)} with {config}")
        env = DummyVecEnv([make_env])
        callback = EpisodeRewardCallback()

        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            batch_size=config.batch_size,
            buffer_size=50000,
            learning_starts=500,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=250,
            exploration_fraction=config.exploration_fraction,
            exploration_final_eps=config.exploration_final_eps,
            verbose=0,
            tensorboard_log=tb_log_dir,
        )

        model.learn(
            total_timesteps=total_timesteps, callback=callback, progress_bar=False
        )
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=eval_episodes, deterministic=True
        )

        run_path = os.path.join(PROJECT_ROOT, "models", "dqn", f"dqn_run_{run_idx}.zip")
        model.save(run_path)

        convergence = classify_convergence(callback.episode_rewards)
        final_episode_reward = (
            float(callback.episode_rewards[-1])
            if callback.episode_rewards
            else float("nan")
        )

        row = {
            "run": run_idx,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "batch_size": config.batch_size,
            "exploration_fraction": config.exploration_fraction,
            "exploration_final_eps": config.exploration_final_eps,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "final_episode_reward": final_episode_reward,
            "episodes_observed": len(callback.episode_rewards),
            "convergence": convergence,
        }
        results.append(row)

        print(
            f"[DQN] Run {run_idx} -> mean_reward={mean_reward:.2f}, "
            f"std={std_reward:.2f}, convergence={convergence}"
        )

        if mean_reward > best_score:
            best_score = float(mean_reward)
            best_run_path = run_path
            best_params = {
                "run": run_idx,
                "learning_rate": config.learning_rate,
                "gamma": config.gamma,
                "batch_size": config.batch_size,
                "exploration_fraction": config.exploration_fraction,
                "exploration_final_eps": config.exploration_final_eps,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
            }

        env.close()

    csv_path = os.path.join(PROJECT_ROOT, "models", "dqn", "dqn_experiments.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    if best_run_path:
        best_model = DQN.load(best_run_path)
        best_model.save(os.path.join(PROJECT_ROOT, "models", "dqn", "best_model.zip"))

    with open(
        os.path.join(PROJECT_ROOT, "models", "dqn", "best_params.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best_params, f, indent=2)

    print("\n[DQN] Hyperparameter search completed.")
    print(f"Best mean reward: {best_score:.2f}")
    print(f"Saved CSV logs at: {csv_path}")


if __name__ == "__main__":
    run_dqn_hyperparameter_search()
    try_generate_plots()
