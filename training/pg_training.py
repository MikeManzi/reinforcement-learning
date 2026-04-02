from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
from stable_baselines3 import PPO
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
                "[PG] Summary plot generation skipped (required best-model artifacts are not all available yet)."
            )
        else:
            print(f"[PG] Summary plot image ready at: {summary_output}")

        if convergence_output is None:
            print(
                "[PG] Episodes-to-converge plot generation skipped (required experiment logs are not all available yet)."
            )
        else:
            print(f"[PG] Episodes-to-converge plot image ready at: {convergence_output}")
    except Exception as exc:
        print(f"[PG] Plot generation failed: {exc}")


def get_tensorboard_log_dir() -> str | None:
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("[PG] TensorBoard not installed; continuing without tensorboard logs.")
        return None

    return os.path.join(PROJECT_ROOT, "models", "pg", "tb_logs_ppo")


@dataclass
class PPOConfig:
    learning_rate: float
    gamma: float
    batch_size: int


@dataclass
class ReinforceConfig:
    learning_rate: float
    gamma: float
    batch_episodes: int


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


class ReinforcePolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    variability = float(np.std(window))

    if slope > 0.05 and variability < 25.0:
        return "fast-improving"
    if slope >= 0.0 and variability < 40.0:
        return "stable"
    return "noisy-or-stagnant"


def ppo_grid() -> List[PPOConfig]:
    return [
        PPOConfig(3e-4, 0.99, 64),
        PPOConfig(1e-4, 0.98, 64),
        PPOConfig(5e-4, 0.97, 32),
        PPOConfig(7e-4, 0.99, 32),
        PPOConfig(2e-4, 0.995, 128),
        PPOConfig(4e-4, 0.96, 64),
        PPOConfig(8e-4, 0.95, 32),
        PPOConfig(6e-4, 0.98, 128),
        PPOConfig(9e-4, 0.97, 64),
        PPOConfig(3e-4, 0.94, 32),
    ]


def reinforce_grid() -> List[ReinforceConfig]:
    return [
        ReinforceConfig(1e-3, 0.99, 6),
        ReinforceConfig(8e-4, 0.98, 8),
        ReinforceConfig(7e-4, 0.97, 6),
        ReinforceConfig(5e-4, 0.99, 10),
        ReinforceConfig(3e-4, 0.96, 8),
        ReinforceConfig(9e-4, 0.95, 12),
        ReinforceConfig(6e-4, 0.98, 10),
        ReinforceConfig(4e-4, 0.97, 12),
        ReinforceConfig(2e-4, 0.99, 8),
        ReinforceConfig(1.2e-3, 0.94, 6),
    ]


def run_ppo_search(
    total_timesteps: int = 15000, eval_episodes: int = 6
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    best_score = -np.inf
    best_model_path = ""
    best_params: Dict[str, float] = {}
    tb_log_dir = get_tensorboard_log_dir()

    for run_idx, config in enumerate(ppo_grid(), start=1):
        print(f"\n[PPO] Run {run_idx}/10 with {config}")
        env = DummyVecEnv([make_env])
        callback = EpisodeRewardCallback()

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            batch_size=config.batch_size,
            n_steps=256,
            n_epochs=10,
            ent_coef=0.01,
            clip_range=0.2,
            verbose=0,
            tensorboard_log=tb_log_dir,
        )

        model.learn(
            total_timesteps=total_timesteps, callback=callback, progress_bar=False
        )
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=eval_episodes, deterministic=True
        )

        run_path = os.path.join(PROJECT_ROOT, "models", "pg", f"ppo_run_{run_idx}.zip")
        model.save(run_path)

        convergence = classify_convergence(callback.episode_rewards)
        final_episode_reward = (
            float(callback.episode_rewards[-1])
            if callback.episode_rewards
            else float("nan")
        )

        row = {
            "algorithm": "PPO",
            "run": run_idx,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "batch_size": config.batch_size,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "final_episode_reward": final_episode_reward,
            "episodes_observed": len(callback.episode_rewards),
            "convergence": convergence,
        }
        results.append(row)

        print(
            f"[PPO] Run {run_idx} -> mean_reward={mean_reward:.2f}, "
            f"std={std_reward:.2f}, convergence={convergence}"
        )

        if mean_reward > best_score:
            best_score = float(mean_reward)
            best_model_path = run_path
            best_params = {
                "run": run_idx,
                "learning_rate": config.learning_rate,
                "gamma": config.gamma,
                "batch_size": config.batch_size,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
            }

        env.close()

    if best_model_path:
        best_model = PPO.load(best_model_path)
        best_model.save(
            os.path.join(PROJECT_ROOT, "models", "pg", "best_ppo_model.zip")
        )

    with open(
        os.path.join(PROJECT_ROOT, "models", "pg", "best_ppo_params.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best_params, f, indent=2)

    return results


def _discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    returns = []
    g = 0.0
    for reward in reversed(rewards):
        g = reward + gamma * g
        returns.insert(0, g)

    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    if returns_tensor.numel() > 1:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (
            returns_tensor.std() + 1e-8
        )
    return returns_tensor


def evaluate_reinforce_model(
    policy: ReinforcePolicy, episodes: int = 8
) -> Tuple[float, float]:
    env = NutritionEnv(render_mode=None, max_steps=10)
    totals = []

    policy.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0.0

            while not (done or truncated):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = policy(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward

            totals.append(total_reward)

    env.close()
    return float(np.mean(totals)), float(np.std(totals))


def run_reinforce_search(episodes_per_run: int = 220) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    best_score = -np.inf
    best_state = None
    best_params: Dict[str, float] = {}
    best_entropy_curve: List[float] = []

    for run_idx, config in enumerate(reinforce_grid(), start=1):
        print(f"\n[REINFORCE] Run {run_idx}/10 with {config}")

        env = NutritionEnv(render_mode=None, max_steps=10)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        policy = ReinforcePolicy(obs_dim, action_dim)
        optimizer = Adam(policy.parameters(), lr=config.learning_rate)

        episode_rewards: List[float] = []
        update_entropies: List[float] = []

        for episode in range(episodes_per_run):
            batch_log_probs: List[torch.Tensor] = []
            batch_returns: List[torch.Tensor] = []
            batch_entropies: List[float] = []

            for _ in range(config.batch_episodes):
                obs, _ = env.reset()
                done = False
                truncated = False

                log_probs: List[torch.Tensor] = []
                rewards: List[float] = []

                while not (done or truncated):
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    logits = policy(obs_t)
                    dist = Categorical(logits=logits)
                    action = dist.sample()

                    obs, reward, done, truncated, _ = env.step(int(action.item()))
                    log_probs.append(dist.log_prob(action))
                    batch_entropies.append(float(dist.entropy().item()))
                    rewards.append(float(reward))

                returns_t = _discounted_returns(rewards, config.gamma)
                batch_log_probs.extend(log_probs)
                batch_returns.extend([ret for ret in returns_t])
                episode_rewards.append(float(np.sum(rewards)))

            if batch_log_probs:
                loss = 0.0
                for lp, ret in zip(batch_log_probs, batch_returns):
                    loss = loss + (-lp * ret)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_entropy = float(np.mean(batch_entropies)) if batch_entropies else 0.0
            update_entropies.append(mean_entropy)

            if (episode + 1) % 40 == 0:
                recent_mean = (
                    float(np.mean(episode_rewards[-40:])) if episode_rewards else 0.0
                )
                print(
                    f"[REINFORCE] Run {run_idx}, update {episode + 1}/{episodes_per_run}, recent_mean={recent_mean:.2f}"
                )

        mean_reward, std_reward = evaluate_reinforce_model(policy)

        run_path = os.path.join(
            PROJECT_ROOT, "models", "pg", f"reinforce_run_{run_idx}.pt"
        )
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "config": {
                    "learning_rate": config.learning_rate,
                    "gamma": config.gamma,
                    "batch_episodes": config.batch_episodes,
                },
            },
            run_path,
        )

        entropy_curve_path = os.path.join(
            PROJECT_ROOT, "models", "pg", f"reinforce_run_{run_idx}_entropy.csv"
        )
        with open(entropy_curve_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["update_index", "mean_policy_entropy"]
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "update_index": idx + 1,
                        "mean_policy_entropy": value,
                    }
                    for idx, value in enumerate(update_entropies)
                ]
            )

        convergence = classify_convergence(episode_rewards)
        final_episode_reward = (
            float(episode_rewards[-1]) if episode_rewards else float("nan")
        )

        row = {
            "algorithm": "REINFORCE",
            "run": run_idx,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "batch_size": config.batch_episodes,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "final_episode_reward": final_episode_reward,
            "episodes_observed": len(episode_rewards),
            "convergence": convergence,
        }
        results.append(row)

        print(
            f"[REINFORCE] Run {run_idx} -> mean_reward={mean_reward:.2f}, "
            f"std={std_reward:.2f}, convergence={convergence}"
        )

        if mean_reward > best_score:
            best_score = float(mean_reward)
            best_entropy_curve = list(update_entropies)
            best_state = {
                "state_dict": policy.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "config": {
                    "learning_rate": config.learning_rate,
                    "gamma": config.gamma,
                    "batch_episodes": config.batch_episodes,
                },
            }
            best_params = {
                "run": run_idx,
                "learning_rate": config.learning_rate,
                "gamma": config.gamma,
                "batch_size": config.batch_episodes,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
            }

        env.close()

    if best_state is not None:
        torch.save(
            best_state,
            os.path.join(PROJECT_ROOT, "models", "pg", "best_reinforce_model.pt"),
        )

    with open(
        os.path.join(PROJECT_ROOT, "models", "pg", "best_reinforce_params.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best_params, f, indent=2)

    best_entropy_path = os.path.join(
        PROJECT_ROOT, "models", "pg", "best_reinforce_entropy.csv"
    )
    with open(best_entropy_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["update_index", "mean_policy_entropy"])
        writer.writeheader()
        writer.writerows(
            [
                {
                    "update_index": idx + 1,
                    "mean_policy_entropy": value,
                }
                for idx, value in enumerate(best_entropy_curve)
            ]
        )

    return results


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_policy_gradient_training() -> None:
    os.makedirs(os.path.join(PROJECT_ROOT, "models", "pg"), exist_ok=True)

    ppo_results = run_ppo_search(total_timesteps=15000, eval_episodes=6)
    reinforce_results = run_reinforce_search(episodes_per_run=220)

    write_csv(
        os.path.join(PROJECT_ROOT, "models", "pg", "ppo_experiments.csv"), ppo_results
    )
    write_csv(
        os.path.join(PROJECT_ROOT, "models", "pg", "reinforce_experiments.csv"),
        reinforce_results,
    )

    combined = ppo_results + reinforce_results
    write_csv(
        os.path.join(PROJECT_ROOT, "models", "pg", "pg_experiments.csv"), combined
    )

    print("\n[PG] Training and hyperparameter search complete.")
    print("Saved logs in models/pg/")


if __name__ == "__main__":
    run_policy_gradient_training()
    try_generate_plots()
