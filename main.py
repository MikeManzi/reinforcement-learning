from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO

from environment.custom_env import NutritionEnv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class BestModelInfo:
    algorithm: str
    mean_reward: float
    path: str


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


def _read_best_from_csv(csv_path: str) -> Optional[Tuple[float, Dict[str, str]]]:
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    best_row = max(rows, key=lambda r: float(r["mean_reward"]))
    return float(best_row["mean_reward"]), best_row


def discover_best_model() -> Optional[BestModelInfo]:
    candidates = []

    dqn_csv = os.path.join(PROJECT_ROOT, "models", "dqn", "dqn_experiments.csv")
    dqn_best = os.path.join(PROJECT_ROOT, "models", "dqn", "best_model.zip")
    result = _read_best_from_csv(dqn_csv)
    if result is not None and os.path.exists(dqn_best):
        candidates.append(BestModelInfo("DQN", result[0], dqn_best))

    ppo_csv = os.path.join(PROJECT_ROOT, "models", "pg", "ppo_experiments.csv")
    ppo_best = os.path.join(PROJECT_ROOT, "models", "pg", "best_ppo_model.zip")
    result = _read_best_from_csv(ppo_csv)
    if result is not None and os.path.exists(ppo_best):
        candidates.append(BestModelInfo("PPO", result[0], ppo_best))

    reinforce_csv = os.path.join(
        PROJECT_ROOT, "models", "pg", "reinforce_experiments.csv"
    )
    reinforce_best = os.path.join(
        PROJECT_ROOT, "models", "pg", "best_reinforce_model.pt"
    )
    result = _read_best_from_csv(reinforce_csv)
    if result is not None and os.path.exists(reinforce_best):
        candidates.append(BestModelInfo("REINFORCE", result[0], reinforce_best))

    if not candidates:
        return None

    return max(candidates, key=lambda c: c.mean_reward)


def run_with_sb3_model(algorithm: str, model_path: str, env: NutritionEnv) -> None:
    model = DQN.load(model_path) if algorithm == "DQN" else PPO.load(model_path)

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        env.render()
        total_reward += reward
        print(
            f"Algorithm={algorithm:<11} Action={int(action):02d} "
            f"Ingredient={info.get('selected_ingredient', 'N/A'):<10} Reward={reward:>6.2f}"
        )
        time.sleep(0.4)

    print(f"\nFinal total reward: {total_reward:.2f}")
    print(f"Final ingredient list: {info['selected_ingredients']}")


def run_with_reinforce(model_path: str, env: NutritionEnv) -> None:
    checkpoint = torch.load(model_path, map_location="cpu")
    policy = ReinforcePolicy(checkpoint["obs_dim"], checkpoint["action_dim"])
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    with torch.no_grad():
        while not (terminated or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            print(
                f"Algorithm=REINFORCE  Action={action:02d} "
                f"Ingredient={info.get('selected_ingredient', 'N/A'):<10} Reward={reward:>6.2f}"
            )
            time.sleep(0.4)

    print(f"\nFinal total reward: {total_reward:.2f}")
    print(f"Final ingredient list: {info['selected_ingredients']}")


def evaluate_sb3_model(
    algorithm: str, model_path: str, episodes: int = 20
) -> Dict[str, float]:
    model = DQN.load(model_path) if algorithm == "DQN" else PPO.load(model_path)
    env = NutritionEnv(render_mode=None, max_steps=10)

    rewards = []
    successes = 0
    unique_counts = []

    for _ in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward

        rewards.append(total_reward)
        successes += int(info.get("is_fully_balanced", False))
        unique_counts.append(float(info.get("unique_ingredient_count", 0)))

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(successes / episodes),
        "avg_unique_ingredients": float(np.mean(unique_counts)),
    }


def evaluate_reinforce_model(model_path: str, episodes: int = 20) -> Dict[str, float]:
    checkpoint = torch.load(model_path, map_location="cpu")
    policy = ReinforcePolicy(checkpoint["obs_dim"], checkpoint["action_dim"])
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()

    env = NutritionEnv(render_mode=None, max_steps=10)
    rewards = []
    successes = 0
    unique_counts = []

    with torch.no_grad():
        for _ in range(episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = policy(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

            rewards.append(total_reward)
            successes += int(info.get("is_fully_balanced", False))
            unique_counts.append(float(info.get("unique_ingredient_count", 0)))

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(successes / episodes),
        "avg_unique_ingredients": float(np.mean(unique_counts)),
    }


def main() -> None:
    best_model = discover_best_model()
    if best_model is None:
        print("No trained models found. Run training scripts first:")
        print("python training/dqn_training.py")
        print("python training/pg_training.py")
        return

    print(
        f"Selected best model: {best_model.algorithm} (mean_reward={best_model.mean_reward:.2f})"
    )

    if best_model.algorithm in {"DQN", "PPO"}:
        metrics = evaluate_sb3_model(best_model.algorithm, best_model.path, episodes=20)
    else:
        metrics = evaluate_reinforce_model(best_model.path, episodes=20)

    print("\nEvaluation over 20 episodes:")
    print(f"Mean reward: {metrics['mean_reward']:.2f}")
    print(f"Success rate: {metrics['success_rate'] * 100:.1f}%")
    print(f"Average unique ingredients: {metrics['avg_unique_ingredients']:.2f}")
    print("\nShowing one rendered rollout...")

    env = NutritionEnv(render_mode="human", max_steps=10)
    try:
        if best_model.algorithm in {"DQN", "PPO"}:
            run_with_sb3_model(best_model.algorithm, best_model.path, env)
        else:
            run_with_reinforce(best_model.path, env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
