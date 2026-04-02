from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO
from tensorboard.backend.event_processing import event_accumulator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.custom_env import NutritionEnv


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


def _read_best_run(best_params_path: str) -> Optional[int]:
    if not os.path.exists(best_params_path):
        return None

    with open(best_params_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    run = data.get("run")
    return int(run) if run is not None else None


def _read_tensorboard_scalar(
    run_dir: str, candidate_tags: List[str]
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    if not os.path.exists(run_dir):
        return np.array([]), np.array([]), None

    ea = event_accumulator.EventAccumulator(run_dir)
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    tag = None
    for candidate in candidate_tags:
        if candidate in available:
            tag = candidate
            break

    if tag is None:
        return np.array([]), np.array([]), None

    events = ea.Scalars(tag)
    if not events:
        return np.array([]), np.array([]), tag

    steps = np.array([ev.step for ev in events], dtype=np.float32)
    values = np.array([ev.value for ev in events], dtype=np.float32)
    return steps, values, tag


def _evaluate_sb3_best(model_cls, model_path: str, episodes: int) -> np.ndarray:
    model = model_cls.load(model_path)
    env = NutritionEnv(render_mode=None, max_steps=10)
    rewards: List[float] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    return np.array(rewards, dtype=np.float32)


def _evaluate_reinforce_best(model_path: str, episodes: int) -> np.ndarray:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    policy = ReinforcePolicy(checkpoint["obs_dim"], checkpoint["action_dim"])
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()

    env = NutritionEnv(render_mode=None, max_steps=10)
    rewards: List[float] = []

    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = policy(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            rewards.append(total_reward)

    env.close()
    return np.array(rewards, dtype=np.float32)


def _load_reinforce_entropy_curve(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        return np.array([]), np.array([])

    x_vals: List[float] = []
    y_vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_vals.append(float(row["update_index"]))
            y_vals.append(float(row["mean_policy_entropy"]))

    return np.array(x_vals, dtype=np.float32), np.array(y_vals, dtype=np.float32)


def _read_experiment_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def generate_and_save_convergence_plot() -> Optional[str]:
    dqn_rows = _read_experiment_rows(
        os.path.join(PROJECT_ROOT, "models", "dqn", "dqn_experiments.csv")
    )
    ppo_rows = _read_experiment_rows(
        os.path.join(PROJECT_ROOT, "models", "pg", "ppo_experiments.csv")
    )
    reinforce_rows = _read_experiment_rows(
        os.path.join(PROJECT_ROOT, "models", "pg", "reinforce_experiments.csv")
    )

    if not (dqn_rows and ppo_rows and reinforce_rows):
        print("[PLOT] Episode-to-converge plot skipped (missing experiment CSV logs).")
        return None

    method_rows = [
        ("DQN", dqn_rows, "#d62728"),
        ("PPO", ppo_rows, "#2ca02c"),
        ("REINFORCE", reinforce_rows, "#1f77b4"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for ax, (name, rows, color) in zip(axes, method_rows):
        runs = [int(row["run"]) for row in rows]
        episodes = [float(row["episodes_observed"]) for row in rows]
        mean_rewards = [float(row["mean_reward"]) for row in rows]

        best_idx = int(np.argmax(mean_rewards))

        ax.plot(runs, episodes, marker="o", linewidth=1.8, color=color)
        ax.scatter(
            [runs[best_idx]],
            [episodes[best_idx]],
            s=90,
            color="#ff7f0e",
            zorder=4,
            label=(
                f"Best run={runs[best_idx]}\n"
                f"episodes={int(episodes[best_idx])}\n"
                f"mean_reward={mean_rewards[best_idx]:.1f}"
            ),
        )

        ax.set_title(f"{name}: Episodes Observed by Run")
        ax.set_xlabel("Run")
        ax.set_ylabel("Episodes Observed")
        ax.set_xticks(runs)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    output_dir = os.path.join(PROJECT_ROOT, "models", "plots")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "episodes_to_converge.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"[PLOT] Saved episodes-to-converge figure at: {output_path}")
    return output_path


def generate_and_save_plots(episodes: int = 100) -> Optional[str]:
    dqn_best = os.path.join(PROJECT_ROOT, "models", "dqn", "best_model.zip")
    ppo_best = os.path.join(PROJECT_ROOT, "models", "pg", "best_ppo_model.zip")
    reinforce_best = os.path.join(
        PROJECT_ROOT, "models", "pg", "best_reinforce_model.pt"
    )

    if not (
        os.path.exists(dqn_best)
        and os.path.exists(ppo_best)
        and os.path.exists(reinforce_best)
    ):
        print(
            "[PLOT] Best model files are incomplete; skipping plot generation for now."
        )
        return None

    dqn_rewards = _evaluate_sb3_best(DQN, dqn_best, episodes)
    ppo_rewards = _evaluate_sb3_best(PPO, ppo_best, episodes)
    reinforce_rewards = _evaluate_reinforce_best(reinforce_best, episodes)

    dqn_cum = np.cumsum(dqn_rewards)
    ppo_cum = np.cumsum(ppo_rewards)
    reinforce_cum = np.cumsum(reinforce_rewards)

    dqn_best_run = _read_best_run(
        os.path.join(PROJECT_ROOT, "models", "dqn", "best_params.json")
    )
    ppo_best_run = _read_best_run(
        os.path.join(PROJECT_ROOT, "models", "pg", "best_ppo_params.json")
    )

    dqn_loss_steps = np.array([])
    dqn_loss_values = np.array([])
    dqn_loss_tag: Optional[str] = None
    if dqn_best_run is not None:
        dqn_run_dir = os.path.join(
            PROJECT_ROOT, "models", "dqn", "tb_logs", f"DQN_{dqn_best_run}"
        )
        dqn_loss_steps, dqn_loss_values, dqn_loss_tag = _read_tensorboard_scalar(
            dqn_run_dir, ["train/loss"]
        )

    ppo_entropy_steps = np.array([])
    ppo_entropy_values = np.array([])
    ppo_entropy_tag: Optional[str] = None
    if ppo_best_run is not None:
        ppo_run_dir = os.path.join(
            PROJECT_ROOT, "models", "pg", "tb_logs_ppo", f"PPO_{ppo_best_run}"
        )
        ppo_entropy_steps, ppo_entropy_values, ppo_entropy_tag = (
            _read_tensorboard_scalar(
                ppo_run_dir, ["train/entropy_loss", "train/entropy"]
            )
        )

    reinforce_entropy_csv = os.path.join(
        PROJECT_ROOT, "models", "pg", "best_reinforce_entropy.csv"
    )
    reinforce_entropy_x, reinforce_entropy_y = _load_reinforce_entropy_curve(
        reinforce_entropy_csv
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)

    ep_idx = np.arange(1, episodes + 1, dtype=np.int32)
    axes[0].plot(ep_idx, dqn_cum, label="DQN", linewidth=2.0)
    axes[0].plot(ep_idx, ppo_cum, label="PPO", linewidth=2.0)
    axes[0].plot(ep_idx, reinforce_cum, label="REINFORCE", linewidth=2.0)
    axes[0].set_title("Cumulative Rewards Over Episodes (Best Models)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    if dqn_loss_steps.size > 0 and dqn_loss_values.size > 0:
        axes[1].plot(dqn_loss_steps, dqn_loss_values, color="#d62728", linewidth=1.8)
        axes[1].set_title("DQN Objective Function Curve")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel(f"{dqn_loss_tag}")
    else:
        axes[1].text(
            0.5, 0.5, "DQN objective curve unavailable", ha="center", va="center"
        )
        axes[1].set_title("DQN Objective Function Curve")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Objective")
    axes[1].grid(True, alpha=0.25)

    plotted_any_pg = False
    if ppo_entropy_steps.size > 0 and ppo_entropy_values.size > 0:
        axes[2].plot(
            ppo_entropy_steps,
            ppo_entropy_values,
            label=f"PPO ({ppo_entropy_tag})",
            linewidth=1.8,
            color="#2ca02c",
        )
        plotted_any_pg = True

    if reinforce_entropy_x.size > 0 and reinforce_entropy_y.size > 0:
        axes[2].plot(
            reinforce_entropy_x,
            reinforce_entropy_y,
            label="REINFORCE entropy",
            linewidth=1.8,
            color="#1f77b4",
        )
        plotted_any_pg = True

    axes[2].set_title("Policy Entropy Curves for Policy Gradient Methods")
    axes[2].set_xlabel("Update Index")
    axes[2].set_ylabel("Entropy")
    axes[2].grid(True, alpha=0.25)

    if plotted_any_pg:
        axes[2].legend()
    else:
        axes[2].text(
            0.5, 0.5, "Policy entropy curves unavailable", ha="center", va="center"
        )

    output_dir = os.path.join(PROJECT_ROOT, "models", "plots")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_summary_subplots.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"[PLOT] Saved training summary figure at: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_and_save_plots(episodes=100)
    generate_and_save_convergence_plot()
