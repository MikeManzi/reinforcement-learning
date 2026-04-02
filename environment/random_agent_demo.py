from __future__ import annotations

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.custom_env import NutritionEnv


def run_random_demo(episodes: int = 3, step_delay: float = 0.5) -> None:
    env = NutritionEnv(render_mode="human", max_steps=10)

    for ep in range(1, episodes + 1):
        observation, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        print(f"\nEpisode {ep}")

        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            chosen = info.get("selected_ingredient", "N/A")
            print(
                f"Action={action:02d} Ingredient={chosen:<10} "
                f"Reward={reward:>6.2f} Calories={info['state_raw'][4]:>6.1f}"
            )
            time.sleep(step_delay)

        print(f"Total reward: {total_reward:.2f}")
        print(f"Final ingredients: {info['selected_ingredients']}")

    env.close()


if __name__ == "__main__":
    run_random_demo()
