from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NutritionEnv(gym.Env):
    """Custom nutrition environment for step-by-step meal construction."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 10) -> None:
        super().__init__()

        self.ingredients: Dict[str, Dict[str, float]] = {
            "chicken": {
                "protein": 31,
                "carbs": 0,
                "fats": 4,
                "vitamins": 8,
                "calories": 165,
            },
            "rice": {
                "protein": 3,
                "carbs": 28,
                "fats": 0.3,
                "vitamins": 2,
                "calories": 130,
            },
            "broccoli": {
                "protein": 3,
                "carbs": 7,
                "fats": 0.4,
                "vitamins": 24,
                "calories": 35,
            },
            "avocado": {
                "protein": 2,
                "carbs": 9,
                "fats": 15,
                "vitamins": 12,
                "calories": 160,
            },
            "egg": {
                "protein": 6,
                "carbs": 0.6,
                "fats": 5,
                "vitamins": 7,
                "calories": 72,
            },
            "milk": {
                "protein": 3.4,
                "carbs": 5,
                "fats": 1,
                "vitamins": 6,
                "calories": 42,
            },
            "beans": {
                "protein": 9,
                "carbs": 27,
                "fats": 0.9,
                "vitamins": 9,
                "calories": 127,
            },
            "fish": {
                "protein": 22,
                "carbs": 0,
                "fats": 12,
                "vitamins": 11,
                "calories": 206,
            },
            "bread": {
                "protein": 8,
                "carbs": 49,
                "fats": 3.2,
                "vitamins": 3,
                "calories": 265,
            },
            "cheese": {
                "protein": 25,
                "carbs": 1.3,
                "fats": 33,
                "vitamins": 5,
                "calories": 402,
            },
            "banana": {
                "protein": 1.1,
                "carbs": 23,
                "fats": 0.3,
                "vitamins": 10,
                "calories": 96,
            },
            "oats": {
                "protein": 17,
                "carbs": 66,
                "fats": 7,
                "vitamins": 4,
                "calories": 389,
            },
        }

        self.ingredient_names: List[str] = list(self.ingredients.keys())
        self.action_space = spaces.Discrete(len(self.ingredient_names))

        self.max_steps = max_steps
        self.max_values = np.array(
            [130.0, 230.0, 90.0, 100.0, 900.0, float(max_steps)], dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.2, shape=(6,), dtype=np.float32
        )

        self.ideal_macro_ratios = np.array([0.30, 0.40, 0.30], dtype=np.float32)
        self.min_calories = 550.0
        self.max_calories = 900.0

        self.render_mode = render_mode
        self.renderer = None

        self.state = np.zeros(6, dtype=np.float32)
        self.selected_ingredients: List[str] = []
        self.ingredient_counts: Dict[str, int] = {}
        self.current_step = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.state = np.zeros(6, dtype=np.float32)
        self.current_step = 0
        self.selected_ingredients = []
        self.ingredient_counts = {name: 0 for name in self.ingredient_names}

        observation = self._get_observation()
        info = self._build_info()
        return observation, info

    def step(self, action: int):
        ingredient = self.ingredient_names[action]
        nutrient = self.ingredients[ingredient]

        previous_score = self._balance_score()

        self.state[0] += nutrient["protein"]
        self.state[1] += nutrient["carbs"]
        self.state[2] += nutrient["fats"]
        self.state[3] += nutrient["vitamins"]
        self.state[4] += nutrient["calories"]
        self.state[5] += 1

        self.selected_ingredients.append(ingredient)
        self.ingredient_counts[ingredient] = (
            self.ingredient_counts.get(ingredient, 0) + 1
        )
        self.current_step += 1

        reward = 0.0
        new_score = self._balance_score()

        if new_score > previous_score:
            reward += 2.0

        if self._is_close_to_ideal():
            reward += 10.0

        # Repetition penalty to discourage ingredient collapse.
        current_count = self.ingredient_counts.get(ingredient, 0)
        if current_count == 2:
            reward -= 1.0
        elif current_count == 3:
            reward -= 3.0
        elif current_count >= 4:
            reward -= 6.0

        if self._unique_ingredient_count() >= 4:
            reward += 2.0

        if self._excessive_nutrient_detected():
            reward -= 5.0

        if self.state[5] > 8:
            reward -= 2.0

        terminated = False
        truncated = False

        if self._max_ingredient_usage() > 2:
            reward -= 10.0
            terminated = True

        if self._is_fully_balanced():
            reward += 50.0
            terminated = True

        if self.state[4] > self.max_calories:
            reward -= 10.0
            terminated = True

        if self.current_step >= self.max_steps and not terminated:
            truncated = True

        observation = self._get_observation()
        info = self._build_info()
        info["selected_ingredient"] = ingredient

        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        if self.renderer is None:
            from .rendering import NutritionRenderer

            self.renderer = NutritionRenderer(width=920, height=520)

        return self.renderer.render(
            selected_ingredients=self.selected_ingredients,
            nutrients=self.state.copy(),
            normalized_observation=self._get_observation(),
            is_balanced=self._is_fully_balanced(),
            is_unhealthy=self._excessive_nutrient_detected()
            or self.state[4] > self.max_calories,
            mode=self.render_mode,
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _get_observation(self) -> np.ndarray:
        normalized = np.divide(
            self.state,
            self.max_values,
            out=np.zeros_like(self.state),
            where=self.max_values > 0,
        )
        return normalized.astype(np.float32)

    def _macro_ratios(self) -> np.ndarray:
        # Convert macro grams to calories to compute realistic macro ratios.
        protein_cal = self.state[0] * 4.0
        carbs_cal = self.state[1] * 4.0
        fats_cal = self.state[2] * 9.0
        total_macro_cal = protein_cal + carbs_cal + fats_cal

        if total_macro_cal <= 0:
            return np.zeros(3, dtype=np.float32)

        return np.array(
            [
                protein_cal / total_macro_cal,
                carbs_cal / total_macro_cal,
                fats_cal / total_macro_cal,
            ],
            dtype=np.float32,
        )

    def _balance_score(self) -> float:
        ratios = self._macro_ratios()
        macro_distance = float(np.abs(ratios - self.ideal_macro_ratios).sum())

        calories = float(self.state[4])
        if calories < self.min_calories:
            calorie_penalty = (self.min_calories - calories) / self.min_calories
        elif calories > self.max_calories:
            calorie_penalty = (calories - self.max_calories) / self.max_calories
        else:
            calorie_penalty = 0.0

        vitamins_target = 0.65
        vitamin_level = float(self._get_observation()[3])
        vitamin_penalty = abs(vitamins_target - min(vitamin_level, 1.0))

        return -(macro_distance + calorie_penalty + vitamin_penalty)

    def _is_close_to_ideal(self) -> bool:
        ratios = self._macro_ratios()
        calories = float(self.state[4])
        return (
            float(np.max(np.abs(ratios - self.ideal_macro_ratios))) < 0.08
            and self.min_calories <= calories <= self.max_calories
            and self._get_observation()[3] >= 0.55
        )

    def _is_fully_balanced(self) -> bool:
        ratios = self._macro_ratios()
        calories = float(self.state[4])
        vitamins = float(self._get_observation()[3])
        ingredient_count = int(self.state[5])
        unique_count = self._unique_ingredient_count()
        return (
            float(np.max(np.abs(ratios - self.ideal_macro_ratios))) < 0.05
            and 600.0 <= calories <= 820.0
            and 0.6 <= vitamins <= 1.2
            and 4 <= ingredient_count <= 8
            and unique_count >= 4
            and self._max_ingredient_usage() <= 2
        )

    def _excessive_nutrient_detected(self) -> bool:
        normalized = self._get_observation()
        ratios = self._macro_ratios()
        return (
            normalized[0] > 1.0
            or normalized[1] > 1.0
            or normalized[2] > 1.0
            or normalized[3] > 1.05
            or float(np.max(ratios)) > 0.65
        )

    def _build_info(self) -> dict:
        return {
            "state_raw": self.state.copy(),
            "macro_ratios": self._macro_ratios(),
            "selected_ingredients": list(self.selected_ingredients),
            "ingredient_counts": dict(self.ingredient_counts),
            "unique_ingredient_count": self._unique_ingredient_count(),
            "max_ingredient_usage": self._max_ingredient_usage(),
            "is_close_to_ideal": self._is_close_to_ideal(),
            "is_fully_balanced": self._is_fully_balanced(),
        }

    def _unique_ingredient_count(self) -> int:
        return sum(1 for count in self.ingredient_counts.values() if count > 0)

    def _max_ingredient_usage(self) -> int:
        if not self.ingredient_counts:
            return 0
        return max(self.ingredient_counts.values())


if __name__ == "__main__":
    env = NutritionEnv(render_mode=None)
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    print("Episode finished. Total reward:", total_reward)
    print("Ingredients:", info["selected_ingredients"])
    env.close()
