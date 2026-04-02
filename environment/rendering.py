from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pygame


class NutritionRenderer:
    """Lightweight pygame renderer for the nutrition environment."""

    def __init__(self, width: int = 920, height: int = 520) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Nutrition Assistant Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.small_font = pygame.font.SysFont("consolas", 16)

    def render(
        self,
        selected_ingredients: List[str],
        nutrients: Sequence[float],
        normalized_observation: np.ndarray,
        is_balanced: bool,
        is_unhealthy: bool,
        mode: str = "human",
    ):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((245, 247, 250))

        pygame.draw.rect(self.screen, (27, 38, 59), (0, 0, self.width, 72))
        title = self.font.render("Nutrition Meal Builder", True, (240, 245, 255))
        self.screen.blit(title, (20, 20))

        status_text = (
            "Balanced"
            if is_balanced
            else "Unhealthy" if is_unhealthy else "In Progress"
        )
        status_color = (
            (46, 160, 67)
            if is_balanced
            else (204, 64, 64) if is_unhealthy else (77, 122, 255)
        )
        status_surface = self.font.render(f"Status: {status_text}", True, status_color)
        self.screen.blit(status_surface, (650, 20))

        ingredients_header = self.font.render(
            "Selected Ingredients", True, (20, 20, 20)
        )
        self.screen.blit(ingredients_header, (20, 90))

        displayed = selected_ingredients[-10:]
        for i, ingredient in enumerate(displayed):
            line = self.small_font.render(f"{i + 1}. {ingredient}", True, (30, 30, 30))
            self.screen.blit(line, (20, 125 + i * 26))

        bars_x = 340
        bars_y = 100
        bar_width = 520
        bar_height = 24

        labels = ["Protein", "Carbs", "Fats", "Vitamins", "Calories", "Count"]
        colors = [
            (70, 130, 180),
            (226, 149, 49),
            (205, 92, 92),
            (60, 179, 113),
            (128, 90, 213),
            (96, 96, 96),
        ]

        for idx, label in enumerate(labels):
            ratio = float(np.clip(normalized_observation[idx], 0.0, 1.2))
            y = bars_y + idx * 58
            pygame.draw.rect(
                self.screen,
                (218, 220, 225),
                (bars_x, y, bar_width, bar_height),
                border_radius=5,
            )
            pygame.draw.rect(
                self.screen,
                colors[idx],
                (bars_x, y, int((bar_width * ratio) / 1.2), bar_height),
                border_radius=5,
            )
            label_surface = self.small_font.render(
                f"{label}: {nutrients[idx]:.1f}", True, (20, 20, 20)
            )
            self.screen.blit(label_surface, (bars_x, y - 22))

        legend = self.small_font.render(
            "Bars are normalized to target meal limits", True, (40, 40, 40)
        )
        self.screen.blit(legend, (340, 470))

        if mode == "human":
            pygame.display.flip()
            self.clock.tick(12)
            return None

        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    def close(self) -> None:
        pygame.quit()
