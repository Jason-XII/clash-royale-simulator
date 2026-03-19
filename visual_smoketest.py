#!/usr/bin/env python3
"""Visualise a gym env game with random actions using the existing pygame renderer."""

import sys, os, random
sys.path.append("src")

import pygame
from visualize_battle import BattleVisualizer, SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, RED, PURPLE
from clash_env import ClashRoyaleEnv


class EnvVisualizer(BattleVisualizer):
    """Hijack BattleVisualizer to render a ClashRoyaleEnv game."""

    def __init__(self, env: ClashRoyaleEnv):
        # BattleVisualizer.__init__ creates its own battle – we need to
        # call it first for the pygame / font / hitbox setup, then swap in
        # the env's battle.
        super().__init__()
        self.env = env
        # Point the visualizer at the env's battle state
        self.battle = env._battle

    # Override: we don't want the parent's test setup
    def setup_test_battle(self):
        pass

    def run(self):
        print("🎮 Gym Env Visual Smoke Test")
        print("Controls:  SPACE=pause  1-5=speed  ESC=quit")

        self.paused = False
        self.speed = 1
        running = True
        done = False
        total_reward = 0.0
        steps = 0
        step_interval = 1.0 / 3.0  # seconds between steps at speed=1
        accumulator = 0.0
        last_time = pygame.time.get_ticks() / 1000.0

        while running:
            # --- pygame events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                        self.speed = event.key - pygame.K_0

            # --- step the env ---
            now = pygame.time.get_ticks() / 1000.0
            if not self.paused and not done:
                accumulator += (now - last_time) * self.speed
                while accumulator >= step_interval and not done:
                    accumulator -= step_interval
                    action = self.env.action_space.sample()
                    obs, reward, done, trunc, info = self.env.step(action)
                    total_reward += reward
                    steps += 1
                self.battle = self.env._battle
            last_time = now

            # --- draw ---
            self.screen.fill(WHITE)
            self.draw_arena()
            self.draw_towers()
            self.draw_entities()
            self.draw_ui()

            # Overlay: step count and reward
            hud = self.font.render(
                f"Step {steps}  Reward {total_reward:+.2f}", True, (0, 0, 0)
            )
            self.screen.blit(hud, (10, SCREEN_HEIGHT - 30))

            if self.paused:
                txt = self.large_font.render("PAUSED", True, RED)
                self.screen.blit(txt, txt.get_rect(center=(SCREEN_WIDTH // 2, 30)))
            if self.speed > 1:
                txt = self.font.render(f"Speed: {self.speed}x", True, PURPLE)
                self.screen.blit(txt, (10, 10))
            if done:
                w = info.get("winner", -1)
                label = {0: "AGENT WINS", 1: "OPPONENT WINS"}.get(w, "DRAW")
                txt = self.large_font.render(label, True, (200, 0, 0))
                self.screen.blit(txt, txt.get_rect(center=(SCREEN_WIDTH // 2, 60)))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        print(f"\nFinished — steps={steps}  reward={total_reward:+.2f}")


if __name__ == "__main__":
    env = ClashRoyaleEnv()
    env.reset()
    vis = EnvVisualizer(env)
    vis.run()
