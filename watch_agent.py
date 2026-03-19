#!/usr/bin/env python3
"""Watch a trained PPO agent play against the random opponent."""

import sys, os
sys.path.append("src")

import pygame
from stable_baselines3 import PPO
from visualize_battle import BattleVisualizer, SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, RED, PURPLE, BLUE
from clash_env import ClashRoyaleEnv


class AgentVisualizer(BattleVisualizer):
    def __init__(self, env, model):
        super().__init__()
        self.env = env
        self.model = model
        self.battle = env._battle

    def setup_test_battle(self):
        pass

    def run(self):
        print("🎮 Watching trained agent")
        print("Controls:  SPACE=pause  1-5=speed  R=reset  ESC=quit")

        self.paused = False
        self.speed = 1
        running = True
        done = False
        total_reward = 0.0
        steps = 0
        obs, _ = self.env.reset()
        self.battle = self.env._battle

        step_interval = 1.0 / 3.0
        accumulator = 0.0
        last_time = pygame.time.get_ticks() / 1000.0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        obs, _ = self.env.reset()
                        self.battle = self.env._battle
                        done = False
                        total_reward = 0.0
                        steps = 0
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                        self.speed = event.key - pygame.K_0

            now = pygame.time.get_ticks() / 1000.0
            if not self.paused and not done:
                accumulator += (now - last_time) * self.speed
                while accumulator >= step_interval and not done:
                    accumulator -= step_interval
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, trunc, info = self.env.step(int(action))
                    total_reward += reward
                    steps += 1
                self.battle = self.env._battle
            last_time = now

            self.screen.fill(WHITE)
            self.draw_arena()
            self.draw_towers()
            self.draw_entities()
            self.draw_ui()

            hud = self.font.render(
                f"Step {steps}  Reward {total_reward:+.2f}", True, (0, 0, 0)
            )
            self.screen.blit(hud, (10, SCREEN_HEIGHT - 30))
            p0_hand = ", ".join(self.battle.players[0].hand)
            p1_hand = ", ".join(self.battle.players[1].hand)
            h0 = self.font.render(f"Agent: {p0_hand}", True, BLUE)
            h1 = self.font.render(f"Opp:   {p1_hand}", True, RED)
            self.screen.blit(h0, (10, SCREEN_HEIGHT - 55))
            self.screen.blit(h1, (10, SCREEN_HEIGHT - 42))
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
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/warmup/ragesparky/ppo_ragesparky_final"
    env = ClashRoyaleEnv(agent_deck=["Giant", "ZapMachine", "Rage", "DarkPrince",
                                  "MiniPekka", "Zap", "Wizard", "MinionHorde"])
    env.reset()
    model = PPO.load(model_path)
    vis = AgentVisualizer(env, model)
    vis.run()