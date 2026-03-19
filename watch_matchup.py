#!/usr/bin/env python3
"""Watch two trained agents play against each other.

Usage:
    python watch_matchup.py --a logbait --b ragesparky
    python watch_matchup.py --a-model models/selfplay/logbait/latest.zip --a-deck logbait \
                            --b-model models/selfplay/hogcycle/latest.zip --b-deck hogcycle
"""

import sys, os, argparse
sys.path.append("src")

import pygame
import numpy as np
from stable_baselines3 import PPO
from visualize_battle import BattleVisualizer, SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, RED, BLUE, PURPLE
from train_selfplay import SelfPlayEnv, DECKS, load_state
from clash_env import (
    ClashRoyaleEnv, _silence, _card_feature, resolve_card_name,
    SPELL_REGISTRY, Position, CARD_FEAT_DIM, NUM_HAND_CARDS,
    FLAT_DIM, SPATIAL_CHANNELS, ARENA_W, ARENA_H, MAX_HP,
    TOWER_KING_HP, TOWER_PRINCESS_HP, MAX_ELIXIR, MAX_TIME,
    NUM_POSITIONS, GRID_POSITIONS, P0_MAX_DEPLOY_Y,
    Troop, Building,
)


class MatchupVisualizer(BattleVisualizer):
    def __init__(self, env, model_a, model_b, name_a="Agent A", name_b="Agent B"):
        super().__init__()
        self.env = env
        self.model_a = model_a
        self.model_b = model_b
        self.name_a = name_a
        self.name_b = name_b
        self.battle = env._battle

    def setup_test_battle(self):
        pass

    def run(self):
        print(f"🎮 {self.name_a} vs {self.name_b}")
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
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3,
                                       pygame.K_4, pygame.K_5):
                        self.speed = event.key - pygame.K_0

            now = pygame.time.get_ticks() / 1000.0
            if not self.paused and not done:
                accumulator += (now - last_time) * self.speed
                while accumulator >= step_interval and not done:
                    accumulator -= step_interval
                    action, _ = self.model_a.predict(obs, deterministic=True)
                    obs, reward, done, trunc, info = self.env.step(int(action))
                    total_reward += reward
                    steps += 1
                self.battle = self.env._battle
            last_time = now

            # Draw
            self.screen.fill(WHITE)
            self.draw_arena()
            self.draw_towers()
            self.draw_entities()
            self.draw_ui()

            # HUD
            hud = self.font.render(
                f"Step {steps}  Reward {total_reward:+.2f}", True, (0, 0, 0)
            )
            self.screen.blit(hud, (10, SCREEN_HEIGHT - 30))

            # Deck labels
            a_label = self.font.render(f"Blue: {self.name_a}", True, BLUE)
            b_label = self.font.render(f"Red: {self.name_b}", True, RED)
            self.screen.blit(a_label, (10, SCREEN_HEIGHT - 55))
            self.screen.blit(b_label, (10, SCREEN_HEIGHT - 42))

            # Hands
            p0_hand = ", ".join(self.battle.players[0].hand)
            p1_hand = ", ".join(self.battle.players[1].hand)
            h0 = self.font.render(f"  Hand: {p0_hand}", True, BLUE)
            h1 = self.font.render(f"  Hand: {p1_hand}", True, RED)
            self.screen.blit(h0, (10, SCREEN_HEIGHT - 80))
            self.screen.blit(h1, (10, SCREEN_HEIGHT - 67))

            if self.paused:
                txt = self.large_font.render("PAUSED", True, RED)
                self.screen.blit(txt, txt.get_rect(center=(SCREEN_WIDTH // 2, 30)))
            if self.speed > 1:
                txt = self.font.render(f"Speed: {self.speed}x", True, PURPLE)
                self.screen.blit(txt, (10, 10))
            if done:
                w = info.get("winner", -1)
                label = {0: f"{self.name_a} WINS",
                         1: f"{self.name_b} WINS"}.get(w, "DRAW")
                txt = self.large_font.render(label, True, (200, 0, 0))
                self.screen.blit(txt, txt.get_rect(center=(SCREEN_WIDTH // 2, 60)))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        print(f"\nFinished — steps={steps}  reward={total_reward:+.2f}")


def find_model(deck_name):
    """Find the best available model for a deck."""
    state = load_state()
    # Try selfplay latest
    path = state["latest_models"].get(deck_name)
    if path and os.path.exists(path + ".zip"):
        return path
    # Try warmup
    path = os.path.join("models", "warmup", deck_name, f"ppo_{deck_name}_final")
    if os.path.exists(path + ".zip"):
        return path
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="ragesparky", help="Deck name for player A (blue)")
    parser.add_argument("--b", default="royalrecruits", help="Deck name for player B (red)")
    parser.add_argument("--a-model", default=None, help="Override model path for A")
    parser.add_argument("--b-model", default=None, help="Override model path for B")
    args = parser.parse_args()

    deck_a = DECKS[args.a]
    deck_b = DECKS[args.b]

    model_a_path = args.a_model or find_model(args.a)
    model_b_path = args.b_model or find_model(args.b)

    if not model_a_path:
        print(f"No model found for {args.a}")
        sys.exit(1)
    if not model_b_path:
        print(f"No model found for {args.b}")
        sys.exit(1)

    print(f"Loading {args.a}: {model_a_path}")
    print(f"Loading {args.b}: {model_b_path}")

    model_a = PPO.load(model_a_path, device="cpu")
    model_b = PPO.load(model_b_path, device="cpu")

    # Create env with model_b as opponent
    env = SelfPlayEnv(
        agent_deck=deck_a,
        opponent_pool=[(model_b_path, deck_b)],
        random_deck_pool=[deck_b],
        random_ratio=1.0,  # use random_ratio=1.0 to avoid loading model in reset
    )
    obs, _ = env.reset()

    # Override opponent to use model_b
    defs = env._loader.load_card_definitions()
    env._opp_model = model_b
    env._opp_feats = {}
    for name in deck_b:
        resolved = resolve_card_name(name, defs)
        stats = env._loader.get_card(resolved)
        env._opp_feats[name] = _card_feature(stats)
    env._opp_is_spell = {}
    for name in deck_b:
        resolved = resolve_card_name(name, defs)
        env._opp_is_spell[name] = resolved in SPELL_REGISTRY
    env.opponent_fn = env._model_opponent

    vis = MatchupVisualizer(env, model_a, model_b, name_a=args.a, name_b=args.b)
    vis.run()
