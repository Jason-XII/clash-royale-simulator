"""Play the simulator yourself against a trained model.

    python play_vs_model.py memory0.zip [seed]

You are blue (bottom), the model is red. Click a card (or press 1-4), then a
tile. Your cards land at once; the model's land after its live delay, as in
training. Space pauses, Esc quits.

Every deployment of both players is saved with its tick, together with the seed
and starting decks, to captures/sim_<time>.json. The simulator is deterministic,
so `replay` reproduces a game exactly (e.g. to build imitation data later).
"""
import json
from pathlib import Path
import sys
import time

import pygame

from card_utils import Card
from environment import CREnv, LIVE_PLAY_DELAY, to_world
from new_visualization import AH, AW, AX, AY, BLACK, BLUE, RED, TILE, W, H, WHITE, Visualizer
from train_core import FrozenPolicy

FPS = 20
TICKS_PER_DECISION = 10      # the model decides every half second, as in training
HAND_Y = AY + AH + 30


class Game(Visualizer):
    def __init__(self, checkpoint, seed):
        self.model = FrozenPolicy(checkpoint)
        self.env = CREnv(opponent_model=self.model, play_delay=LIVE_PLAY_DELAY)
        self.model.bind_env(self.env)
        self.env.reset(seed=seed)
        super().__init__(self.env.battle)
        self.screen = pygame.display.set_mode((W, H + 70))
        pygame.display.set_caption(f"You (blue) vs {Path(checkpoint).name} (red)")
        self.selected = None
        self.message = ""
        self.record = {"checkpoint": str(checkpoint), "seed": seed,
                       "decks": [list(p.cycle) for p in self.battle.players], "deployments": []}

    # --- input ------------------------------------------------------------

    def slot_at(self, sx, sy):
        if HAND_Y <= sy <= HAND_Y + 50 and AX <= sx < AX + 4 * 90:
            return (sx - AX) // 90
        return None

    def tile_at(self, sx, sy):
        if AX <= sx < AX + AW and AY <= sy < AY + AH:
            return int(32 - (sy - AY) / TILE), int((sx - AX) / TILE)    # (y, x) in blue's view
        return None

    def play(self, slot, y, x):
        card = self.battle.players[0].cycle[slot]
        if self.battle.deploy_card(0, card, to_world(0, y, x)):
            self.record["deployments"].append(dict(tick=self.battle.tick, player=0, slot=slot, card=card, y=y, x=x))
            self.selected, self.message = None, ""
        else:
            self.message = f"can't play {card} there" if self.battle.players[0].can_play_card(card) \
                else f"not enough elixir for {card}"

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.type == pygame.KEYDOWN and pygame.K_1 <= event.key <= pygame.K_4:
                self.selected = event.key - pygame.K_1
            elif event.type == pygame.MOUSEBUTTONDOWN:
                slot, tile = self.slot_at(*event.pos), self.tile_at(*event.pos)
                if slot is not None:
                    self.selected = slot
                elif tile is not None and self.selected is not None:
                    self.play(self.selected, *tile)

    # --- simulation ---------------------------------------------------------

    def model_turn(self):
        """Decide from the current snapshot, then deploy (with the live delay)."""
        slot, y, x = map(int, self.model(self.env.observe(1)))
        if slot:
            card = self.battle.players[1].cycle[slot - 1]
            self.env._deploy(1, (slot, y, x))
            self.record["deployments"].append(dict(tick=self.battle.tick, player=1, slot=slot - 1,
                                                   card=card, y=y, x=x))

    def tick(self):
        if self.battle.tick % TICKS_PER_DECISION == 0:
            self.model_turn()
        self.battle.step(1 / FPS)

    # --- drawing ------------------------------------------------------------

    def draw_ui(self):
        p0 = self.battle.players[0]
        for slot, card in enumerate(p0.cycle[:4]):
            left = AX + slot * 90
            affordable = p0.can_play_card(card)
            pygame.draw.rect(self.screen, (255, 230, 120) if slot == self.selected else (220, 220, 220),
                             (left, HAND_Y, 84, 50))
            pygame.draw.rect(self.screen, BLACK, (left, HAND_Y, 84, 50), 1)
            for row, text in enumerate((f"{slot + 1}. {card}", f"cost {Card(card).elixir}")):
                label = self.font.render(text, True, BLACK if affordable else (150, 150, 150))
                self.screen.blit(label, (left + 4, HAND_Y + 6 + row * 18))
        nxt = self.font.render(f"next: {p0.cycle[4]}", True, BLACK)
        self.screen.blit(nxt, (AX + 4 * 90 + 6, HAND_Y + 6))
        pygame.draw.rect(self.screen, (200, 200, 200), (AX, HAND_Y - 20, AW, 12))
        pygame.draw.rect(self.screen, (200, 60, 200), (AX, HAND_Y - 20, AW * p0.elixir / 10, 12))
        status, color = f"elixir {p0.elixir:.1f}   t={self.battle.time:.0f}s   {self.message}", BLACK
        if self.battle.game_over:
            won = self.battle.winner == 0
            status, color = f"GAME OVER: {'you win' if won else 'model wins'}   (Esc to quit)", BLUE if won else RED
        self.screen.blit(self.font.render(status, True, color), (AX, AY - 30))
        if self.paused:
            self.screen.blit(self.font.render("PAUSED", True, RED), (AX + AW // 2 - 20, AY - 15))

    def save(self):
        self.record.update(winner=self.battle.winner, ticks=self.battle.tick)
        path = Path("captures") / time.strftime("sim_%Y%m%d_%H%M%S.json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(self.record, indent=1))
        print("saved", path)

    def run(self):
        while self.running:
            self.process_events()
            self.clock.tick(FPS)
            if not self.paused and not self.battle.game_over:
                self.tick()
            self.screen.fill(WHITE)
            self.draw_arena()
            self.draw_entities()
            self.draw_ui()
            pygame.display.flip()
        self.save()
        pygame.quit()


def replay(record):
    """Rebuild a recorded game's battle. The model's deployments go through
    CREnv._deploy in the same order, so its random delays come out the same."""
    env = CREnv(play_delay=LIVE_PLAY_DELAY)
    env.reset(seed=record["seed"])
    pending = list(record["deployments"])
    while env.battle.tick <= record["ticks"] and not env.battle.game_over:   # game over doesn't advance tick
        while pending and pending[0]["tick"] == env.battle.tick:
            d = pending.pop(0)
            if d["player"] == 0:
                env.battle.deploy_card(0, d["card"], to_world(0, d["y"], d["x"]))
            else:
                env._deploy(1, (d["slot"] + 1, d["y"], d["x"]))
        env.battle.step(1 / FPS)
    return env.battle


if __name__ == "__main__":
    Game(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else int(time.time())).run()
