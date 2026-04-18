import pygame
from battle import BattleState, Building
from arena import Position
from player import PlayerState

# Initializes pygame, defines how big is one tile, defines colors
pygame.init()
TILE = 22
AX, AY = 50, 50
AW, AH = 18*TILE, 32*TILE
W, H = AW+120, AH+100
BLUE, RED, GREEN, CYAN, DKGRAY, BLACK, WHITE = (100,100,255),(255,100,100),(100,255,100),(100,255,255),(64,64,64),(0,0,0),(255,255,255)
def w2s(x, y): return int(AX+x*TILE), int(AY+y*TILE)


class Visualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 18)
        self.battle = BattleState(PlayerState(0, player_0_deck, 10), PlayerState(1, player_1_deck, 10))
        self.paused = False
        self.speed = 1
        self.scheduled = []

    def deploy(self, card, pos, player=0, delay=0):
        if delay: self.scheduled.append((delay, player, card, pos))
        else:
            self.battle.players[player].elixir = 10
            self.battle.deploy_card(player, card, Position(*pos))

    def draw_arena(self):
        pygame.draw.rect(self.screen, GREEN, (AX,AY,AW,AH))
        ry = AY+15*TILE
        pygame.draw.rect(self.screen, CYAN, (AX, ry, AW, 2*TILE))
        for bx in [2, 13]:
            pygame.draw.rect(self.screen, DKGRAY, (AX+bx*TILE, ry, 3*TILE, 2*TILE))
        for x in range(19): pygame.draw.line(self.screen, (0,150,0), (AX+x*TILE,AY), (AX+x*TILE,AY+AH), 1)
        for y in range(33): pygame.draw.line(self.screen, (0,150,0), (AX,AY+y*TILE), (AX+AW,AY+y*TILE), 1)

    def draw_entities(self):
        for e in self.battle.entities.values():
            if not e.is_alive: continue
            sx, sy = w2s(e.position.x, e.position.y)
            color = BLUE if e.player == 0 else RED
            r = int(e.data.collision_radius * TILE)
            pygame.draw.circle(self.screen, color, (sx,sy), max(r,4))
            pygame.draw.circle(self.screen, BLACK, (sx,sy), max(r,4), 1)
            # Name
            lbl = self.font.render(e.name, True, BLACK)
            self.screen.blit(lbl, lbl.get_rect(center=(sx, sy+r+10)))
            # HP bar
            if e.hp > 0:
                bw = max(r*2, 16)
                pygame.draw.rect(self.screen, BLACK, (sx-bw//2-1, sy-r-12, bw+2, 5))
                hp_width = (e.hp/e.data.hp)*bw if not e.shield_health else (e.shield_health/e.data.shield_health)*bw
                pygame.draw.rect(self.screen, GREEN, (sx-bw//2, sy-r-11, hp_width, 3))
                if isinstance(e, Building):
                    hp_txt = self.font.render(str(int(e.hp)), True, WHITE)
                    self.screen.blit(hp_txt, hp_txt.get_rect(center=(sx, sy)))

    def draw_ui(self):
        text = f"t={self.battle.time:.1f}s  tick={self.battle.tick}  speed={self.speed}x"
        if self.battle.game_over:
            text += f"  Winner={'RED' if self.battle.winner == 1 else 'BLUE'}"
        txt = self.font.render(text, True, BLACK)
        self.screen.blit(txt, (AX, AY+AH+10))
        if self.paused:
            p = self.font.render("PAUSED", True, RED)
            self.screen.blit(p, (AX+AW//2-20, AY-20))

    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE: running = False
                    elif ev.key == pygame.K_SPACE: self.paused = not self.paused
                    elif pygame.K_1 <= ev.key <= pygame.K_5: self.speed = ev.key - pygame.K_0

            if not self.paused and not self.battle.game_over:
                for _ in range(self.speed):
                    self.battle.step(self.battle.dt)
                    for d in self.scheduled[:]:
                        if self.battle.time >= d[0]:
                            self.battle.deploy_card(d[1], d[2], Position(*d[3]))
                            self.scheduled.remove(d)

            self.screen.fill(WHITE)
            self.draw_arena()
            self.draw_entities()
            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

player_0_deck = ['IceWizard', 'Bomber']*4
player_1_deck = ['DarkPrince', 'Prince']*4
if __name__ == "__main__":
    v = Visualizer()
    # v.deploy('IceWizard', (3.5, 13.5), delay=10)
    v.deploy('Prince', (18.5, 22.5), player=1)
    # v.deploy('Giant', (10.5, 6.5), delay=2)
    v.deploy('DarkPrince', (2.5, 22.5), player=1)
    v.run()
