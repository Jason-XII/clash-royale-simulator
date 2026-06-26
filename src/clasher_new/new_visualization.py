import pygame
from battle import BattleState, Building, Projectile
from arena import Position
from player import PlayerState

# Initializes pygame, defines how big is one tile, defines colors
pygame.init()
TILE = 22
AX, AY = 50, 50
AW, AH = 18*TILE, 32*TILE
W, H = AW+120, AH+100
BLUE, RED, GREEN, CYAN, DKGRAY, BLACK, WHITE = (100,100,255),(255,100,100),(100,255,100),(100,255,255),(64,64,64),(0,0,0),(255,255,255)

def w2s(x, y):
    y = 32 - y
    return int(AX + x * TILE), int(AY + y * TILE)


class Visualizer:
    def __init__(self, battle=None):
        """If given a battle object, then render that battle."""
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 18)
        self.battle = battle or BattleState(PlayerState(0, player_0_deck, 10), PlayerState(1, player_1_deck, 10))
        self.paused = False
        self.running = True
        self.speed = 1
        self.scheduled = []

    def deploy(self, card, pos, player=0, delay=0):
        if delay:
            self.scheduled.append((delay, player, card, pos))
        else:
            self.battle.deploy_card(player, card, Position(*pos))

    def draw_arena(self):
        pygame.draw.rect(self.screen, GREEN, (AX,AY,AW,AH))
        ry = AY+15*TILE
        pygame.draw.rect(self.screen, CYAN, (AX, ry, AW, 2*TILE))
        for bx in [2, 13]:
            pygame.draw.rect(self.screen, DKGRAY, (AX+bx*TILE, ry, 3*TILE, 2*TILE))
        pygame.draw.rect(self.screen, DKGRAY, (AX, AY, 6*TILE, TILE))
        pygame.draw.rect(self.screen, DKGRAY, (AX+12*TILE, AY, 6 * TILE, TILE))
        pygame.draw.rect(self.screen, DKGRAY, (AX, AY+31*TILE, 6 * TILE, TILE))
        pygame.draw.rect(self.screen, DKGRAY, (AX + 12 * TILE, AY+31*TILE, 6 * TILE, TILE))
        for x in range(19): pygame.draw.line(self.screen, (0,150,0), (AX+x*TILE,AY), (AX+x*TILE,AY+AH), 1)
        for y in range(33): pygame.draw.line(self.screen, (0,150,0), (AX,AY+y*TILE), (AX+AW,AY+y*TILE), 1)

    def draw_entities(self):
        for e in self.battle.entities.values():
            if not e.is_alive: continue
            sx, sy = w2s(e.position.x, e.position.y)
            color = BLUE if e.player == 0 else RED
            r = int(e.data.collision_radius * TILE)
            if isinstance(e, Projectile):
                pygame.draw.circle(self.screen, color, (sx,sy), max(r,4), 2)
            else:
                pygame.draw.circle(self.screen, color, (sx, sy), max(r, 4))
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
        text = f"t={self.battle.time:.1f}s  tick={self.battle.tick}  speed={self.speed}x time={self.battle.time:.2f}"
        if self.battle.game_over:
            text += f"  Winner={'RED' if self.battle.winner == 1 else 'BLUE'}"
        txt = self.font.render(text, True, BLACK)
        self.screen.blit(txt, (AX, AY+AH+10))
        if self.paused:
            p = self.font.render("PAUSED", True, RED)
            self.screen.blit(p, (AX+AW//2-20, AY-20))

    def process_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.running = False
                elif ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif pygame.K_1 <= ev.key <= pygame.K_5:
                    self.speed = ev.key - pygame.K_0

    def render_frame(self):
        self.screen.fill(WHITE)
        self.draw_arena()
        self.draw_entities()
        self.draw_ui()
        pygame.display.flip()

    def run(self):
        while self.running:
            self.process_events()
            dt = self.clock.tick(60) / 1000.0
            if not self.paused and not self.battle.game_over:
                for _ in range(self.speed):
                    self.battle.step(dt)
                    for d in self.scheduled[:]:
                        if self.battle.time >= d[0]:
                            self.battle.players[d[1]].elixir = 10
                            self.battle.deploy_card(d[1], d[2], Position(*d[3]))
                            self.scheduled.remove(d)
            self.render_frame()
        pygame.quit()

player_0_deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
player_1_deck = ['Minions', 'Archer', 'MiniPekka', 'Musketeer', 'Giant', 'Fireball', 'Arrows', 'Knight']

schedule = [('Knight', (9.5, 0.5), 0, 0),
            ('Archer', (8.5, 31.5), 1, 1.376),
            ('Musketeer', (12.5, 6.5), 0, 8.447),
            ('Musketeer', (16.5, 29.5), 1, 11.13),
            ('Minions', (3.5, 12.5), 0, 18.49),
            ('MiniPekka', (14.5, 20.5), 1, 18.936),
            ('MiniPekka', (14.5, 14.5), 0, 19.436)]

schedule2 = [('Minions', (10.5, 10.5), 0, 0)]


if __name__ == "__main__":
    v = Visualizer()
    for each in schedule:
        v.deploy(*each)
    v.run()
