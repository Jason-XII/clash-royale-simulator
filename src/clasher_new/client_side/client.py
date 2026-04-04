import random

import pygame
import socket
import json
import threading
from card_utils import card_data, Card
from player import PlayerState
import os
font_path = os.path.join(os.path.dirname(pygame.__file__), "freesansbold.ttf")

DEBUG = False

# --- Config ---
CARDS = [
    "Knight", "Giant", "Archer", "Goblins", "Pekka", "MiniPekka",
    "Minions", "Skeletons", "SkeletonArmy", "Balloon", "Witch",
    "Barbarians", "Golem", "Valkyrie", "Bomber", "Musketeer",
    "BabyDragon", "Prince", "Wizard", "SpearGoblins",
    "GiantSkeleton", "HogRider", "MinionHorde"
]

english_names = [card_data[each]['englishName'] for each in CARDS]
resolved = dict(zip(english_names, CARDS))

CARD_SIZE = 80
CARD_W, CARD_H = 60, 90
COLS = 6
PORT = 9999

pygame.init()
screen = pygame.display.set_mode((700, 700))
pygame.display.set_caption("Card Selection")
font = pygame.font.Font(font_path, 18)
small_font = pygame.font.Font(font_path, 12)
clock = pygame.time.Clock()

# --- Load images ---
images = {}
for name in english_names:
    try:
        img = pygame.image.load(f"images/{name}.png")
        images[name] = pygame.transform.scale(img, (CARD_W, CARD_H))
    except:
        surf = pygame.Surface((CARD_W, CARD_H))
        surf.fill((80, 80, 80))
        images[name] = surf

# --- Card selection screen ---
def card_selection_screen():
    selected = []
    while True:
        screen.fill((30, 30, 30))
        for i, name in enumerate(english_names):
            col, row = i % COLS, i // COLS
            x, y = 20 + col * (CARD_W + 10), 20 + row * (CARD_H + 20)
            screen.blit(images[name], (x, y))
            if name in selected:
                pygame.draw.rect(screen, (0, 255, 0), (x, y, CARD_W, CARD_H), 3)
            lbl = small_font.render(name[:8], True, (200, 200, 200))
            screen.blit(lbl, (x, y + CARD_H + 2))

        msg = font.render(f"Select 8 cards ({len(selected)}/8)", True, (255, 255, 255))
        screen.blit(msg, (20, 600))
        if len(selected) == 8:
            btn = pygame.draw.rect(screen, (0, 180, 0), (500, 590, 160, 45))
            screen.blit(font.render("Confirm", True, (255,255,255)), (535, 603))

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for i, name in enumerate(english_names):
                    col, row = i % COLS, i // COLS
                    x, y = 20 + col * (CARD_W + 10), 20 + row * (CARD_H + 20)
                    if x <= mx <= x+CARD_W and y <= my <= y+CARD_H:
                        if name in selected: selected.remove(name)
                        elif len(selected) < 8: selected.append(name)
                if len(selected) == 8 and 500 <= mx <= 660 and 590 <= my <= 635:
                    return selected

# --- IP input screen ---
def ip_input_screen():
    ip = ""
    while True:
        screen.fill((30, 30, 30))
        screen.blit(font.render("Enter server IP:", True, (255,255,255)), (200, 270))
        pygame.draw.rect(screen, (60,60,60), (150, 310, 400, 40))
        screen.blit(font.render(ip, True, (255,255,0)), (160, 320))
        screen.blit(font.render("Press ENTER to connect", True, (180,180,180)), (190, 370))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and ip: return ip
                elif event.key == pygame.K_BACKSPACE: ip = ip[:-1]
                elif event.unicode.isprintable(): ip += event.unicode

# --- Main ---
if not DEBUG:
    deck = card_selection_screen()
    deck = [resolved[each] for each in deck]
else:
    deck = random.sample(CARDS, 8)
if not DEBUG:
    server_ip = ip_input_screen()
else:
    server_ip = "100.64.167.179"

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((server_ip, PORT))

# Receive hello
hello = json.loads(sock.makefile().readline())
player_id = hello['player_id']
print(f"Connected as Player {player_id}")
sock.sendall((json.dumps({'type': 'deck', 'cards': deck}) + '\n').encode())
print("Waiting for opponent...")
json.loads(sock.makefile().readline())  # 'start' message
print("Game started!")

# State receiver thread
state = {'entities': [], 'elixir': [5, 5], 'game_over': False, 'winner': None}
def receiver():
    f = sock.makefile()
    while True:
        try:
            msg = json.loads(f.readline())
            if msg['type'] == 'state':
                state.update(msg)
        except: break

threading.Thread(target=receiver, daemon=True).start()

TILE = 22
AX, AY = 50, 50
AW, AH = 18 * TILE, 32 * TILE
W, H = AW + 120, AH + 100
BLUE, RED, GREEN, CYAN, DKGRAY, BLACK, WHITE = (
    (100, 100, 255), (255, 100, 100), (100, 255, 100),
    (100, 255, 255), (64, 64, 64), (0, 0, 0), (255, 255, 255)
)

def w2s(x, y):
    if player_id == 0:
        y = 32 - y
    return int(AX + x * TILE), int(AY + y * TILE)

screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
render_font = pygame.font.Font(None, 18)

def draw_arena():
    pygame.draw.rect(screen, GREEN, (AX, AY, AW, AH))
    ry = AY + 15 * TILE
    pygame.draw.rect(screen, CYAN, (AX, ry, AW, 2 * TILE))
    for bx in [2, 13]:
        pygame.draw.rect(screen, DKGRAY, (AX + bx * TILE, ry, 3 * TILE, 2 * TILE))
    for x in range(19):
        pygame.draw.line(screen, (0, 150, 0), (AX + x * TILE, AY), (AX + x * TILE, AY + AH), 1)
    for y in range(33):
        pygame.draw.line(screen, (0, 150, 0), (AX, AY + y * TILE), (AX + AW, AY + y * TILE), 1)

def draw_entities():
    for e in state.get('entities', []):
        sx, sy = w2s(e['x'], e['y'])
        color = BLUE if e['player'] == player_id else RED
        r = e['collision_radius']*TILE
        pygame.draw.circle(screen, color, (sx, sy), r)
        pygame.draw.circle(screen, BLACK, (sx, sy), r, 1)
        lbl = render_font.render(e['card_name'], True, BLACK)
        screen.blit(lbl, lbl.get_rect(center=(sx, sy + r + 10)))
        if e['hp'] > 0 and e['max_hp'] > 0:
            bw = max(r * 2, 16)
            pygame.draw.rect(screen, BLACK, (sx - bw // 2 - 1, sy - r - 12, bw + 2, 5))
            pygame.draw.rect(screen, GREEN, (sx - bw // 2, sy - r - 11, int((e['hp'] / e['max_hp']) * bw), 3))
            if e.get('type') == 'building':
                hp_txt = render_font.render(str(int(e['hp'])), True, WHITE)
                screen.blit(hp_txt, hp_txt.get_rect(center=(sx, sy)))

def draw_ui():
    t = state.get('time', 0)
    text = f"Player {player_id}  t={t:.1f}s  Elixir={state['elixir'][player_id]:.1f}"
    if state['game_over']:
        winner = state['winner']
        text += f"  GAME OVER - {'YOU WIN!' if winner == player_id else 'YOU LOSE'}"
    txt = render_font.render(text, True, BLACK)
    screen.blit(txt, (AX, AY + AH + 10))

player_state = PlayerState(player_id, list(deck), 5)
selected_card = None
dragging = False
drag_pos = (0, 0)

# Card display sizing: fill arena width with 4 cards + gaps
CARD_GAP = 8
HAND_CARD_W = (AW - CARD_GAP * 3) // 4
HAND_CARD_H = int(HAND_CARD_W * 420 / 285)
HAND_Y = AY + AH + 30

# Pre-scale card images for hand display
hand_images = {}
hand_images_bw = {}
for name in deck:
    img = images[card_data[name]['englishName']]
    hand_images[name] = pygame.transform.scale(img, (HAND_CARD_W, HAND_CARD_H))
    bw = hand_images[name].copy()
    arr = pygame.surfarray.pixels3d(bw)
    grey = arr.mean(axis=2, keepdims=True).astype(arr.dtype)
    arr[:] = grey
    del arr
    hand_images_bw[name] = bw

def get_hand_rects():
    """Return list of (card_name, rect) for the 4 cards in hand."""
    hand = player_state.cycle[:4]
    rects = []
    for i, name in enumerate(hand):
        x = AX + i * (HAND_CARD_W + CARD_GAP)
        rects.append((name, pygame.Rect(x, HAND_Y, HAND_CARD_W, HAND_CARD_H)))
    return rects

def draw_hand():
    for name, rect in get_hand_rects():
        can_play = player_state.elixir >= Card(name).elixir
        img = hand_images[name] if can_play else hand_images_bw[name]
        if dragging and selected_card == name:
            # Draw dimmed in slot while dragging
            screen.blit(hand_images_bw[name], rect)
        else:
            screen.blit(img, rect)
        # Elixir cost badge
        cost = str(Card(name).elixir)
        badge = render_font.render(cost, True, WHITE)
        pygame.draw.circle(screen, (128, 0, 128), (rect.x + 12, rect.y + 12), 10)
        screen.blit(badge, badge.get_rect(center=(rect.x + 12, rect.y + 12)))

def draw_elixir_bar():
    bar_w = AW
    bar_h = 14
    bar_x, bar_y = AX, HAND_Y - 20
    elixir = player_state.elixir
    pygame.draw.rect(screen, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h))
    fill_w = int((elixir / 10.0) * bar_w)
    pygame.draw.rect(screen, (160, 32, 240), (bar_x, bar_y, fill_w, bar_h))
    txt = render_font.render(f"{elixir:.1f}/10", True, WHITE)
    screen.blit(txt, txt.get_rect(center=(bar_x + bar_w // 2, bar_y + bar_h // 2)))

# Resize window to fit hand
H_NEW = HAND_Y + HAND_CARD_H + 20
screen = pygame.display.set_mode((W, H_NEW))

# --- Main game loop ---
running = True
while running:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

        elif ev.type == pygame.MOUSEBUTTONDOWN and not state['game_over']:
            mx, my = ev.pos
            for name, rect in get_hand_rects():
                if rect.collidepoint(mx, my) and player_state.elixir >= Card(name).elixir:
                    selected_card = name
                    dragging = True
                    drag_pos = (mx, my)

        elif ev.type == pygame.MOUSEMOTION and dragging:
            drag_pos = ev.pos

        elif ev.type == pygame.MOUSEBUTTONUP and dragging:
            mx, my = ev.pos
            wx = (mx - AX) / TILE
            wy = (my - AY) / TILE
            if 0 <= wx <= 18 and 0 <= wy <= 32 and player_state.can_play_card(selected_card):
                sock.sendall((json.dumps({
                    'type': 'deploy',
                    'card': selected_card,
                    'x': wx, 'y': wy if player_id == 1 else 32-wy
                }) + '\n').encode())
                player_state.play_card(selected_card)
            selected_card = None
            dragging = False

    # Sync elixir from server
    player_state.elixir = state['elixir'][player_id]

    # Draw
    screen.fill(WHITE)
    draw_arena()
    draw_entities()
    draw_elixir_bar()
    draw_hand()
    draw_ui()

    # Draw dragged card at cursor
    if dragging and selected_card:
        img = hand_images[selected_card]
        screen.blit(img, (drag_pos[0] - HAND_CARD_W // 2, drag_pos[1] - HAND_CARD_H // 2))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
