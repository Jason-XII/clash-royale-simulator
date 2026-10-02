import pygame
import json
from pathlib import Path
from threading import Thread

from run_raw_capture import mainloop
from environment import Bank, entity_names, card_types
from card_utils import Card

import numpy as np
import time
import subprocess
import random

from stable_baselines3 import PPO

import masked_spatial  # registers MaskedSpatialPolicy so PPO.load can unpickle it
from battle import BattleState, Building
from player import PlayerState
from core import Position as _Pos

pygame.init()
TILE = 22
AX, AY = 50, 50
AW, AH = 18*TILE, 32*TILE
W, H = AW+120, AH+100
BLUE, RED, GREEN, CYAN, DKGRAY, BLACK, WHITE = (100,100,255),(255,100,100),(100,255,100),(100,255,255),(64,64,64),(0,0,0),(255,255,255)

steps = ('bank-11.1M',)
models = [PPO.load("final.zip", device="cpu")]

# Static legal-deploy tiles for the local player (own-half zones; fences & tower tiles
# excluded). The live client has no simulator battle, so we approximate legality with the
# arena rules; the real client still enforces true legality on each swipe.
_MASK_BATTLE = BattleState(PlayerState(0, [], 0), PlayerState(1, [], 0))
_MASK_BATTLE.building_positions = [
    (e.position.x, e.position.y, e.data.collision_radius)
    for e in _MASK_BATTLE.entities.values() if isinstance(e, Building)
]
_TROOP_TILES = np.array([
    [_MASK_BATTLE.can_place_troop(0, _Pos(x + .5, y + .5))
     for x in range(18)] for y in range(32)
], dtype=np.int8)


def build_legal_mask(hand_names, own_elixir):
    mask = np.zeros((4, 32, 18), dtype=np.int8)
    for slot, name in enumerate(hand_names[:4]):
        c = Card(name)
        if c.elixir <= own_elixir:
            mask[slot] = 1 if c.type == "spell" else _TROOP_TILES
    return mask

xlow = 16
xhigh = 1053
ylow = 295
yhigh = 1714

x_tile_width = (xhigh-xlow)/18
y_tile_width = (yhigh-ylow)/30
ylow -= y_tile_width
yhigh += y_tile_width

def slot_to_screen(slot):
    return 335+(slot-1)*200, 2220

def tile_to_screen(tile_x, tile_y):
    return xlow+x_tile_width*(tile_x+0.5), yhigh-y_tile_width*(tile_y+0.5)

def swipe(slot, y, x):
    x1, y1 = slot_to_screen(slot)
    x2, y2 = tile_to_screen(x, y)
    subprocess.run(
        [
            "adb",
            "-s",
            "emulator-5554",
            "shell",
            "input",
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            "350",
        ],
        check=True,
    )

def w2s(x, y):
    return int(AX + x * TILE), int(AY + y * TILE)

with open('cards.json') as f:
    card_data = json.loads(f.read())
cards = {each['id']: each['name'] for each in card_data['items']}

class Visualizer:
    def __init__(self):
        """If given a battle object, then render that battle."""
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 18)
        self.entities = {}
        self.snapshot = {}
        self.local_player_index = None
        self.running = True
        self.start_time = time.time()
        self.observation_history = []
        self.last_observation_ms = None
        self.model_index = 0
        self.model = models[self.model_index]
        self.model_label = steps[self.model_index]
        self.bank = Bank()
        self.pending = None             # (card, battle clock) of a swipe the game hasn't shown yet
        self.last_decision_clock = None
        self.tower_info = {}
        self.score_started = False
        self.last_score_snapshot_ms = None
        self.last_score_time = 0.0
        self.own_area = 0.0
        self.damage_area = 0.0
        self.own_integrity = 1.0
        self.enemy_integrity = 1.0
        self.own_crowns_lost = 0
        self.enemy_crowns_lost = 0
        self.capture = None             # this game's capture file, opened when the battle starts
        self.last_captured_ms = None

    def record(self, event, **fields):
        """Append one JSON line to captures/game_<date>_<time>.jsonl, flushed immediately."""
        if self.capture is None:
            Path('captures').mkdir(exist_ok=True)
            path = Path('captures') / time.strftime('game_%Y%m%d_%H%M%S.jsonl')
            self.capture = path.open('w')
            print('Recording this game to', path)
            self.capture.write(json.dumps({'event': 'start', 'model': self.model_label,
                                           'local_player_index': self.local_player_index}) + '\n')
        self.capture.write(json.dumps({'event': event, **fields}, default=int) + '\n')
        self.capture.flush()

    def capture_snapshot(self):
        """Record each new snapshot once the battle has started (all six towers seen)."""
        snapshot = self.snapshot
        if self.score_started and snapshot['t_ms'] != self.last_captured_ms:
            self.last_captured_ms = snapshot['t_ms']
            self.record('snapshot', snapshot=snapshot)

    def own_view(self, entity):
        """Entity position in the model's own view (own towers near y=0), as in training.

        Measured from a capture as player 0: raw y already is own-view y and raw x is
        mirrored (a swipe at tile x lands at raw x = 18 - x). Player 1 is assumed to be
        the 180-degree rotation of that; confirm it with a capture played as player 1.
        """
        x, y = entity['pos_x_7c'] / 1000, entity['pos_y_80'] / 1000
        return (18 - x, y) if self.local_player_index == 0 else (x, 32 - y)

    @staticmethod
    def is_king(tower):
        # kind_30 is not the tower type (the king switches 12 -> 13 when it activates);
        # both king towers sit on the center line.
        return abs(tower['pos_x_7c'] - 9000) < 100

    def enemy_troops_in_half(self):
        """Pointers of enemy troops in our half, for Bank (same rule as in training).

        Towers and spells don't count; unknown cards do, to stay on the safe side.
        """
        intruders = set()
        for entity in self.snapshot['entities']:
            if entity['side_78'] == self.local_player_index or entity['card_id_ac'] == -1:
                continue
            name = cards.get(entity['card_id_ac'])
            try:
                if name is not None and Card(name).type == 'spell':
                    continue
            except KeyError:
                pass
            if self.own_view(entity)[1] < 16:
                intruders.add(entity['ptr'])
        return intruders

    def update_score(self):
        if not self.snapshot.get('entity_list_valid'):
            return
        if self.snapshot['t_ms'] == self.last_score_snapshot_ms:
            return
        towers = [
            e for e in self.snapshot['entities']
            if e['card_id_ac'] == -1 and e['kind_30'] in (12, 13)
        ]
        if not self.score_started:
            if len(towers) < 6:
                return
            self.tower_info = {
                e['ptr']: (e['side_78'], self.is_king(e)) for e in towers
            }
            self.score_started = True
            # In training the own king tower is at y=3; anything else means the model
            # would see the arena upside down (see own_view).
            own_king = [e for e in towers if self.is_king(e) and e['side_78'] == self.local_player_index]
            if not own_king or abs(self.own_view(own_king[0])[1] - 3) > 0.5:
                print(f'WARNING: own king tower is not at own-view y=3 for player '
                      f'{self.local_player_index}; fix own_view before trusting this game.')

        # Known towers that disappear from the entity list remain at zero.
        ratios = {pointer: 0.0 for pointer in self.tower_info}
        for entity in towers:
            pointer = entity['ptr']
            if pointer in ratios and entity['max_hp_14'] > 0:
                ratios[pointer] = float(np.clip(
                    entity['hp_10'] / entity['max_hp_14'], 0.0, 1.0
                ))

        own = enemy = 0.0
        own_princess_lost = enemy_princess_lost = 0
        own_king_lost = enemy_king_lost = False
        for pointer, (side, king) in self.tower_info.items():
            ratio = ratios[pointer]
            is_own = side == self.local_player_index
            weight = 0.4 if king else 0.3
            if is_own:
                own += weight * ratio
                own_king_lost |= king and ratio == 0
                own_princess_lost += int(not king and ratio == 0)
            else:
                enemy += weight * ratio
                enemy_king_lost |= king and ratio == 0
                enemy_princess_lost += int(not king and ratio == 0)

        own = float(np.clip(own, 0.0, 1.0))
        enemy = float(np.clip(enemy, 0.0, 1.0))
        battle_time = float(self.snapshot['battle_clock_220'])
        if self.last_score_snapshot_ms is not None and battle_time >= self.last_score_time:
            dt = battle_time - self.last_score_time
            self.own_area += 0.5 * (self.own_integrity + own) * dt
            self.damage_area += 0.5 * (
                (1 - self.enemy_integrity) + (1 - enemy)
            ) * dt
        self.last_score_snapshot_ms = self.snapshot['t_ms']
        self.last_score_time = battle_time
        self.own_integrity = own
        self.enemy_integrity = enemy
        self.own_crowns_lost = 3 if own_king_lost else own_princess_lost
        self.enemy_crowns_lost = 3 if enemy_king_lost else enemy_princess_lost

    def scores(self, final=False):
        remaining = max(0.0, 300.0 - self.last_score_time)
        own_future = self.own_integrity
        enemy_future = self.enemy_integrity
        if final and self.own_crowns_lost > self.enemy_crowns_lost:
            own_future = 0.0
        if final and self.enemy_crowns_lost > self.own_crowns_lost:
            enemy_future = 0.0
        defense = 100 * (self.own_area + own_future * remaining) / 300
        offense = 100 * (self.damage_area + (1 - enemy_future) * remaining) / 300
        if self.enemy_crowns_lost > self.own_crowns_lost:
            result, result_score = 'win', 100
        elif self.enemy_crowns_lost < self.own_crowns_lost:
            result, result_score = 'loss', 0
        else:
            result, result_score = 'draw', 50
        total = 0.45 * defense + 0.40 * offense + 0.15 * result_score
        return total, defense, offense, result

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
        obs = np.zeros((32, 18, 15), dtype=np.float32)
        for entity in list(self.snapshot['entities']):
            if entity['card_id_ac'] == -1 and entity['kind_30'] in (12, 13):
                name = "KingTower" if self.is_king(entity) else 'King_PrincessTowers'
            else:
                if entity['card_id_ac'] in cards:
                    name = cards[entity['card_id_ac']]
                elif entity['card_id_ac'] == -1: continue
                elif str(entity['card_id_ac']).startswith('13'):
                    real_id = entity['card_id_ac'] + 13000000
                    name = 'Evo ' + cards[real_id]
                elif str(entity['card_id_ac']).startswith('203'):
                    real_id = entity['card_id_ac'] - 177000000
                    name = "Hero" + str(cards.get(real_id))
                else:
                    print('Entity unknown:', entity['card_id_ac'])
                    name = entity['card_id_ac']
            r = 0.5 * TILE
            x, y = self.own_view(entity)
            color = BLUE if entity['side_78'] == self.local_player_index else RED

            # we turn that into an environment compatible observation
            card = Card(name)
            entity_id = entity_names.index(name)
            card_type = card_types.index(card.type)
            player_id = int(entity['side_78'] != self.local_player_index)
            elixir = card.elixir
            is_air = int(card.is_air_unit)
            attacks_ground, attacks_air = int(card.attack_ground), int(card.attack_air)

            speed = card.speed
            hp_left = np.log(entity['hp_10']) / 10 if entity['hp_10'] != 0 else 0
            hp_percentage = entity['hp_10'] / card.hp if card.hp != 0 else 0
            hit_speed = card.hit_speed
            attack_range = card.range / 3
            sight_range = card.sight_range / 3
            damage = card.damage / 200
            projectile_damage = card.projectile_data.damage / 200
            x1 = int(np.clip(x, 0, 17))
            y1 = int(np.clip(y, 0, 31))
            obs_arr = np.array([entity_id, card_type, player_id, elixir, speed, is_air, attacks_ground, attacks_air,
                                hp_left, hp_percentage, hit_speed, attack_range, sight_range, damage,
                                projectile_damage])
            obs[y1][x1] = obs_arr.copy()

            sx, sy = w2s(x, 32 - y)  # draw own side at the bottom, like the phone screen
            pygame.draw.circle(self.screen, color, (sx, sy), max(r, 4))
            pygame.draw.circle(self.screen, BLACK, (sx, sy), max(r, 4), 1)
            lbl = self.font.render(str(name), True, BLACK)
            self.screen.blit(lbl, lbl.get_rect(center=(sx, sy + r + 10)))

            bw = max(r * 2, 16)
            max_hp = entity.get('max_hp_14')
            hp_width = (entity['hp_10'] / max_hp) * bw if max_hp > 0 else bw
            pygame.draw.rect(self.screen, BLACK, (sx - bw // 2 - 1, sy - r - 12, bw + 2, 5))
            pygame.draw.rect(self.screen, GREEN, (sx - bw // 2, sy - r - 11, hp_width, 3))
            hp_txt = self.font.render(str(int(entity['hp_10'])), True, WHITE)
            self.screen.blit(hp_txt, hp_txt.get_rect(center=(sx, sy)))
        hand = []
        for each in self.snapshot['hand']:
            data_id = each['data_id_40']
            if data_id not in cards:
                return
            hand.append(cards[data_id])
        next_card_id = self.snapshot['next_card_data_id_40']
        if len(hand) != 4 or next_card_id not in cards:
            return
        hand.append(cards[next_card_id])
        hand_names = list(hand)
        hand = np.array([entity_names.index(each) for each in hand], dtype=np.int32)

        snapshot_ms = self.snapshot['t_ms']
        if not self.observation_history:
            self.observation_history = [obs.copy() for _ in range(8)]
            self.last_observation_ms = snapshot_ms
        elif snapshot_ms - self.last_observation_ms >= 500:
            self.observation_history.append(obs.copy())
            self.observation_history = self.observation_history[-8:]
            self.last_observation_ms = snapshot_ms
        grid = np.stack(self.observation_history)

        battle_time = self.snapshot['battle_clock_220']
        if battle_time < 120:
            phase, time_left = 0, 120 - battle_time
        elif battle_time < 180:
            phase, time_left = 1, 180 - battle_time
        elif battle_time < 240:
            phase, time_left = 2, 240 - battle_time
        else:
            phase, time_left = 3, 300 - battle_time
        final_observation = {
            'grid': grid,
            'hand': hand,
            'elixir': np.array([self.snapshot['own_elixir_1e0']], dtype=np.float32),
            'phase': phase,
            'time_till_next_phase': np.array([time_left / 120.0], dtype=np.float32),
        }
        final_observation['legal_mask'] = build_legal_mask(
            hand_names, self.snapshot['own_elixir_1e0'])
        if time.time() - self.start_time > 0.5:
            self.start_time = time.time()
            elixir = self.snapshot['own_elixir_1e0']
            clock = self.snapshot['battle_clock_220']
            if clock == self.last_decision_clock:
                return                      # the battle clock stopped: the game is over
            self.last_decision_clock = clock
            # A swipe takes ~2 s to leave the hand (~1 s to cost elixir). In training a play
            # lands at once, so don't decide again on the stale state; give up after 3 s.
            if self.pending is not None:
                card, since = self.pending
                if card in hand_names[:4] and clock - since < 3:
                    return
                self.pending = None
            # Same rules as training: no decision while banking or while no card is playable.
            if (self.bank.active(elixir, self.enemy_troops_in_half)
                    or not final_observation['legal_mask'].any()):
                return
            action = self.model.predict(final_observation, deterministic=False)[0]
            slot, y, x = self.bank.resolve(action, self.enemy_troops_in_half)
            if self.capture is not None:
                self.record('decision', t_ms=self.snapshot['t_ms'],
                            battle_clock=self.snapshot['battle_clock_220'], elixir=elixir,
                            hand=hand_names, action=list(action), played=[slot, y, x],
                            bank_target=self.bank.target)
            if slot:
                card_name = entity_names[hand[slot - 1]]
                if Card(card_name).elixir > elixir: return
                swipe(slot, y, x)
                self.pending = (card_name, clock)

    def draw_ui(self):
        hand = []
        for each in self.snapshot['hand']:
            data_id = each['data_id_40']
            if data_id in cards:
                hand.append(cards[data_id])
            else:
                print('Unknown card in hand:' , data_id)
                hand.append(str(data_id))
        text = f"t={self.snapshot['battle_clock_220']:.1f}s elixir={self.snapshot['own_elixir_1e0']} hand={hand}"
        txt = self.font.render(text, True, BLACK)
        self.screen.blit(txt, (AX, AY+AH+10))
        if self.score_started:
            total, defense, offense, result = self.scores()
            text = (
                f"model={self.model_label} total={total:.1f} defense={defense:.1f} "
                f"offense={offense:.1f} result={result}"
            )
            txt = self.font.render(text, True, BLACK)
            self.screen.blit(txt, (AX, AY+AH+30))

    def process_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False

    def render_frame(self):
        self.screen.fill(WHITE)
        self.clock.tick(60)
        self.draw_arena()
        self.draw_entities()
        self.draw_ui()
        pygame.display.flip()

    def run(self):
        while self.running:
            self.process_events()
            if self.snapshot and self.local_player_index is not None:
                if self.snapshot['battle_clock_220'] is not None:
                    self.update_score()
                    self.capture_snapshot()
                    self.render_frame()
        if self.score_started:
            total, defense, offense, result = self.scores(final=True)
            print(
                f"model={self.model_label} total={total:.2f} "
                f"defense={defense:.2f} offense={offense:.2f} result={result}"
            )
            if self.capture is not None:
                self.record('end', total=total, defense=defense, offense=offense, result=result,
                            own_crowns_lost=self.own_crowns_lost,
                            enemy_crowns_lost=self.enemy_crowns_lost)
                self.capture.close()
        pygame.quit()

window = Visualizer()
t = Thread(target=mainloop, args=(window, ), daemon=True)
t.start()
window.run()
