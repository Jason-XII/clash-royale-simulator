import battle, player
from new_visualization import Visualizer
from core import Position
from card_utils import Card

import gymnasium as gym
from gymnasium import spaces
from random import shuffle, randint
import time
import numpy as np

from stable_baselines3.common.env_checker import check_env

player_0_deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
player_1_deck = ['Minions', 'Archer', 'MiniPekka', 'Musketeer', 'Giant', 'Fireball', 'Arrows', 'Knight']

b = battle.BattleState(player.PlayerState(0, player_0_deck, 10),
                       player.PlayerState(1, player_1_deck, 10))

deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']

entity_names = ['None', 'Knight', 'MiniPekka', 'Arrows', 'Minions', 'Archer',
                'Musketeer', 'Fireball', 'Giant', 'King_PrincessTowers',
                'KingTower', 'ArrowsSpell', 'FireballSpell']
# The agent has to learn that it can only deploy fireball and arrows, and the entities that actually appear are
# the arrows/fireball+spells thingy.

card_types = ['troop', 'character', 'spell', 'building']
# Troop mean princess tower, short for tower troop.
# Actual troops are represented as "characters".
speed_types = [0, 0.75, 1.0, 1.5]


class CREnv(gym.Env):
    max_entities = 128
    feature_count = 14
    def __init__(self, opponent_model=None, visualize=False, speed=1.0):
        super().__init__()
        self.opponent = opponent_model
        self.battle: battle.BattleState = None
        self.speed = speed
        self.visualize = visualize
        self.visualizer = None

        entity_id_high = np.array(
            [len(entity_names) - 1, len(card_types) - 1, 1],
            dtype=np.int64,
        )

        self.observation_space = spaces.Dict({
            "entity_ids": spaces.Box(
                low=0,
                high=np.broadcast_to(
                    entity_id_high,
                    (self.max_entities, 3),
                ).copy(),
                dtype=np.int64,
            ),
            "entity_features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.max_entities, self.feature_count),
                dtype=np.float32,
            ),
            "entity_mask": spaces.MultiBinary(self.max_entities),
            "hand": spaces.Box(
                low=0,
                high=len(entity_names) - 1,
                shape=(5,),
                dtype=np.int64,
            ),
            "hand_mask": spaces.MultiBinary(5),
            "state": spaces.Box(
                low=np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32),
                high=np.array([np.inf, 1.0, 4.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.MultiDiscrete([5, 32 * 18])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        shuffle(player_0_deck)
        shuffle(player_1_deck)
        self.battle = battle.BattleState(player.PlayerState(0, player_0_deck[:], 5.0),
                       player.PlayerState(1, player_1_deck[:], 5.0))
        if self.visualize:
            self.visualizer = Visualizer(self.battle)
        # Now return initial observation
        return self.observe(0), {}

    def decode_action(self, action):
        slot, tile = action
        return int(slot), int(tile) // 18, int(tile) % 18

    def opponent_action(self):
        obs1 = self.observe(1)
        slot, y, x = self.decode_action(self.opponent(obs1))
        p1 = self.battle.players[1]
        if slot != 0:
            card_name = p1.cycle[slot - 1]
            self.battle.deploy_card(1, card_name, Position(18-(x+0.5), 32-(y+0.5)))
            # Yes, this transformation seems weird, but it should be correct


    def step(self, action):
        """
        The action is a tuple with three values: (slot, y, x). When slot=0, no action is performed. Else deploy card on
        slot to the corresponding position on the arena.
        A decision is made every 30 frames (which is half a second). The reward is calculated by the damage dealt/taken,
        destroyed tower/lost tower and won game/lose game.
        The opponent is a function that takes in the observation and outputs the action tuple.
        """

        p0, p1 = self.battle.players
        blue_hps_old = p0.king_tower_hp+p0.left_tower_hp+p0.right_tower_hp
        red_hps_old = p1.king_tower_hp+p1.left_tower_hp+p1.right_tower_hp
        blue_left = 3-p0.get_crown_count()
        red_left = 3-p1.get_crown_count()

        slot, y, x = self.decode_action(action)
        if slot != 0:
            card_name = p0.cycle[slot-1]
            self.battle.deploy_card(0, card_name, Position(x+0.5, y+0.5))

        self.opponent_action()
        # only make decisions per half second
        for i in range(30):
            if self.battle.game_over:
                break
            for j in range(int(self.speed)):
                self.battle.step(1/60)
            if self.visualizer:
                self.visualizer.render_frame()
                time.sleep(1/60)
        blue_hps_new = p0.king_tower_hp+p0.left_tower_hp+p0.right_tower_hp
        red_hps_new = p1.king_tower_hp+p1.left_tower_hp+p1.right_tower_hp
        blue_left_new = 3-p0.get_crown_count()
        red_left_new = 3-p1.get_crown_count()

        reward = 3*(red_left-red_left_new)-3*(blue_left-blue_left_new)+0.001*(red_hps_old-red_hps_new)-0.0012*(blue_hps_old-blue_hps_new)
        if self.battle.game_over:
            #print('Battle over.', self.battle.winner, reward, p0.king_tower_hp, p0.left_tower_hp, p0.right_tower_hp,
            #      p1.king_tower_hp, p1.left_tower_hp, p1.right_tower_hp)
            if self.battle.winner == 0:
                reward += 10
            else:
                reward -= 10

        return self.observe(0), reward, self.battle.game_over, self.battle.game_over, {}


    def observe(self, player_id_observe=0):
        """Gives a representation of game state"""
        entity_list = []
        entity_features = []
        for _, each in self.battle.entities.items():
            if not each.is_alive: continue
            if each.name not in entity_names: continue
            # Previously I filtered out all projectiles thinking that they were useless;
            # But fireball and arrows are spells, so this makes the model blind to spells, which is a big mistake.
            # Now I only filter out those not in the known entity names list.
            entity_id = entity_names.index(each.name)
            card_type = card_types.index(each.data.type)
            player_id = int(each.player!=player_id_observe)
            # This makes the opponent observation more accurate
            elixir = each.data.elixir
            entity_list.append((entity_id, card_type, player_id))

            is_air = int(each.data.is_air_unit)
            attacks_ground, attacks_air = int(each.data.attack_ground), int(each.data.attack_air)

            speed = each.data.speed
            hp_left = np.log(each.hp) / 10 if each.hp != 0 else 0
            hp_percentage = each.hp / each.data.hp if each.data.hp != 0 else 0
            hit_speed = each.data.hit_speed
            attack_range = each.data.range / 3
            sight_range = each.data.sight_range / 3
            damage = each.data.damage / 200
            projectile_damage = each.data.projectile_data.damage / 200

            x, y = each.position.x/18, each.position.y/32
            if player_id == 1:
                x = 1-x
                y = 1-y
            obs_arr = (elixir, speed, is_air, attacks_ground, attacks_air, hp_left, hp_percentage,
                       hit_speed, attack_range, sight_range, damage, projectile_damage,
                       x, y)
            # Although cooldown is an important factor, I currently don't know how to observe that cooldown yet
            # in the real game, so I'll leave it for now.
            entity_features.append(obs_arr)

        hand = [entity_names.index(each) for each in self.battle.players[player_id_observe].cycle[:5]]
        elixir = self.battle.players[player_id_observe].elixir
        hand_mask = [Card(each).elixir <= elixir for each in self.battle.players[player_id_observe].cycle[:4]] + [False]
        clock = self.battle.time
        if clock < 120:
            phase = 1
        elif clock < 180:
            phase = 2
        elif clock < 240:
            phase = 3
        else:
            phase = 4
        normalized_clock = clock / 180
        crown_difference = self.battle.players[1-player_id_observe].get_crown_count()-self.battle.players[player_id_observe].get_crown_count()

        state = [normalized_clock, elixir/10, phase, crown_difference/3]

        entity_count = len(entity_list)
        if entity_count > self.max_entities:
            raise ValueError("Too many entities")
        entity_ids = np.zeros((self.max_entities, 3), dtype=np.int64)
        entity_ids[:entity_count] = np.asarray(
            [entity[:3] for entity in entity_list[:entity_count]],
            dtype=np.int64,
        )
        # The value might need to change later when the observation changes
        entity_features_array = np.zeros(
            (self.max_entities, self.feature_count),
            dtype=np.float32,
        )
        entity_features_array[:entity_count] = np.asarray(entity_features[:entity_count], dtype=np.float32)

        entity_mask = np.zeros(self.max_entities, dtype=np.bool_)
        entity_mask[:entity_count] = True

        return {
            "entity_ids": entity_ids,
            "entity_features": entity_features_array,
            "entity_mask": entity_mask,
            "hand": np.asarray(hand, dtype=np.int64),
            "hand_mask": np.asarray(hand_mask, dtype=np.bool_),
            "state": np.asarray(state, dtype=np.float32),
        }


def random_strategy(observation):
    slot = randint(0, 4)
    tile = randint(0, 32 * 18 - 1)
    return slot, tile

if __name__ == '__main__':
    env = CREnv(random_strategy)
    check_env(env)


