from clasher_new import battle, player

import gymnasium as gym
from random import shuffle
import numpy as np

player_0_deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
player_1_deck = ['Minions', 'Archer', 'MiniPekka', 'Musketeer', 'Giant', 'Fireball', 'Arrows', 'Knight']

b = battle.BattleState(player.PlayerState(0, player_0_deck, 10),
                       player.PlayerState(1, player_1_deck, 10))

deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']

entity_names = ['None', 'Knight', 'MiniPekka', 'ArrowsSpell', 'Minions', 'Archer',
                'Musketeer', 'FireballSpell', 'Giant', 'King_PrincessTowers',
                'KingTower']

card_types = ['troop', 'spell', 'building']
speed_types = [0, 0.75, 1.0, 1.5]


class CREnv(gym.Env):
    def __init__(self, opponent_model=None):
        super().__init__()
        self.opponent = opponent_model
        self.battle: battle.BattleState = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        shuffle(player_0_deck)
        shuffle(player_1_deck)
        self.battle = battle.BattleState(player.PlayerState(0, player_0_deck[:], 10),
                       player.PlayerState(1, player_1_deck[:], 10))
        # Now return initial observation
        pass

    def observe(self):
        """Gives a representation of game state"""
        obs = np.zeros((32, 18, 15))
        for each in self.battle.entities.items():
            entity_id = entity_names.index(each.name)
            player_id = each.player
            elixir = each.data.elixir - 3
            card_type = card_types.index(each.data.type)
            speed = speed_types.index(each.data.speed)
            is_air = int(each.data.is_air_unit)
            attacks_ground, attacks_air = int(each.data.attack_ground), int(each.data.attack_air)

            hp_left = np.log(each.hp) / 10
            hp_percentage = each.hp / each.data.hp
            hit_speed = each.data.hit_speed
            attack_range = each.data.range / 3
            sight_range = each.data.sight_range / 3
            damage = each.data.damage / 200
            projectile_damage = each.data.projectile_data.get('damage', 0) / 200

            x, y = int(each.position.x), int(each.position.y)
            obs_arr = np.array([entity_id, player_id, elixir, card_type, speed, is_air, attacks_ground, attacks_air,
                                hp_left, hp_percentage, hit_speed, attack_range, sight_range, damage, projectile_damage])
            obs[y][x] = obs_arr.copy()
        return obs


