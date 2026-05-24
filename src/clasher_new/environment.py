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
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(32, 18, 15), dtype=np.float32),
            "hand": gym.spaces.Box(low=0, high=len(entity_names) - 1, shape=(5,), dtype=np.int32)
        })
        self.action_space = gym.spaces.MultiDiscrete([5, 32, 18])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        shuffle(player_0_deck)
        shuffle(player_1_deck)
        self.battle = battle.BattleState(player.PlayerState(0, player_0_deck[:], 10),
                       player.PlayerState(1, player_1_deck[:], 10))
        # Now return initial observation
        return self.observe(), {}

    def step(self, action):
        obs = self.observe()
        slot, x, y = action
        if slot != 0:
            card_name = self.battle.players[0].cycle[slot-1]
            self.battle.deploy_card(0, card_name, (x+0.5, y+0.5))
        opponent_action = self.opponent(obs)
        slot, x, y = opponent_action
        if slot != 0:
            card_name = self.battle.players[1].cycle[slot - 1]
            self.battle.deploy_card(1, card_name, (x + 0.5, y + 0.5))
        # only make decisions per half second
        for i in range(30):
            if self.battle.game_over:
                break
            self.battle.step(1/60)

        reward = 0
        return self.observe(), reward, self.battle.game_over, self.battle.game_over, {}


    def observe(self):
        """Gives a representation of game state"""
        obs = np.zeros((32, 18, 15))
        for id, each in self.battle.entities.items():
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

        hand = [entity_names.index(each) for each in self.battle.players[0].cycle[:5]]

        return {
            'grid': obs,
            'hand': hand,
            'elixir': self.battle.players[0].elixir
        }

if __name__ == '__main__':
    print(b.players[0].cycle)
