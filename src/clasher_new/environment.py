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
            "hand": gym.spaces.Box(low=0, high=len(entity_names) - 1, shape=(5,), dtype=np.int32),
            "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32)
        })
        self.action_space = gym.spaces.MultiDiscrete([5, 32, 18])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        shuffle(player_0_deck)
        shuffle(player_1_deck)
        self.battle = battle.BattleState(player.PlayerState(0, player_0_deck[:], 10),
                       player.PlayerState(1, player_1_deck[:], 10))
        # Now return initial observation
        return self.observe(0), {}

    def step(self, action):
        obs0, obs1 = self.observe(0), self.observe(1)
        p0, p1 = self.battle.players
        blue_hps_old = p0.king_tower_hp+p0.left_tower_hp+p0.right_tower_hp
        red_hps_old = p1.king_tower_hp+p1.left_tower_hp+p1.right_tower_hp
        blue_left = 3-p1.get_crown_count()
        red_left = 3-p0.get_crown_count() # reversed because it counts destroyed towers
        slot, x, y = action
        if slot != 0:
            card_name = p0.cycle[slot-1]
            self.battle.deploy_card(0, card_name, (x+0.5, y+0.5))
        opponent_action = self.opponent(obs1)
        slot, x, y = opponent_action
        if slot != 0:
            card_name = p1.cycle[slot - 1]
            self.battle.deploy_card(1, card_name, ((17-x)+0.5, (31-x)+0.5))
            # Yes, this transformation seems weird, but it should be correct
        # only make decisions per half second
        for i in range(30):
            if self.battle.game_over:
                break
            self.battle.step(1/60)
        blue_hps_new = p0.king_tower_hp+p0.left_tower_hp+p0.right_tower_hp
        red_hps_new = p1.king_tower_hp+p1.left_tower_hp+p1.right_tower_hp
        blue_left_new = 3-p1.get_crown_count()
        red_left_new = 3-p0.get_crown_count()

        reward = 5*(red_left-red_left_new)-5*(blue_left-blue_left_new)+0.02*(red_hps_old-red_hps_new)-0.02*(blue_hps_old-blue_hps_new)
        if self.battle.game_over:
            if self.battle.winner == 0:
                reward += 10
            else:
                reward -= 10
        return self.observe(0), reward, self.battle.game_over, self.battle.game_over, {}


    def observe(self, player_id_observe=0):
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
            if player_id == 1:
                x = 17-x
                y = 31-y
            obs_arr = np.array([entity_id, player_id, elixir, card_type, speed, is_air, attacks_ground, attacks_air,
                                hp_left, hp_percentage, hit_speed, attack_range, sight_range, damage, projectile_damage])
            obs[y][x] = obs_arr.copy()

        hand = [entity_names.index(each) for each in self.battle.players[player_id_observe].cycle[:5]]

        return {
            'grid': obs,
            'hand': hand,
            'elixir': np.array([self.battle.players[player_id_observe].elixir])
        }

if __name__ == '__main__':
    print(b.players[0].cycle)
