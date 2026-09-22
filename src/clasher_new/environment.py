import battle, player
from core import Position

import gymnasium as gym
from random import shuffle, randint
import time
import numpy as np
import random

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
    ACTION_HISTORY_LENGTH = 8
    ACTION_HISTORY_FEATURES = 6  # card_id, x, y, accepted, played, age

    def __init__(self, opponent_model=None, opponent_pool=None, visualize=False, speed=1.0,
                 reward_mode="potential", gamma=0.997, shaping_scale=1.0):
        super().__init__()
        if reward_mode not in ("legacy", "potential"):
            raise ValueError("reward_mode must be legacy or potential")
        if not 0 <= gamma <= 1 or not np.isfinite(shaping_scale) or shaping_scale < 0:
            raise ValueError("gamma must be in [0, 1]; shaping_scale must be finite and nonnegative")
        self.reward_mode, self.gamma, self.shaping_scale = reward_mode, gamma, shaping_scale
        self.opponent = opponent_model
        self.opponent_pool = opponent_pool
        self.battle: battle.BattleState = None
        self.speed = speed
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8, 32, 18, 15), dtype=np.float32),
            "hand": gym.spaces.Box(low=0, high=len(entity_names) - 1, shape=(5,), dtype=np.int32),
            "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            "phase": gym.spaces.Discrete(4),
            "time_till_next_phase": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "action_history": gym.spaces.Box(
                low=np.tile(
                    np.array([0.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    (self.ACTION_HISTORY_LENGTH, 2, 1),
                ),
                high=np.tile(
                    np.array([len(entity_names) - 1, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    (self.ACTION_HISTORY_LENGTH, 2, 1),
                ),
                shape=(self.ACTION_HISTORY_LENGTH, 2, self.ACTION_HISTORY_FEATURES),
                dtype=np.float32,
            ),
            "placement_mask": gym.spaces.Box(
                low=0.0, high=1.0, shape=(5, 32, 18), dtype=np.float32,
            ),
        })
        self.action_space = gym.spaces.MultiDiscrete([5, 32, 18])

        self.visualize = visualize
        self.visualizer = None

        self.history = {0: [], 1: []}
        self.action_history = []
        self.fps = 20

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        shuffle(player_0_deck)
        shuffle(player_1_deck)
        if self.opponent_pool:
            self.opponent = random.choice(self.opponent_pool)
        if callable(getattr(self.opponent, 'reset', None)):
            self.opponent.reset()
        self.battle = battle.BattleState(player.PlayerState(0, player_0_deck[:], 5.0),
                       player.PlayerState(1, player_1_deck[:], 5.0))
        self.initial_hp = np.maximum(self._reward_state()[0].sum(axis=1), 1.0)
        if self.visualize:
            from new_visualization import Visualizer
            self.visualizer = Visualizer(self.battle)
        # Now return initial observation
        self.history = {0: [], 1: []}
        self.action_history = []
        observation = self.observe(0)
        # Seed both frame stacks from the same initial battle state.
        self.observe(1)
        return observation, {}

    def _record_action(self, player_id, card_name, position, accepted):
        self.action_history.append({
            "player": player_id,
            "card_id": entity_names.index(card_name) if card_name in entity_names else 0,
            "x": float(position.x) if position is not None else 9.0,
            "y": float(position.y) if position is not None else 16.0,
            "accepted": float(accepted),
            "time": float(self.battle.time),
        })
        self.action_history = self.action_history[-2 * self.ACTION_HISTORY_LENGTH:]

    def _encode_action_history(self, player_id):
        encoded = np.zeros((self.ACTION_HISTORY_LENGTH, 2, self.ACTION_HISTORY_FEATURES), dtype=np.float32)
        streams = {0: [], 1: []}
        for event in self.action_history:
            stream = 0 if event["player"] == player_id else 1
            streams[stream].append(event)
        for stream, events in streams.items():
            for index, event in enumerate(events[-self.ACTION_HISTORY_LENGTH:]):
                x = np.clip((event["x"] / 17.0) * 2.0 - 1.0, -1.0, 1.0)
                y = np.clip((event["y"] / 31.0) * 2.0 - 1.0, -1.0, 1.0)
                if event["player"] != player_id:
                    x, y = -x, -y
                encoded[-len(events[-self.ACTION_HISTORY_LENGTH:]) + index, stream] = (
                    event["card_id"], x, y, event["accepted"],
                    float(event["card_id"] != 0),
                    np.clip((self.battle.time - event["time"]) / 4.0, 0.0, 1.0),
                )
        return encoded

    def opponent_action(self):
        obs1 = self.observe(1)
        opponent_action = self.opponent(obs1)
        slot, y, x = opponent_action
        p1 = self.battle.players[1]
        if slot != 0:
            card_name = p1.cycle[slot - 1]
            position = Position(18-(x+0.5), 32-(y+0.5))
            accepted = self.battle.deploy_card(1, card_name, position)
            self._record_action(1, card_name, Position(x+0.5, y+0.5), accepted)
        else:
            self._record_action(1, None, None, True)


    def step(self, action):
        """
        The action is a tuple with three values: (slot, y, x). When slot=0, no action is performed. Else deploy card on
        slot to the corresponding position on the arena.
        At speed=1, a decision advances 10 frames (half a second).
        Reward is the configured outcome/shaping objective.
        The opponent is a function that takes in the observation and outputs the action tuple.
        """

        p0 = self.battle.players[0]
        before = self._reward_state()

        slot, y, x = action
        accepted = True
        card_name = None
        if slot != 0:
            card_name = p0.cycle[slot-1]
            position = Position(x+0.5, y+0.5)
            accepted = self.battle.deploy_card(0, card_name, position)
            self._record_action(0, card_name, position, accepted)
        else:
            self._record_action(0, None, None, True)

        self.opponent_action()
        # only make decisions per half second
        for i in range(self.fps//2):
            if self.battle.game_over:
                break
            for j in range(int(self.speed)):
                self.battle.step(1/self.fps)
            if self.visualizer:
                self.visualizer.render_frame()
                time.sleep(1/self.fps)
        outcome, shaping = self._reward_components(before, self._reward_state())
        # King destruction, sudden death, and the game's tiebreak all terminate.
        return self.observe(0), outcome + shaping, self.battle.game_over, False, {
            "accepted": accepted,
            "card_name": card_name,
            "reward_outcome": outcome,
            "reward_shaping": shaping,
        }

    def _reward_state(self):
        return (np.array([(p.king_tower_hp, p.left_tower_hp, p.right_tower_hp)
                          for p in self.battle.players], dtype=np.float64),
                np.array([p.get_crown_count() for p in self.battle.players]))

    def _potential(self, state):
        hp, lost = state
        health = np.clip(np.maximum(hp, 0).sum(axis=1) / self.initial_hp, 0, 1)
        return float(0.5 * (health[0] - health[1] + (lost[1] - lost[0]) / 3))

    def _reward_components(self, before, after):
        outcome = (10.0 if self.battle.winner == 0 else -10.0) if self.battle.game_over else 0.0
        if self.reward_mode == "legacy":
            damage = (before[0] - after[0]).sum(axis=1)
            lost = after[1] - before[1]
            shaping = 5 * (lost[1] - lost[0]) + 0.001 * damage[1] - 0.0012 * damage[0]
        else:
            # Terminal Phi MUST be zero, including time-based game endings.
            following = 0.0 if self.battle.game_over else self._potential(after)
            shaping = self.shaping_scale * (self.gamma * following - self._potential(before))
        return outcome, float(shaping)


    def _placement_mask(self, player_id):
        mask = np.zeros((5, 32, 18), dtype=np.float32)
        mask[0] = 1.0  # no-op has a canonical placement
        state = self.battle.players[player_id]
        enemy = self.battle.players[1 - player_id]
        for slot, card_name in enumerate(state.cycle[:4], start=1):
            if not state.can_play_card(card_name):
                continue
            card = battle.Card(card_name)
            for y in range(32):
                for x in range(18):
                    local_position = Position(x + 0.5, y + 0.5)
                    position = local_position if player_id == 0 else Position(
                        18.0 - local_position.x, 32.0 - local_position.y
                    )
                    valid = True
                    if card.type != "spell":
                        if self.battle.is_position_occupied_by_building(position, 0):
                            valid = False
                        elif player_id == 0:
                            if position.y <= 1.0 and (position.x <= 6.0 or position.x > 12.0):
                                valid = False
                            elif position.y >= 21.0:
                                valid = False
                            elif position.y >= 15.0:
                                tower_hp = enemy.right_tower_hp if position.x > 9 else enemy.left_tower_hp
                                if tower_hp > 0:
                                    valid = False
                        else:
                            if position.y > 31.0 and (position.x <= 6.0 or position.x > 12.0):
                                valid = False
                            elif position.y <= 10.0:
                                valid = False
                            elif position.y <= 17.0:
                                tower_hp = enemy.right_tower_hp if position.x > 9 else enemy.left_tower_hp
                                if tower_hp > 0:
                                    valid = False
                    mask[slot, y, x] = float(valid)
        return mask

    def observe(self, player_id_observe=0):
        """Gives a representation of game state"""
        obs = np.zeros((32, 18, 15), dtype=np.float32)
        for id, each in self.battle.entities.items():
            if not each.is_alive: continue
            if each.name not in entity_names: continue
            entity_id = entity_names.index(each.name)
            card_type = card_types.index(each.data.type)
            player_id = each.player != player_id_observe # This way, own troops are always labeled as 0
            elixir = each.data.elixir
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

            x = int(np.clip(each.position.x, 0, 17))
            y = int(np.clip(each.position.y, 0, 31))

            if player_id_observe == 1:
                x = 17 - x
                y = 31 - y
            # sometimes, because of collision issues, x might be exactly 18 for player 0, which breaks the code
            # so it needs to be clipped

            obs_arr = np.array([entity_id, card_type, player_id, elixir, speed, is_air, attacks_ground, attacks_air,
                                hp_left, hp_percentage, hit_speed, attack_range, sight_range, damage, projectile_damage])
            obs[y][x] = obs_arr.copy()

        hand = np.array([entity_names.index(each) for each in self.battle.players[player_id_observe].cycle[:5]],
                        dtype=np.int32)
        battle_time = self.battle.time
        if battle_time < 120:
            phase = 1
            time_left = 120-battle_time
        elif battle_time < 180:
             phase = 2
             time_left = 180 - battle_time
        elif battle_time < 240:
            phase = 3
            time_left = 240 - battle_time
        else:
            phase = 4
            time_left = 300 - battle_time
        history = self.history[player_id_observe]
        if not history:
            history.extend(obs.copy() for _ in range(8))
        history.append(obs)
        if len(history) > 8:
            history.pop(0)
        return {
            'grid': np.stack(history),
            'hand': hand,
            'elixir': np.array([self.battle.players[player_id_observe].elixir], dtype=np.float32),
            'phase': phase-1,
            'time_till_next_phase': np.array([time_left/120.0], dtype=np.float32),
            'action_history': self._encode_action_history(player_id_observe),
            'placement_mask': self._placement_mask(player_id_observe),
        }


def random_strategy(observation):
    slot = randint(0, 4)
    y = randint(0, 31)
    x = randint(0, 17)
    return slot, y, x

if __name__ == '__main__':
    env = CREnv(random_strategy, visualize=False)
    check_env(env)
