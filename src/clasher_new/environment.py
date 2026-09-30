import battle, player
from core import Position

import gymnasium as gym
from contextlib import contextmanager
from random import randint
import time
import numpy as np
import random

from stable_baselines3.common.env_checker import check_env

player_0_deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
player_1_deck = ['Minions', 'Archer', 'MiniPekka', 'Musketeer', 'Giant', 'Fireball', 'Arrows', 'Knight']
INITIAL_DECKS = (tuple(player_0_deck), tuple(player_1_deck))

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


class CardSaving:
    """An unaffordable card means wait, then redecide without deploying it."""

    def __init__(self):
        self.card = None

    def waiting(self, player):
        if self.card is not None and player.can_play_card(self.card):
            self.card = None
        return self.card is not None

    def resolve(self, player, action):
        slot = int(action[0])
        if slot and not player.can_play_card(player.cycle[slot - 1]):
            self.card = player.cycle[slot - 1]
            return (0, 0, 0)
        return action


class CREnv(gym.Env):
    def __init__(self, opponent_model=None, opponent_pool=None, visualize=False,
                 speed=1.0, discount_gamma=0.997, allow_saving=False):
        super().__init__()
        if not 0 < discount_gamma <= 1:
            raise ValueError("discount_gamma must be in (0, 1]")
        self.opponent = opponent_model
        self.opponent_pool = opponent_pool
        self.battle: battle.BattleState = None
        self.speed = speed
        self.discount_gamma = float(discount_gamma)
        self.allow_saving = bool(allow_saving)
        self.saving = CardSaving()
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8, 32, 18, 15), dtype=np.float32),
            "hand": gym.spaces.Box(low=0, high=len(entity_names) - 1, shape=(5,), dtype=np.int32),
            "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            "phase": gym.spaces.Discrete(4),
            "time_till_next_phase": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        self.action_space = gym.spaces.MultiDiscrete([5, 32, 18])

        self.visualize = visualize
        self.visualizer = None

        self.history = {0: [], 1: []}
        self.fps = 20

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.saving = CardSaving()
        decks = [self.np_random.permutation(cards).tolist() for cards in INITIAL_DECKS]
        opponent_seed = int(self.np_random.integers(2**32))
        self.opponent_random = random.Random(opponent_seed)
        self.opponent_numpy = np.random.RandomState(opponent_seed)
        if self.opponent_pool:
            self.opponent = self.opponent_pool[int(self.np_random.integers(len(self.opponent_pool)))]
        if callable(getattr(self.opponent, 'reset', None)):
            with self._opponent_rng():
                self.opponent.reset()
        self.battle = battle.BattleState(player.PlayerState(0, decks[0], 5.0),
                       player.PlayerState(1, decks[1], 5.0))
        if self.visualize:
            from new_visualization import Visualizer
            self.visualizer = Visualizer(self.battle)
        # Now return initial observation
        self.history = {0: [], 1: []}
        observation = self.observe(0)
        # Seed both frame stacks from the same initial battle state.
        self.observe(1)
        return observation, {}

    @contextmanager
    def _opponent_rng(self):
        """Isolate legacy scripts that use process-global Python/NumPy RNGs."""
        python_state, numpy_state = random.getstate(), np.random.get_state()
        random.setstate(self.opponent_random.getstate())
        np.random.set_state(self.opponent_numpy.get_state())
        try:
            yield
        finally:
            self.opponent_random.setstate(random.getstate())
            self.opponent_numpy.set_state(np.random.get_state())
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def opponent_action(self):
        obs1 = self.observe(1)
        with self._opponent_rng():
            opponent_action = self.opponent(obs1)
        slot, y, x = opponent_action
        p1 = self.battle.players[1]
        if slot != 0:
            card_name = p1.cycle[slot - 1]
            self.battle.deploy_card(1, card_name, Position(18-(x+0.5), 32-(y+0.5)))
            # Yes, this transformation seems weird, but it should be correct


    def step(self, action):
        p0 = self.battle.players[0]
        saving_card = None
        if getattr(self, "allow_saving", False):
            action = self.saving.resolve(p0, action)
            saving_card = self.saving.card
        observation, reward, terminated, truncated, info = self._step_once(action)
        elapsed_seconds = info["elapsed_seconds"]
        while not (terminated or truncated) and (
            (saving_card is not None and self.saving.waiting(p0))
            or not any(p0.can_play_card(card) for card in p0.cycle[:4])
        ):
            observation, next_reward, terminated, truncated, next_info = self._step_once((0, 0, 0))
            reward += self.discount_gamma ** (elapsed_seconds / 0.5) * next_reward
            elapsed_seconds += next_info["elapsed_seconds"]
        discount_steps = elapsed_seconds / 0.5
        info["elapsed_seconds"] = elapsed_seconds
        info["discount_steps"] = discount_steps
        info["transition_discount"] = self.discount_gamma ** discount_steps
        if saving_card is not None:
            info["saving_card"] = saving_card
        return observation, reward, terminated, truncated, info

    def _step_once(self, action):
        """
        The action is a tuple with three values: (slot, y, x). When slot=0, no action is performed. Else deploy card on
        slot to the corresponding position on the arena.
        Advance half a second. Reward combines the game outcome with
        elapsed-time potential shaping from tower count and HP advantage.
        The opponent is a function that takes in the observation and outputs the action tuple.
        """

        p0, p1 = self.battle.players
        time_before = self.battle.time
        blue_hps_old = p0.king_tower_hp+p0.left_tower_hp+p0.right_tower_hp
        red_hps_old = p1.king_tower_hp+p1.left_tower_hp+p1.right_tower_hp
        blue_left = 3-p0.get_crown_count()
        red_left = 3-p1.get_crown_count()

        slot, y, x = action
        if slot != 0:
            card_name = p0.cycle[slot-1]
            self.battle.deploy_card(0, card_name, Position(x+0.5, y+0.5))

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
        blue_hps_new = p0.king_tower_hp+p0.left_tower_hp+p0.right_tower_hp
        red_hps_new = p1.king_tower_hp+p1.left_tower_hp+p1.right_tower_hp
        blue_left_new = 3-p0.get_crown_count()
        red_left_new = 3-p1.get_crown_count()

        potential_before = (
            2 * (blue_left - red_left)
            + 0.0001 * (blue_hps_old - red_hps_old)
        )
        potential_after = (
            2 * (blue_left_new - red_left_new)
            + 0.0001 * (blue_hps_new - red_hps_new)
        )
        elapsed = self.battle.time - time_before
        discount = self.discount_gamma ** (elapsed / 0.5)
        if self.battle.game_over:
            potential_after = 0.0

        reward = discount * potential_after - potential_before
        if self.battle.game_over:
            reward += 10 if self.battle.winner == 0 else -10
        return self.observe(0), reward, self.battle.game_over, self.battle.game_over, {
            "elapsed_seconds": self.battle.time - time_before,
        }


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
            'time_till_next_phase': np.array([time_left/120.0], dtype=np.float32)
        }


def random_strategy(observation):
    slot = randint(0, 4)
    y = randint(0, 31)
    x = randint(0, 17)
    return slot, y, x

if __name__ == '__main__':
    env = CREnv(random_strategy, visualize=False)
    check_env(env)
