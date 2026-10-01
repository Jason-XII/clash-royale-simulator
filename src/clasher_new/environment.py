"""Gymnasium environment: player 0 is the learner, player 1 is `opponent_model`.

Both players see the arena from their own side (own towers near y=0), so the same
policy or script can play either side. Actions are (slot, y, x); slot 0 waits.
"""
from contextlib import contextmanager
import random
import time

import gymnasium as gym
import numpy as np

import battle
import player
from card_utils import Card
from core import Position

player_0_deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
player_1_deck = ['Minions', 'Archer', 'MiniPekka', 'Musketeer', 'Giant', 'Fireball', 'Arrows', 'Knight']
INITIAL_DECKS = (tuple(player_0_deck), tuple(player_1_deck))

# Index 0 means an empty tile. Spells appear on the board as their projectiles.
entity_names = ['None', 'Knight', 'MiniPekka', 'Arrows', 'Minions', 'Archer',
                'Musketeer', 'Fireball', 'Giant', 'King_PrincessTowers',
                'KingTower', 'ArrowsSpell', 'FireballSpell']
# 'troop' is the towers' type ("tower troop"); actual troops are 'character'.
card_types = ['troop', 'character', 'spell', 'building']
ENTITY_ID = {name: i for i, name in enumerate(entity_names)}
CARD_TYPE_ID = {name: i for i, name in enumerate(card_types)}

FRAMES = 8            # observation history length, one frame per half second
DECISION_SECONDS = 0.5


def to_world(player_id, y, x):
    """Center of tile (y, x) in `player_id`'s own view, in arena coordinates."""
    if player_id == 0:
        return Position(x + 0.5, y + 0.5)
    return Position(18 - (x + 0.5), 32 - (y + 0.5))


class CardSaving:
    """Choosing an unaffordable card waits until it is affordable, then redecides."""

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
    def __init__(self, opponent_model=None, visualize=False, speed=1.0,
                 discount_gamma=0.997, allow_saving=False):
        super().__init__()
        if not 0 < discount_gamma <= 1:
            raise ValueError("discount_gamma must be in (0, 1]")
        self.opponent = opponent_model
        self.battle: battle.BattleState = None
        self.speed = speed
        self.discount_gamma = float(discount_gamma)
        self.allow_saving = bool(allow_saving)
        self.saving = CardSaving()
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(FRAMES, 32, 18, 15), dtype=np.float32),
            "hand": gym.spaces.Box(low=0, high=len(entity_names) - 1, shape=(5,), dtype=np.int32),
            "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            "phase": gym.spaces.Discrete(4),
            "time_till_next_phase": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        self.action_space = gym.spaces.MultiDiscrete([5, 32, 18])
        self.visualize = visualize
        self.visualizer = None
        self.history = {0: [], 1: []}
        self.tile_cache = {}  # legal troop tiles, see legal_mask
        self.fps = 20

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.saving = CardSaving()
        decks = [self.np_random.permutation(cards).tolist() for cards in INITIAL_DECKS]
        opponent_seed = int(self.np_random.integers(2**32))
        self.opponent_random = random.Random(opponent_seed)
        self.opponent_numpy = np.random.RandomState(opponent_seed)
        if callable(getattr(self.opponent, 'reset', None)):
            with self._opponent_rng():
                self.opponent.reset()
        self.battle = battle.BattleState(player.PlayerState(0, decks[0], 5.0),
                                         player.PlayerState(1, decks[1], 5.0))
        if self.visualize:
            from new_visualization import Visualizer
            self.visualizer = Visualizer(self.battle)
        self.history = {0: [], 1: []}
        observation = self.observe(0)
        self.opponent_observation = self.observe(1)
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

    def legal_mask(self, player_id):
        """(4, 32, 18) mask of legal deployments, in `player_id`'s own view."""
        player = self.battle.players[player_id]
        mask = np.zeros((4, 32, 18), dtype=np.int8)
        playable = [player.can_play_card(card) for card in player.cycle[:4]]
        if not any(playable):
            return mask
        # can_place_troop depends only on these, so reuse the tiles until one changes.
        enemy = self.battle.players[1 - player_id]
        key = (player_id, enemy.left_tower_hp > 0, enemy.right_tower_hp > 0,
               tuple(self.battle.building_positions))
        if key not in self.tile_cache:
            self.tile_cache[key] = np.array([[self.battle.can_place_troop(player_id, to_world(player_id, y, x))
                                              for x in range(18)] for y in range(32)], dtype=np.int8)
        tiles = self.tile_cache[key]
        for slot, card in enumerate(player.cycle[:4]):
            if playable[slot]:
                mask[slot] = 1 if Card(card).type == "spell" else tiles
        return mask

    def _deploy(self, player_id, action):
        slot, y, x = action
        if slot:
            card = self.battle.players[player_id].cycle[slot - 1]
            self.battle.deploy_card(player_id, card, to_world(player_id, y, x))

    def opponent_action(self):
        """Player 1's action, decided from the same moment player 0 decided from.

        Not RNG-isolated (only reset() is): current opponents draw global
        randomness only in reset(). Wrap this in _opponent_rng() for one that doesn't.
        """
        return self.opponent(self.opponent_observation)

    def step(self, action):
        """One decision. Skips ahead (and sums discounted rewards) while no card is
        playable, or while a saved card is still unaffordable."""
        p0 = self.battle.players[0]
        saving_card = None
        if self.allow_saving:
            action = self.saving.resolve(p0, action)
            saving_card = self.saving.card
        observation, reward, done, _, info = self._step_once(action)
        elapsed = info["elapsed_seconds"]
        while not done and ((saving_card is not None and self.saving.waiting(p0))
                            or not any(p0.can_play_card(card) for card in p0.cycle[:4])):
            observation, next_reward, done, _, next_info = self._step_once((0, 0, 0))
            reward += self.discount_gamma ** (elapsed / DECISION_SECONDS) * next_reward
            elapsed += next_info["elapsed_seconds"]
        info["elapsed_seconds"] = elapsed
        info["discount_steps"] = elapsed / DECISION_SECONDS
        info["transition_discount"] = self.discount_gamma ** info["discount_steps"]
        if saving_card is not None:
            info["saving_card"] = saving_card
        return observation, reward, done, done, info

    def _potential(self):
        """Shaping potential: 2 per tower advantage plus 0.0001 per HP advantage."""
        p0, p1 = self.battle.players
        towers = (3 - p0.get_crown_count()) - (3 - p1.get_crown_count())
        hp = (p0.king_tower_hp + p0.left_tower_hp + p0.right_tower_hp) \
            - (p1.king_tower_hp + p1.left_tower_hp + p1.right_tower_hp)
        return 2 * towers + 0.0001 * hp

    def _step_once(self, action):
        """Apply both players' actions and simulate half a second.

        Both players decide before either deployment lands, so neither can react
        to the other's card within the same step.
        Reward is outcome (+/-10) plus potential shaping with an elapsed-time
        discount; the terminal potential is zero.
        """
        time_before = self.battle.time
        potential_before = self._potential()
        opponent_action = self.opponent_action()
        self._deploy(0, action)
        self._deploy(1, opponent_action)
        for _ in range(int(self.fps * DECISION_SECONDS)):
            if self.battle.game_over:
                break
            for _ in range(int(self.speed)):
                self.battle.step(1 / self.fps)
            if self.visualizer:
                self.visualizer.render_frame()
                time.sleep(1 / self.fps)
        elapsed = self.battle.time - time_before
        done = self.battle.game_over
        potential_after = 0.0 if done else self._potential()
        reward = self.discount_gamma ** (elapsed / DECISION_SECONDS) * potential_after - potential_before
        if done:
            reward += 10 if self.battle.winner == 0 else -10
        observation = self.observe(0)
        self.opponent_observation = self.observe(1)
        return observation, reward, done, done, {"elapsed_seconds": elapsed}

    def observe(self, player_id=0):
        """Append the current frame to `player_id`'s history and return its observation."""
        frame = np.zeros((32, 18, 15), dtype=np.float32)
        for entity in self.battle.entities.values():
            if not entity.is_alive or entity.name not in ENTITY_ID:
                continue
            data = entity.data
            # Clip: collisions can push a unit to exactly x=18 or y=32.
            x = int(min(max(entity.position.x, 0), 17))
            y = int(min(max(entity.position.y, 0), 31))
            if player_id == 1:
                x, y = 17 - x, 31 - y
            # ponytail: one entity per tile; later entities overwrite earlier ones.
            frame[y][x] = np.array([
                ENTITY_ID[entity.name], CARD_TYPE_ID[data.type],
                entity.player != player_id,  # 0 = own, 1 = enemy
                data.elixir, data.speed, int(data.is_air_unit),
                int(data.attack_ground), int(data.attack_air),
                np.log(entity.hp) / 10 if entity.hp != 0 else 0,
                entity.hp / data.hp if data.hp != 0 else 0,
                data.hit_speed, data.range / 3, data.sight_range / 3,
                data.damage / 200, data.projectile_data.damage / 200,
            ])

        history = self.history[player_id]
        if not history:
            history.extend(frame.copy() for _ in range(FRAMES))
        history.append(frame)
        if len(history) > FRAMES:
            history.pop(0)

        now = self.battle.time
        phase_end = next(end for end in (120, 180, 240, 300) if now < end or end == 300)
        return {
            'grid': np.stack(history),
            'hand': np.array([ENTITY_ID[card] for card in self.battle.players[player_id].cycle[:5]],
                             dtype=np.int32),
            'elixir': np.array([self.battle.players[player_id].elixir], dtype=np.float32),
            'phase': (120, 180, 240, 300).index(phase_end),
            'time_till_next_phase': np.array([(phase_end - now) / 120.0], dtype=np.float32),
        }
