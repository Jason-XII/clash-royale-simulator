"""Gymnasium environment: player 0 is the learner, player 1 is `opponent_model`.

Both players see the arena from their own side (own towers near y=0), so the same
policy or script can play either side. Actions are (slot, y, x): slot 0 waits
half a second, slots 1-4 play that hand card at tile (y, x), and the remaining
slots bank elixir (see `Bank`).
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
CARD_SLOTS = 4
BANK_TARGETS = (4, 5, 6, 7, 8, 9, 10)  # slot 5 + i banks until elixir >= BANK_TARGETS[i]
DECISION_SECONDS = 0.5


def to_world(player_id, y, x):
    """Center of tile (y, x) in `player_id`'s own view, in arena coordinates."""
    if player_id == 0:
        return Position(x + 0.5, y + 0.5)
    return Position(18 - (x + 0.5), 32 - (y + 0.5))


def enemy_troops_in_half(state, player_id):
    """Ids of enemy troops in `player_id`'s half of a simulated battle (y < 16 is player 0's)."""
    return {key for key, entity in state.entities.items()
            if isinstance(entity, battle.Troop) and entity.is_alive
            and entity.player != player_id
            and (entity.position.y < 16) == (player_id == 0)}


class Bank:
    """A bank action: keep waiting until elixir reaches the target.

    Ends early when a new enemy troop crosses into this player's half, so the
    player gets to respond. Enemies already there when banking began (the player
    saw them when choosing to bank) do not interrupt it.

    `intruders` is a function returning the ids of enemy troops in the player's
    half: `enemy_troops_in_half` in the simulator, the live client's own in-game.
    It is only called while banking.
    """

    def __init__(self):
        self.target = None

    def resolve(self, action, intruders):
        """Start banking if `action` is a bank action; return what to do right now."""
        slot = int(action[0])
        if slot <= CARD_SLOTS:
            return action
        self.target = BANK_TARGETS[slot - CARD_SLOTS - 1]
        self.seen = intruders()
        return (0, 0, 0)

    def active(self, elixir, intruders):
        """Whether to keep waiting. Clears itself once the bank ends."""
        if self.target is not None and (elixir >= self.target or not intruders() <= self.seen):
            self.target = None
        return self.target is not None


LIVE_PLAY_DELAY = (0.8, 1.0)  # seconds between a tap and the card landing in the live game (hand-timed)


class CREnv(gym.Env):
    def __init__(self, opponent_model=None, visualize=False, speed=1.0, discount_gamma=0.997,
                 play_delay=(0.0, 0.0)):
        """`play_delay` (low, high): each card lands a uniform random number of seconds
        after it is played, like the 1-2 s it takes in the live game. Off by default."""
        super().__init__()
        self.play_delay = play_delay
        if not 0 < discount_gamma <= 1:
            raise ValueError("discount_gamma must be in (0, 1]")
        self.opponent = opponent_model
        self.battle: battle.BattleState = None
        self.speed = speed
        self.discount_gamma = float(discount_gamma)
        self.bank = Bank()
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(FRAMES, 32, 18, 15), dtype=np.float32),
            "hand": gym.spaces.Box(low=0, high=len(entity_names) - 1, shape=(5,), dtype=np.int32),
            "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            "phase": gym.spaces.Discrete(4),
            "time_till_next_phase": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            # Memory inputs; checkpoints from before them never see these keys.
            "queue": gym.spaces.Box(low=0, high=len(entity_names) - 1, shape=(3,), dtype=np.int32),
            "opponent_elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            "decision_gap": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        })
        self.action_space = gym.spaces.MultiDiscrete([1 + CARD_SLOTS + len(BANK_TARGETS), 32, 18])
        self.visualize = visualize
        self.visualizer = None
        self.history = {0: [], 1: []}
        self.tile_cache = {}  # legal troop tiles, see legal_mask
        self.fps = 20

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.bank = Bank()
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
            low, high = self.play_delay
            delay = self.np_random.uniform(low, high) if high > 0 else 0.0
            self.battle.deploy_card(player_id, card, to_world(player_id, y, x), delay)

    def opponent_action(self):
        """Player 1's action, decided from the same moment player 0 decided from.

        Not RNG-isolated (only reset() is): current opponents draw global
        randomness only in reset(). Wrap this in _opponent_rng() for one that doesn't.
        """
        return self.opponent(self.opponent_observation)

    def step(self, action):
        """One decision. Skips ahead (and sums discounted rewards) while banking,
        and while no card is playable."""
        p0 = self.battle.players[0]
        intruders = lambda: enemy_troops_in_half(self.battle, 0)
        action = self.bank.resolve(action, intruders)
        bank_target = self.bank.target
        observation, reward, done, _, info = self._step_once(action)
        elapsed = info["elapsed_seconds"]
        while not done and (self.bank.active(p0.elixir, intruders)
                            or not any(p0.can_play_card(card) for card in p0.cycle[:4])):
            observation, next_reward, done, _, next_info = self._step_once((0, 0, 0))
            reward += self.discount_gamma ** (elapsed / DECISION_SECONDS) * next_reward
            elapsed += next_info["elapsed_seconds"]
        info["elapsed_seconds"] = elapsed
        observation["decision_gap"] = np.array([elapsed], dtype=np.float32)
        info["discount_steps"] = elapsed / DECISION_SECONDS
        info["transition_discount"] = self.discount_gamma ** info["discount_steps"]
        if bank_target is not None:
            info["bank_target"] = bank_target
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
            # Mirror the exact position for player 1 before taking the tile, so both
            # players see the same tiles (kings sit exactly on x=9, y=3/29).
            px, py = entity.position.x, entity.position.y
            if player_id == 1:
                px, py = 18 - px, 32 - py
            # Clip: collisions can push a unit to exactly x=18 or y=32.
            x = int(min(max(px, 0), 17))
            y = int(min(max(py, 0), 31))
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
            # The rest of the own cycle after `hand`, and the opponent's exact elixir
            # (the live client reads it from game memory).
            'queue': np.array([ENTITY_ID[card] for card in self.battle.players[player_id].cycle[5:8]],
                              dtype=np.int32),
            'opponent_elixir': np.array([self.battle.players[1 - player_id].elixir], dtype=np.float32),
            # Seconds since this player's previous decision. Set by whoever decides:
            # step() for the learner, the actor itself for player 1.
            'decision_gap': np.zeros(1, dtype=np.float32),
        }
