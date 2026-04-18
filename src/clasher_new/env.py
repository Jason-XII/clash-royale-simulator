"""
clash_env.py — Gymnasium environment for Clash Royale RL training.
Place in src/clasher_new/ or ensure that directory is on sys.path.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from battle import BattleState, Troop, Building, Projectile, TimedExplosive
from player import PlayerState
from card_utils import Card, card_data
from core import Position
from arena import TileGrid

# ── Card & entity mappings ──────────────────────────────────────────
ALL_CARD_NAMES = sorted(card_data.keys())
CARD_TO_ID = {name: i + 1 for i, name in enumerate(ALL_CARD_NAMES)}  # 0 = empty
NUM_CARD_IDS = len(CARD_TO_ID) + 1

ARENA_W, ARENA_H = 18, 32

# Per-tile feature channels:
#  0  has_entity        (0/1)
#  1  is_friendly       (1=mine, 0=enemy)
#  2  card_id           (int, for embedding lookup)
#  3  entity_type       (1=troop, 2=building, 3=projectile, 4=timed_explosive)
#  4  hp_ratio          (current / max)
#  5  max_hp_norm       (max_hp / 5000)
#  6  deploy_delay      (seconds remaining)
#  7  hit_speed         (seconds per attack)
#  8  has_projectiles   (0/1)
#  9  is_area_damage    (0/1)
# 10  damage_norm       (damage / 500)
# 11  sight_range_norm  (sight_range / 10)
# 12  attack_range_norm (range / 10)
# 13  speed_norm        (speed / 5)
# 14  is_air            (0/1)
# 15  collision_radius  (tiles)
NUM_TILE_FEATURES = 16

# Tower max HPs (for normalization)
KING_MAX_HP = 4824.0
PRINCESS_MAX_HP = 3052.0


class ClashRoyaleEnv(gym.Env):
    """
    Gymnasium environment for Clash Royale self-play RL.

    Observations are always from the acting player's perspective
    (own side at bottom, enemy at top).

    Invalid actions (wrong zone, insufficient elixir, etc.) become no-ops.
    """

    metadata = {"render_modes": [None], "render_fps": 3}
    TICKS_PER_DECISION = 20  # 60 fps ÷ 3 decisions/sec

    def __init__(self, deck_p0, deck_p1, opponent_fn=None,
                 reward_shaping=1.0, render_mode=None):
        """
        Args:
            deck_p0:         list of 8 card names for player 0.
            deck_p1:         list of 8 card names for player 1.
            opponent_fn:     callable(obs) → action for player 1.
                             None = player 1 idles (use step_both for self-play).
            reward_shaping:  float in [0, 1].  Anneal toward 0 during training
                             so the agent converges on pure win/loss.
        """
        super().__init__()
        self.deck_p0 = list(deck_p0)
        self.deck_p1 = list(deck_p1)
        self.opponent_fn = opponent_fn
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "arena": spaces.Box(-np.inf, np.inf,
                                shape=(NUM_TILE_FEATURES, ARENA_H, ARENA_W),
                                dtype=np.float32),
            "global": spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32),
            "hand":   spaces.MultiDiscrete([NUM_CARD_IDS] * 5),  # 4 hand + next
        })

        # card 0-3 = play that hand slot, 4 = no-op
        self.action_space = spaces.Dict({
            "card":     spaces.Discrete(5),
            "position": spaces.Box(
                np.array([0.0, 0.0]),
                np.array([float(ARENA_W), float(ARENA_H)]),
                dtype=np.float32),
        })

        self.battle = None
        self._prev_hp = None

    # ── Gym API ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p0 = PlayerState(0, list(self.deck_p0), 5.0)
        p1 = PlayerState(1, list(self.deck_p1), 5.0)
        self.battle = BattleState(p0, p1)
        self.battle.update_player_hp()
        self._prev_hp = self._hp_snapshot()
        return self._get_obs(0), {}

    def step(self, action):
        """Single-agent step (player 0).  Opponent uses opponent_fn."""
        self._execute_action(0, action)
        if self.opponent_fn is not None:
            self._execute_action(1, self.opponent_fn(self._get_obs(1)))

        self._tick()
        reward = self._reward(0)
        return self._get_obs(0), reward, self.battle.game_over, False, self._info()

    def step_both(self, action_p0, action_p1):
        """Two-agent step for self-play.
        Returns (obs0, obs1, r0, r1, done, info)."""
        self._execute_action(0, action_p0)
        self._execute_action(1, action_p1)
        self._tick()
        r0 = self._reward(0)
        return (self._get_obs(0), self._get_obs(1),
                r0, -r0, self.battle.game_over, self._info())

    # ── Simulation ───────────────────────────────────────────────────

    def _tick(self):
        for _ in range(self.TICKS_PER_DECISION):
            self.battle.step(self.battle.dt)
            if self.battle.game_over:
                break
        self.battle.update_player_hp()          # keep HP in sync

    # ── Observation ──────────────────────────────────────────────────

    def _get_obs(self, pid):
        arena = np.zeros((NUM_TILE_FEATURES, ARENA_H, ARENA_W), np.float32)

        # One entity per tile — highest priority wins
        best = {}                               # (tx, ty) → (priority, entity)
        for e in self.battle.entities.values():
            if not e.is_alive:
                continue
            tx = int(np.clip(e.position.x, 0, ARENA_W - 1))
            ty = int(np.clip(e.position.y, 0, ARENA_H - 1))
            if pid == 1:                        # flip perspective
                tx, ty = ARENA_W - 1 - tx, ARENA_H - 1 - ty
            pri = self._pri(e)
            if (tx, ty) not in best or pri > best[(tx, ty)][0]:
                best[(tx, ty)] = (pri, e)

        for (tx, ty), (_, e) in best.items():
            arena[:, ty, tx] = self._tile_feat(e, pid)

        me  = self.battle.players[pid]
        opp = self.battle.players[1 - pid]
        glob = np.array([
            me.elixir / 10.0,
            max(me.king_tower_hp, 0)   / KING_MAX_HP,
            max(me.left_tower_hp, 0)   / PRINCESS_MAX_HP,
            max(me.right_tower_hp, 0)  / PRINCESS_MAX_HP,
            max(opp.king_tower_hp, 0)  / KING_MAX_HP,
            max(opp.left_tower_hp, 0)  / PRINCESS_MAX_HP,
            max(opp.right_tower_hp, 0) / PRINCESS_MAX_HP,
            self.battle.time / 300.0,
            float(self.battle.time >= 180),     # overtime flag
            float(self.battle.time >= 240),     # double-elixir-overtime flag
        ], np.float32)

        hand = [CARD_TO_ID.get(c, 0) for c in me.cycle[:5]]
        while len(hand) < 5:
            hand.append(0)

        return {"arena": arena,
                "global": np.asarray(glob, np.float32),
                "hand": np.asarray(hand, np.int64)}

    @staticmethod
    def _pri(e):
        if isinstance(e, Building):       return 4
        if isinstance(e, Troop):          return 3
        if isinstance(e, TimedExplosive): return 2
        return 1                           # Projectile / other

    @staticmethod
    def _tile_feat(e, observer):
        f = np.zeros(NUM_TILE_FEATURES, np.float32)
        f[0]  = 1.0
        f[1]  = 1.0 if e.player == observer else 0.0
        f[2]  = float(CARD_TO_ID.get(getattr(e, "card_name", e.data.name), 0))
        f[3]  = {Building: 2, Troop: 1, Projectile: 3,
                 TimedExplosive: 4}.get(type(e), 0)
        max_hp = max(e.data.hp, 1)
        f[4]  = np.clip(e.hp / max_hp, 0, 1)
        f[5]  = max_hp / 5000.0
        f[6]  = getattr(e, "deploy_delay_remaining", 0.0)
        f[7]  = e.data.hit_speed if e.data.hit_speed else 0.0
        f[8]  = float(bool(e.data.projectiles))
        f[9]  = float(e.data.area_damage_radius > 0)
        f[10] = e.data.damage / 500.0
        f[11] = e.data.sight_range / 10.0
        f[12] = e.data.range / 10.0
        f[13] = e.speed / 5.0
        f[14] = float(e.data.is_air_unit)
        f[15] = e.data.collision_radius
        return f

    # ── Actions ──────────────────────────────────────────────────────

    def _execute_action(self, pid, action):
        idx = int(action["card"])
        if idx >= 4:
            return                              # no-op

        hand = self.battle.players[pid].cycle[:4]
        if idx >= len(hand):
            return

        x, y = float(action["position"][0]), float(action["position"][1])
        if pid == 1:                            # un-flip from agent's view
            x, y = ARENA_W - x, ARENA_H - y

        # Snap to tile centre
        tx = int(np.clip(x, 0, ARENA_W - 1)) + 0.5
        ty = int(np.clip(y, 0, ARENA_H - 1)) + 0.5
        pos = Position(tx, ty)

        # Deploy-zone check
        zones = self.battle.arena.get_deploy_zones(pid, self.battle)
        if not any(x1 <= pos.x <= x2 and y1 <= pos.y <= y2
                   for x1, y1, x2, y2 in zones):
            return                              # invalid → no-op

        self.battle.deploy_card(pid, hand[idx], pos)

    # ── Rewards ──────────────────────────────────────────────────────

    def _hp_snapshot(self):
        p = self.battle.players
        return tuple(
            (max(pl.king_tower_hp, 0),
             max(pl.left_tower_hp, 0),
             max(pl.right_tower_hp, 0))
            for pl in p
        )

    def _reward(self, pid):
        curr = self._hp_snapshot()
        prev = self._prev_hp
        self._prev_hp = curr

        opp = 1 - pid
        r = 0.0

        # ---- dominant win / loss signal ----
        if self.battle.game_over:
            if   self.battle.winner == pid: r += 10.0
            elif self.battle.winner == opp: r -= 10.0

        # ---- shaped (annealed) component ----
        if self.reward_shaping > 0:
            s = 0.0
            maxes = (KING_MAX_HP, PRINCESS_MAX_HP, PRINCESS_MAX_HP)

            for i, mx in enumerate(maxes):
                # damage dealt to opponent towers
                s += (prev[opp][i] - curr[opp][i]) / mx
                # damage taken on own towers
                s -= (prev[pid][i] - curr[pid][i]) / mx

            # crown bonuses
            crowns_gained = sum(int(curr[opp][i] <= 0) - int(prev[opp][i] <= 0)
                                for i in range(3))
            crowns_lost   = sum(int(curr[pid][i] <= 0) - int(prev[pid][i] <= 0)
                                for i in range(3))
            s += 2.0 * crowns_gained
            s -= 2.0 * crowns_lost

            r += self.reward_shaping * s

        return r

    # ── Helpers ──────────────────────────────────────────────────────

    def _info(self):
        return {"time": self.battle.time, "winner": self.battle.winner}

    @staticmethod
    def no_op():
        """Return a valid no-op action."""
        return {"card": 4, "position": np.array([0.0, 0.0], np.float32)}


class SelfPlayEnv:
    """
    Thin wrapper for two-agent self-play.

        env = SelfPlayEnv(deck0, deck1)
        obs0, obs1, info = env.reset()
        while not done:
            obs0, obs1, r0, r1, done, info = env.step(a0, a1)
    """
    def __init__(self, deck_p0, deck_p1, reward_shaping=1.0):
        self.env = ClashRoyaleEnv(deck_p0, deck_p1, reward_shaping=reward_shaping)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kw):
        obs0, info = self.env.reset(**kw)
        obs1 = self.env._get_obs(1)
        return obs0, obs1, info

    def step(self, action_p0, action_p1):
        return self.env.step_both(action_p0, action_p1)
