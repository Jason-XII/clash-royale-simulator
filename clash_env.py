"""Gymnasium wrapper for the Clash Royale simulator.

Observation:
    Dict with:
        "spatial" (6, 32, 18) float32 — channels:
            0: own troop HP (normalised)
            1: enemy troop HP (normalised)
            2: own troop is_air flag
            3: enemy troop is_air flag
            4: own building HP (normalised)
            5: enemy building HP (normalised)
        "flat" (FLAT_DIM,) float32 —
            elixir(1) + hand cards(4×CARD_FEAT) + next_card(CARD_FEAT)
            + own towers(3) + enemy towers(3) + game_time(1)

Action (Discrete 161):
    0 = do nothing
    1..160 = play hand slot (0-3) at coarse grid position (0-39)
    Grid: 5 x-positions × 8 y-positions over the full arena.
    Illegal placements (troop on enemy half, not enough elixir) → treated as do nothing.

Reward shaping (per step):
    ± 1× normalised troop damage dealt / received
    ± 4× normalised tower damage dealt / received
    ± 2.0 per tower destroyed / lost
    ± 10.0 on win / loss (king tower destroyed)
    ± 5.0 tiebreaker at timeout
"""

import sys, os, random, math, io, contextlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from clasher.battle import BattleState
from clasher.arena import Position
from clasher.entities import Troop, Building, Entity
from clasher.data import CardDataLoader
from clasher.card_aliases import resolve_card_name
from clasher.spells import SPELL_REGISTRY


# --------------------------------------------------------------------------
# Default decks
# --------------------------------------------------------------------------
LOG_BAIT = [
    "GoblinBarrel", "GoblinGang", "IceSpirits",
    "InfernoTower", "Knight", "Princess", "Rocket", "Log",
]
ROYAL_HOGS = [
    "RoyalRecruits", "RoyalHogs", "DartBarrell", "Fireball",
    "Zap", "GoblinCage", "BarbLog", "Heal",
]


# --------------------------------------------------------------------------
# Coarse action grid: 5 x-positions × 8 y-positions = 40 cells
# --------------------------------------------------------------------------
GRID_XS = [2.0, 5.5, 9.0, 12.5, 16.0]                             # 5
GRID_YS = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0]         # 8
GRID_POSITIONS = [(x, y) for y in GRID_YS for x in GRID_XS]        # 40
NUM_POSITIONS = len(GRID_POSITIONS)                                 # 40
NUM_HAND_CARDS = 4
NUM_ACTIONS = 1 + NUM_HAND_CARDS * NUM_POSITIONS                    # 161

# Player 0 own-half y threshold (troops must be placed on own half)
P0_MAX_DEPLOY_Y = 14.5


# --------------------------------------------------------------------------
# Spatial observation constants
# --------------------------------------------------------------------------
ARENA_W, ARENA_H = 18, 32
SPATIAL_CHANNELS = 6

# Card feature vector
# [elixir, hp, damage, speed, is_spell, is_air, is_splash, is_building]
CARD_FEAT_DIM = 8
FLAT_DIM = 1 + NUM_HAND_CARDS * CARD_FEAT_DIM + CARD_FEAT_DIM + 3 + 3 + 1  # 48

# Normalisation
MAX_HP = 10000.0
TOWER_KING_HP = 4824.0
TOWER_PRINCESS_HP = 3631.0
MAX_ELIXIR = 10.0
MAX_TIME = 360.0
MAX_SPEED = 480.0
TROOP_DMG_NORM = 2000.0   # normalise per-step troop damage
TOWER_DMG_NORM = 5000.0   # normalise per-step tower damage


def _silence(fn, *a, **kw):
    """Call *fn* suppressing stdout prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# --------------------------------------------------------------------------
# Card feature helpers
# --------------------------------------------------------------------------
def _card_feature(card_stats) -> np.ndarray:
    """Fixed-length feature vector for one card."""
    if card_stats is None:
        return np.zeros(CARD_FEAT_DIM, dtype=np.float32)

    elixir = (card_stats.mana_cost or 0) / MAX_ELIXIR
    hp = (card_stats.hitpoints or 0) / MAX_HP
    dmg = (card_stats.damage or 0) / 1000.0
    speed = (card_stats.speed or 0) / MAX_SPEED

    card_type = (card_stats.card_type or "").lower()
    is_spell = 1.0 if card_type == "spell" else 0.0
    is_building = 1.0 if card_type == "building" else 0.0

    raw_char = card_stats.summon_character_data or {}
    is_air = 1.0 if raw_char.get("flyingLevel") else 0.0
    area_r = raw_char.get("areaDamageRadius") or raw_char.get("areaEffectRadius")
    is_splash = 1.0 if area_r else 0.0

    return np.array(
        [elixir, hp, dmg, speed, is_spell, is_air, is_splash, is_building],
        dtype=np.float32,
    )


def _precompute_deck_features(deck: list[str], loader: CardDataLoader) -> dict[str, np.ndarray]:
    """Return {card_name: feature_vector} for every card in deck."""
    defs = loader.load_card_definitions()
    out = {}
    for name in deck:
        resolved = resolve_card_name(name, defs)
        stats = loader.get_card(resolved)
        out[name] = _card_feature(stats)
    return out


# ==========================================================================
# Environment
# ==========================================================================
class ClashRoyaleEnv(gym.Env):
    """Single-agent Clash Royale env.
    Player 0 (blue/bottom) = RL agent.
    Player 1 (red/top) = opponent (random by default).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_deck: list[str] | None = None,
        opponent_deck: list[str] | None = None,
        opponent_fn=None,
        ticks_per_step: int = 10,       # ~0.33 s game-time per RL step
        max_game_ticks: int = 10800,     # 360 s at 30 fps
        data_file: str = "gamedata.json",
        seed: int | None = None,
    ):
        super().__init__()

        self.agent_deck = agent_deck or LOG_BAIT[:]
        self.opponent_deck = opponent_deck or ROYAL_HOGS[:]
        self.opponent_fn = opponent_fn or self._random_opponent
        self.ticks_per_step = ticks_per_step
        self.max_game_ticks = max_game_ticks

        # Shared card loader
        self._loader = CardDataLoader(data_file)
        _silence(self._loader.load_cards)
        _silence(self._loader.load_card_definitions)

        # Pre-compute card features
        self._agent_feats = _precompute_deck_features(self.agent_deck, self._loader)
        self._opp_feats = _precompute_deck_features(self.opponent_deck, self._loader)

        # Resolve which agent cards are spells (can deploy anywhere)
        defs = self._loader.load_card_definitions()
        self._agent_is_spell = {}
        for name in self.agent_deck:
            resolved = resolve_card_name(name, defs)
            self._agent_is_spell[name] = resolved in SPELL_REGISTRY

        # Spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict({
            "spatial": spaces.Box(0.0, 1.0,
                                  shape=(SPATIAL_CHANNELS, ARENA_H, ARENA_W),
                                  dtype=np.float32),
            "flat": spaces.Box(-1.0, 1.0, shape=(FLAT_DIM,), dtype=np.float32),
        })

        self._battle: BattleState | None = None
        self._rng = random.Random(seed)

        # Tracking for reward shaping
        self._prev_tower_hp = None    # {pid: (king, left, right)}
        self._prev_troop_hp = None    # {pid: total_troop_hp}
        self._prev_towers_alive = None  # {pid: count}

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        self._battle = _silence(BattleState)
        self._battle.players[0].elixir = 7.0
        self._battle.players[1].elixir = 7.0
        # Assign decks
        for pid, deck in enumerate([self.agent_deck, self.opponent_deck]):
            p = self._battle.players[pid]
            shuffled = deck[:]
            self._rng.shuffle(shuffled)
            p.deck = deck[:]
            p.hand = shuffled[:4]
            p.cycle_queue.clear()
            p.cycle_queue.extend(shuffled[4:])

        # Init tracking
        self._prev_tower_hp = self._get_tower_hps()
        self._prev_troop_hp = self._get_total_troop_hp()
        self._prev_towers_alive = {pid: self._count_alive_towers(pid) for pid in (0, 1)}

        return self._obs(), {}

    def step(self, action: int):
        assert self._battle is not None, "Call reset() first"
        b = self._battle
        self._spell_deployed = False
        self._spell_cost = 0
        if action > 0:
            action -= 1
            card_slot = action // NUM_POSITIONS
            pos_idx = action % NUM_POSITIONS
            if card_slot < len(b.players[0].hand):
                card_name = b.players[0].hand[card_slot]
                gx, gy = GRID_POSITIONS[pos_idx]
                is_spell = self._agent_is_spell.get(card_name, False)
                if is_spell or gy <= P0_MAX_DEPLOY_Y:
                    if is_spell:
                        defs = b.card_loader.load_card_definitions()
                        stats = b.card_loader.get_card(resolve_card_name(card_name, defs))
                        self._spell_cost = stats.mana_cost if stats else 0
                        self._spell_pre_hp = self._get_total_troop_hp()[1] + sum(self._get_tower_hps()[1])
                    deployed = _silence(b.deploy_card, 0, card_name, Position(gx, gy))
                    if is_spell and deployed:
                        self._spell_deployed = True

        # --- Opponent acts ---
        self.opponent_fn(b, self._rng)

        # --- Tick simulation ---
        for _ in range(self.ticks_per_step):
            if self._is_game_over():
                break
            _silence(b.step)

        # --- Compute reward ---
        reward = self._compute_reward()

        # --- Check done ---
        done = self._is_game_over()
        winner = self._get_winner()

        if done:
            if winner == 0:
                reward += 10.0
            elif winner == 1:
                reward -= 10.0
            else:
                # Tiebreaker: who has more tower HP
                hp0 = sum(self._get_tower_hps()[0])
                hp1 = sum(self._get_tower_hps()[1])
                if hp0 > hp1:
                    reward += 5.0
                elif hp1 > hp0:
                    reward -= 5.0

        return self._obs(), reward, done, False, {"winner": winner, "tick": b.tick}

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self):
        reward = 0.0

        # Tower damage
        curr_tower = self._get_tower_hps()
        for i in range(3):
            # Damage dealt to enemy towers (good)
            enemy_dmg = max(0.0, self._prev_tower_hp[1][i] - curr_tower[1][i])
            reward += 4.0 * enemy_dmg / TOWER_DMG_NORM
            # Damage taken on own towers (bad)
            own_dmg = max(0.0, self._prev_tower_hp[0][i] - curr_tower[0][i])
            reward -= 4.0 * own_dmg / TOWER_DMG_NORM

        # Tower destruction bonuses
        curr_alive = {pid: self._count_alive_towers(pid) for pid in (0, 1)}
        enemy_destroyed = self._prev_towers_alive[1] - curr_alive[1]
        own_destroyed = self._prev_towers_alive[0] - curr_alive[0]
        reward += 2.0 * enemy_destroyed
        reward -= 2.0 * own_destroyed

        # Troop damage
        curr_troop = self._get_total_troop_hp()
        # Enemy troops lost HP (we damaged them) → good
        enemy_troop_dmg = max(0.0, self._prev_troop_hp[1] - curr_troop[1])
        reward += 1.0 * enemy_troop_dmg / TROOP_DMG_NORM
        # Own troops lost HP (they damaged us) → bad
        own_troop_dmg = max(0.0, self._prev_troop_hp[0] - curr_troop[0])
        reward -= 1.0 * own_troop_dmg / TROOP_DMG_NORM
        if self._spell_deployed:
            post_hp = self._get_total_troop_hp()[1] + sum(curr_tower[1])
            if post_hp >= self._spell_pre_hp:
                reward -= 0.5 * self._spell_cost

        # Elixir overflow penalty
        if self._battle.players[0].elixir >= 10.0:
            reward -= 0.1

        # Update tracking
        self._prev_tower_hp = curr_tower
        self._prev_troop_hp = curr_troop
        self._prev_towers_alive = curr_alive

        return reward

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _obs(self):
        b = self._battle
        p0, p1 = b.players

        # ---- Spatial grid ----
        spatial = np.zeros((SPATIAL_CHANNELS, ARENA_H, ARENA_W), dtype=np.float32)

        for entity in b.entities.values():
            if not entity.is_alive:
                continue
            # Grid coords (clamp to arena)
            gx = min(max(int(entity.position.x), 0), ARENA_W - 1)
            gy = min(max(int(entity.position.y), 0), ARENA_H - 1)
            hp_norm = min(entity.hitpoints / MAX_HP, 1.0)
            is_own = entity.player_id == 0

            if isinstance(entity, Building):
                ch = 4 if is_own else 5
                spatial[ch, gy, gx] = max(spatial[ch, gy, gx], hp_norm)
            elif isinstance(entity, Troop):
                ch_hp = 0 if is_own else 1
                ch_air = 2 if is_own else 3
                spatial[ch_hp, gy, gx] = max(spatial[ch_hp, gy, gx], hp_norm)
                if entity.is_air_unit:
                    spatial[ch_air, gy, gx] = 1.0

        # ---- Flat vector ----
        flat = np.zeros(FLAT_DIM, dtype=np.float32)
        idx = 0

        # Elixir
        flat[idx] = p0.elixir / MAX_ELIXIR
        idx += 1

        # Hand cards (4 × CARD_FEAT_DIM)
        for i in range(NUM_HAND_CARDS):
            if i < len(p0.hand):
                feat = self._agent_feats.get(p0.hand[i], np.zeros(CARD_FEAT_DIM))
            else:
                feat = np.zeros(CARD_FEAT_DIM)
            flat[idx:idx + CARD_FEAT_DIM] = feat
            idx += CARD_FEAT_DIM

        # Next card
        if p0.cycle_queue:
            feat = self._agent_feats.get(p0.cycle_queue[0], np.zeros(CARD_FEAT_DIM))
        else:
            feat = np.zeros(CARD_FEAT_DIM)
        flat[idx:idx + CARD_FEAT_DIM] = feat
        idx += CARD_FEAT_DIM

        # Own tower HPs (king, left, right)
        flat[idx] = p0.king_tower_hp / TOWER_KING_HP;    idx += 1
        flat[idx] = p0.left_tower_hp / TOWER_PRINCESS_HP; idx += 1
        flat[idx] = p0.right_tower_hp / TOWER_PRINCESS_HP; idx += 1

        # Enemy tower HPs
        flat[idx] = p1.king_tower_hp / TOWER_KING_HP;    idx += 1
        flat[idx] = p1.left_tower_hp / TOWER_PRINCESS_HP; idx += 1
        flat[idx] = p1.right_tower_hp / TOWER_PRINCESS_HP; idx += 1

        # Game time
        flat[idx] = b.time / MAX_TIME
        idx += 1

        flat = np.clip(flat, -1.0, 1.0)

        return {"spatial": spatial, "flat": flat}

    # ------------------------------------------------------------------
    # Game state helpers
    # ------------------------------------------------------------------
    def _get_tower_hps(self):
        """Return {pid: (king_hp, left_hp, right_hp)}."""
        out = {}
        for pid in (0, 1):
            p = self._battle.players[pid]
            out[pid] = (p.king_tower_hp, p.left_tower_hp, p.right_tower_hp)
        return out

    def _get_total_troop_hp(self):
        """Return {pid: total_hitpoints_of_all_troops}."""
        totals = {0: 0.0, 1: 0.0}
        for e in self._battle.entities.values():
            if e.is_alive and isinstance(e, Troop):
                totals[e.player_id] += e.hitpoints
        return totals

    def _count_alive_towers(self, pid):
        p = self._battle.players[pid]
        c = 0
        if p.king_tower_hp > 0: c += 1
        if p.left_tower_hp > 0: c += 1
        if p.right_tower_hp > 0: c += 1
        return c

    def _is_game_over(self):
        b = self._battle
        if b.game_over:
            return True
        if b.players[0].king_tower_hp <= 0 or b.players[1].king_tower_hp <= 0:
            return True
        if b.tick >= self.max_game_ticks:
            return True
        return False

    def _get_winner(self):
        b = self._battle
        if b.winner is not None:
            return 0 if b.winner == 0 else 1

        if b.players[1].king_tower_hp <= 0:
            return 0  # agent wins
        if b.players[0].king_tower_hp <= 0:
            return 1  # opponent wins
        # Time up: compare crowns (towers destroyed)
        agent_crowns = 3 - self._count_alive_towers(1)
        opp_crowns = 3 - self._count_alive_towers(0)
        if agent_crowns > opp_crowns:
            return 0
        if opp_crowns > agent_crowns:
            return 1
        return -1  # draw

    # ------------------------------------------------------------------
    # Default random opponent
    # ------------------------------------------------------------------
    @staticmethod
    def _random_opponent(battle: BattleState, rng: random.Random):
        """Mimic random_battle.py: 10% chance to play first hand card,
        plus forced play at 10 elixir."""
        p1 = battle.players[1]
        defs = battle.card_loader.load_card_definitions()

        def _try_play():
            if not p1.hand:
                return
            card_name = p1.hand[0]
            resolved = resolve_card_name(card_name, defs)
            card_stats = battle.card_loader.get_card(resolved)
            if not card_stats or p1.elixir < card_stats.mana_cost:
                return
            x = rng.uniform(1, 17)
            y = rng.uniform(18, 31)
            _silence(battle.deploy_card, 1, card_name, Position(x, y))

        # 10% chance per call (called every step = every 10 ticks)
        if rng.random() < 0.1:
            _try_play()

        # Always play at full elixir
        if p1.elixir >= 10.0:
            _try_play()


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    env = ClashRoyaleEnv()
    obs, info = env.reset()
    print(f"Action space:  {env.action_space}")         # Discrete(161)
    print(f"Spatial shape: {obs['spatial'].shape}")      # (6, 32, 18)
    print(f"Flat shape:    {obs['flat'].shape}")         # (48,)

    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        if steps % 100 == 0:
            print(f"  step {steps:4d}  tick {info['tick']:5d}  reward_so_far={total_reward:+.2f}")

    print(f"\nGame over in {steps} steps ({info['tick']} ticks)")
    print(f"Winner: {'Agent' if info['winner']==0 else 'Opponent' if info['winner']==1 else 'Draw'}")
    print(f"Total reward: {total_reward:+.2f}")
