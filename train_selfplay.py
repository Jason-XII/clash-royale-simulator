#!/usr/bin/env python3
"""Phase 2: Self-play league training with stop/resume support.

Usage:
    python train_selfplay.py --device mps --envs 8 --steps-per-round 500000
    # Ctrl+C to stop — will resume from last completed round automatically

State is saved to models/selfplay/state.json after each agent finishes a round.
"""

import os, sys, json, random, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from clash_env import (
    ClashRoyaleEnv, BattleState, Position, resolve_card_name,
    SPELL_REGISTRY, _silence, NUM_POSITIONS, GRID_POSITIONS,
    P0_MAX_DEPLOY_Y, ARENA_W, ARENA_H, MAX_HP, TOWER_KING_HP,
    TOWER_PRINCESS_HP, MAX_ELIXIR, MAX_TIME, CARD_FEAT_DIM,
    NUM_HAND_CARDS, FLAT_DIM, SPATIAL_CHANNELS,
    Troop, Building, _card_feature,
)
from train_warmup import DECKS


# -------------------------------------------------------------------------
# Self-play environment
# -------------------------------------------------------------------------
class SelfPlayEnv(ClashRoyaleEnv):
    """Env where opponent can be a PPO model, a random bot, or a mix."""

    def __init__(
        self,
        agent_deck,
        opponent_pool,      # list of (model_path, deck_cards) tuples
        random_deck_pool,   # list of deck card lists for random bot fallback
        random_ratio=0.2,   # probability of facing random bot
        **kwargs,
    ):
        self._opponent_pool = opponent_pool
        self._random_deck_pool = random_deck_pool
        self._random_ratio = random_ratio
        self._opp_model = None
        self._opp_deck = None
        self._opp_feats = {}
        super().__init__(agent_deck=agent_deck, **kwargs)

    def reset(self, *, seed=None, options=None):
        # Decide opponent for this episode
        if random.random() < self._random_ratio or not self._opponent_pool:
            # Random bot with a random deck
            self._opp_model = None
            self._opp_deck = random.choice(self._random_deck_pool)
            self.opponent_deck = self._opp_deck
            self.opponent_fn = self._random_opponent
        else:
            # Pick a model opponent
            model_path, deck = random.choice(self._opponent_pool)
            self._opp_model = PPO.load(model_path, device="cpu")
            self._opp_deck = deck
            self.opponent_deck = deck
            self.opponent_fn = self._model_opponent
            # Precompute card features for opponent deck
            self._opp_feats = {}
            defs = self._loader.load_card_definitions()
            for name in deck:
                resolved = resolve_card_name(name, defs)
                stats = self._loader.get_card(resolved)
                self._opp_feats[name] = _card_feature(stats)
            # Pre-check which cards are spells
            self._opp_is_spell = {}
            for name in deck:
                resolved = resolve_card_name(name, defs)
                self._opp_is_spell[name] = resolved in SPELL_REGISTRY

        return super().reset(seed=seed, options=options)

    def _build_opp_obs(self):
        """Build observation from player 1's perspective (flipped)."""
        b = self._battle
        p1 = b.players[1]  # opponent is "self" from their perspective
        p0 = b.players[0]  # agent is "enemy" from their perspective

        # Spatial grid — flipped vertically, with own/enemy swapped
        spatial = np.zeros((SPATIAL_CHANNELS, ARENA_H, ARENA_W), dtype=np.float32)
        for entity in b.entities.values():
            gx = min(max(int(entity.position.x), 0), ARENA_W - 1)
            # Flip y: 31 - y so player 1's side appears at bottom
            gy = min(max(31 - int(entity.position.y), 0), ARENA_H - 1)
            hp_norm = min(entity.hitpoints / MAX_HP, 1.0)
            is_own = entity.player_id == 1  # from opp's perspective

            if isinstance(entity, Building):
                ch = 4 if is_own else 5
                spatial[ch, gy, gx] = max(spatial[ch, gy, gx], hp_norm)
            elif isinstance(entity, Troop):
                ch_hp = 0 if is_own else 1
                ch_air = 2 if is_own else 3
                spatial[ch_hp, gy, gx] = max(spatial[ch_hp, gy, gx], hp_norm)
                if entity.is_air_unit:
                    spatial[ch_air, gy, gx] = 1.0

        # Flat vector
        flat = np.zeros(FLAT_DIM, dtype=np.float32)
        idx = 0

        flat[idx] = p1.elixir / MAX_ELIXIR; idx += 1

        for i in range(NUM_HAND_CARDS):
            if i < len(p1.hand):
                feat = self._opp_feats.get(p1.hand[i], np.zeros(CARD_FEAT_DIM))
            else:
                feat = np.zeros(CARD_FEAT_DIM)
            flat[idx:idx + CARD_FEAT_DIM] = feat
            idx += CARD_FEAT_DIM

        if p1.cycle_queue:
            feat = self._opp_feats.get(p1.cycle_queue[0], np.zeros(CARD_FEAT_DIM))
        else:
            feat = np.zeros(CARD_FEAT_DIM)
        flat[idx:idx + CARD_FEAT_DIM] = feat
        idx += CARD_FEAT_DIM

        # Own towers (player 1) then enemy towers (player 0)
        flat[idx] = p1.king_tower_hp / TOWER_KING_HP;     idx += 1
        flat[idx] = p1.left_tower_hp / TOWER_PRINCESS_HP;  idx += 1
        flat[idx] = p1.right_tower_hp / TOWER_PRINCESS_HP; idx += 1
        flat[idx] = p0.king_tower_hp / TOWER_KING_HP;     idx += 1
        flat[idx] = p0.left_tower_hp / TOWER_PRINCESS_HP;  idx += 1
        flat[idx] = p0.right_tower_hp / TOWER_PRINCESS_HP; idx += 1

        flat[idx] = b.time / MAX_TIME; idx += 1

        return {"spatial": spatial, "flat": flat}

    def _model_opponent(self, battle, rng):
        """Use a loaded PPO model to choose opponent actions."""
        p1 = battle.players[1]
        if p1.elixir < 2:
            return

        obs = self._build_opp_obs()
        action, _ = self._opp_model.predict(obs, deterministic=False)
        action = int(action)

        if action == 0:
            return

        action -= 1
        card_slot = action // NUM_POSITIONS
        pos_idx = action % NUM_POSITIONS

        if card_slot < len(p1.hand):
            card_name = p1.hand[card_slot]
            gx, gy_agent = GRID_POSITIONS[pos_idx]
            # Flip y back: the model thinks in "own half at bottom" coordinates
            gy = 32.0 - gy_agent
            is_spell = self._opp_is_spell.get(card_name, False)
            # Player 1 deploys on top half (y > 17.5)
            if is_spell or gy >= (32.0 - P0_MAX_DEPLOY_Y):
                _silence(battle.deploy_card, 1, card_name, Position(gx, gy))

        # Also auto-play at full elixir
        if p1.elixir >= 10.0 and p1.hand:
            card_name = p1.hand[0]
            defs = battle.card_loader.load_card_definitions()
            resolved = resolve_card_name(card_name, defs)
            stats = battle.card_loader.get_card(resolved)
            if stats and p1.elixir >= stats.mana_cost:
                x = rng.uniform(1, 17)
                y = rng.uniform(18, 31)
                _silence(battle.deploy_card, 1, card_name, Position(x, y))


# -------------------------------------------------------------------------
# State management for stop/resume
# -------------------------------------------------------------------------
STATE_DIR = os.path.join("models", "selfplay")
STATE_FILE = os.path.join(STATE_DIR, "state.json")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "round": 0,
        "agent_idx": 0,
        "checkpoints": {name: [] for name in DECKS},
        "latest_models": {},
        "elo": {name: 1200 for name in DECKS},
    }


def save_state(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"  [State saved: round={state['round']} agent_idx={state['agent_idx']}]")


# -------------------------------------------------------------------------
# Build opponent pool for an agent
# -------------------------------------------------------------------------
def build_opponent_pool(agent_name, state, max_checkpoints_per_agent=3):
    """Build (model_path, deck) list from other agents' latest + checkpoints."""
    pool = []
    for name, deck in DECKS.items():
        if name == agent_name:
            continue
        # Add latest model
        if name in state["latest_models"]:
            pool.append((state["latest_models"][name], deck))
        # Add recent checkpoints
        for ckpt_path in state["checkpoints"][name][-max_checkpoints_per_agent:]:
            pool.append((ckpt_path, deck))
    return pool


# -------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------
def make_selfplay_env(agent_deck, opponent_pool, seed=0):
    def _init():
        env = SelfPlayEnv(
            agent_deck=agent_deck,
            opponent_pool=opponent_pool,
            random_deck_pool=list(DECKS.values()),
            random_ratio=0.2,
            seed=seed,
        )
        return env
    return _init


# -------------------------------------------------------------------------
# Train one round for one agent
# -------------------------------------------------------------------------
def train_one_round(agent_name, state, steps_per_round, n_envs, device):
    agent_deck = DECKS[agent_name]
    opponent_pool = build_opponent_pool(agent_name, state)

    save_dir = os.path.join(STATE_DIR, agent_name)
    log_dir = os.path.join("logs", "selfplay", agent_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = SubprocVecEnv([
        make_selfplay_env(agent_deck, opponent_pool, seed=i)
        for i in range(n_envs)
    ])
    env = VecMonitor(env, log_dir)

    eval_env = Monitor(SelfPlayEnv(
        agent_deck=agent_deck,
        opponent_pool=opponent_pool,
        random_deck_pool=list(DECKS.values()),
        random_ratio=0.0,  # eval only against models
    ))

    # Load existing model or create new
    model_path = state["latest_models"].get(agent_name)
    if model_path and os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env, device=device)
        print(f"  Loaded existing model: {model_path}")
    else:
        # Try loading from warmup
        warmup_path = os.path.join("models", "warmup", agent_name,
                                    f"ppo_{agent_name}_final")
        if os.path.exists(warmup_path + ".zip"):
            model = PPO.load(warmup_path, env=env, device=device)
            print(f"  Loaded warmup model: {warmup_path}")
        else:
            model = PPO(
                "MultiInputPolicy", env,
                learning_rate=3e-4, n_steps=2048, batch_size=256,
                n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01,
                verbose=1, device=device, tensorboard_log=log_dir,
            )
            print(f"  Created new model")

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=log_dir,
        eval_freq=20_000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )

    round_num = state["round"]
    print(f"  Training {agent_name} for {steps_per_round:,} steps (round {round_num})...")
    model.learn(
        total_timesteps=steps_per_round,
        callback=[eval_cb],
        progress_bar=True,
        reset_num_timesteps=False,
    )

    # Save
    model_save = os.path.join(save_dir, f"ppo_{agent_name}_r{round_num}")
    model.save(model_save)
    state["latest_models"][agent_name] = model_save
    state["checkpoints"][agent_name].append(model_save)

    env.close()
    eval_env.close()
    print(f"  Saved: {model_save}")


# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20,
                        help="Total rounds of self-play")
    parser.add_argument("--steps-per-round", type=int, default=500_000,
                        help="Training steps per agent per round")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    deck_names = list(DECKS.keys())
    state = load_state()

    start_round = state["round"]
    start_agent = state["agent_idx"]

    print(f"Self-play league: {len(deck_names)} agents, {args.rounds} rounds")
    print(f"Resuming from round {start_round}, agent {start_agent}")
    print(f"Current Elo: {state['elo']}\n")

    for rnd in range(start_round, args.rounds):
        state["round"] = rnd
        agent_start = start_agent if rnd == start_round else 0

        for ai in range(agent_start, len(deck_names)):
            name = deck_names[ai]
            state["agent_idx"] = ai

            print(f"\n{'='*60}")
            print(f"  Round {rnd} / {args.rounds-1} — Agent: {name}")
            print(f"{'='*60}")

            train_one_round(name, state, args.steps_per_round,
                           args.envs, args.device)

            # Save state after each agent completes
            state["agent_idx"] = ai + 1
            save_state(state)

        # Round complete — reset agent index
        state["agent_idx"] = 0
        state["round"] = rnd + 1
        save_state(state)
        print(f"\n*** Round {rnd} complete ***\n")

    print("Self-play training finished!")
