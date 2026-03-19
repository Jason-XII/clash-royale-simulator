#!/usr/bin/env python3
"""Phase 1: Warm-up training — train 4 deck agents vs random opponents.

Each agent trains with its own deck, facing a randomly-chosen opponent deck
from the pool each episode.
"""

import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from clash_env import ClashRoyaleEnv


# ---- Deck pool ---------------------------------------------------------
DECKS = {
    "logbait": [
        "GoblinBarrel", "GoblinGang", "IceSpirits", "InfernoTower",
        "Knight", "Princess", "Rocket", "Log",
    ],
    "hogcycle": [
        "Cannon", "Fireball", "HogRider", "IceGolemite",
        "IceSpirits", "Musketeer", "Skeletons", "Log",
    ],
    "royalrecruits": [
        "RoyalRecruits", "RoyalHogs", "DartBarrell", "Fireball",
        "Zap", "GoblinCage", "BarbLog", "Heal",
    ],
    "ragesparky": [
        "Giant", "ZapMachine", "Rage", "DarkPrince",
        "MiniPekka", "Zap", "Wizard", "MinionHorde",
    ],
}

ALL_DECKS = list(DECKS.values())


# ---- Env with random opponent deck each episode -----------------------
class WarmupEnv(ClashRoyaleEnv):
    """Wraps ClashRoyaleEnv to randomise the opponent deck on each reset."""

    def __init__(self, agent_deck, opponent_pool=None, **kwargs):
        self._opponent_pool = opponent_pool or ALL_DECKS
        # Initialise with first opponent deck (will be overwritten in reset)
        super().__init__(agent_deck=agent_deck,
                         opponent_deck=self._opponent_pool[0], **kwargs)

    def reset(self, *, seed=None, options=None):
        # Pick a random opponent deck for this episode
        self.opponent_deck = random.choice(self._opponent_pool)
        return super().reset(seed=seed, options=options)


# ---- Factory -----------------------------------------------------------
def make_env(agent_deck, seed=0):
    def _init():
        env = WarmupEnv(agent_deck=agent_deck, seed=seed)
        return env
    return _init


# ---- Train one agent ---------------------------------------------------
def train_agent(deck_name, agent_deck, total_timesteps=2_000_000,
                n_envs=4, device="cpu"):
    save_dir = os.path.join("models", "warmup", deck_name)
    log_dir = os.path.join("logs", "warmup", deck_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = SubprocVecEnv([make_env(agent_deck, seed=i) for i in range(n_envs)])
    env = VecMonitor(env, log_dir)

    eval_env = Monitor(WarmupEnv(agent_deck=agent_deck))

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=save_dir,
        name_prefix=f"ppo_{deck_name}",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=log_dir,
        eval_freq=20_000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
    )

    print(f"\n{'='*60}")
    print(f"  Training [{deck_name}] for {total_timesteps:,} steps")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(save_dir, f"ppo_{deck_name}_final")
    model.save(final_path)
    print(f"Saved {deck_name} -> {final_path}")

    env.close()
    eval_env.close()
    return final_path


# ---- Main --------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", default=None,
                        help="Train a single deck (logbait/hogcycle/royalrecruits/ragesparky)")
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    if args.deck:
        if args.deck not in DECKS:
            print(f"Unknown deck '{args.deck}'. Choose from: {list(DECKS.keys())}")
            sys.exit(1)
        train_agent(args.deck, DECKS[args.deck],
                    total_timesteps=args.steps, n_envs=args.envs,
                    device=args.device)
    else:
        # Train all 4 sequentially
        for name, deck in DECKS.items():
            train_agent(name, deck,
                        total_timesteps=args.steps, n_envs=args.envs,
                        device=args.device)

    print("\n✅ Warm-up training complete!")