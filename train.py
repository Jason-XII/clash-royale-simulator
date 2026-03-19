#!/usr/bin/env python3
"""Train a PPO agent to play Clash Royale (Log Bait vs Royal Hogs)."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from clash_env import ClashRoyaleEnv


def make_env(seed=0):
    """Factory for creating a monitored env instance."""
    def _init():
        env = ClashRoyaleEnv()
        env.reset(seed=seed)
        return env
    return _init


if __name__ == "__main__":
    N_ENVS = 4               # parallel environments
    TOTAL_TIMESTEPS = 500_000  # increase as needed
    SAVE_DIR = "models"
    LOG_DIR = "logs"
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Vectorised parallel envs ---
    env = SubprocVecEnv([make_env(seed=i) for i in range(N_ENVS)])
    env = VecMonitor(env, LOG_DIR)

    # --- Single eval env for periodic win-rate checks ---
    eval_env = Monitor(ClashRoyaleEnv())

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=25_000 // N_ENVS,   # save every ~25k total steps
        save_path=SAVE_DIR,
        name_prefix="ppo_clash",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, "best"),
        log_path=LOG_DIR,
        eval_freq=10_000 // N_ENVS,   # evaluate every ~10k total steps
        n_eval_episodes=10,
        deterministic=True,
    )

    # --- PPO ---
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
        ent_coef=0.01,        # encourage exploration early on
        verbose=1,
        tensorboard_log=LOG_DIR,
        device='mps'
    )

    print(f"Training PPO for {TOTAL_TIMESTEPS:,} steps across {N_ENVS} envs...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # --- Save final model ---
    final_path = os.path.join(SAVE_DIR, "ppo_clash_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}")
