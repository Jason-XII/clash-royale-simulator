"""Fine-tune 5.2M with episode-level horizontal reflection.

Run from src/clasher_new: python train_reflection.py
The opponent mixture and saved PPO settings are unchanged from train_counter.
"""
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from environment import CREnv, entity_names
from train_counter import CounterMixture, COUNTER_FRACTION


CHECKPOINT = Path("cr_decision_dir/cr_5200000_steps.zip")
RUN_DIR = Path("cr_reflection")
ADDITIONAL_STEPS = 1_000_000
SEED = 0
N_ENVS = 16


class EpisodeReflection(gym.Wrapper):
    """Reflect only the learner's view/actions, consistently for an episode."""

    def __init__(self, env, seed=0):
        super().__init__(env)
        self.reflection_rng = random.Random(seed)
        self.reflected = False

    def observation(self, obs):
        if not self.reflected:
            return obs
        grid = obs["grid"].copy()
        # Physical king x=9.0 is fixed by reflection; discretization puts it
        # at column 9 (or 8 from the opposite perspective), not a movable tile.
        kings = grid[..., 0] == entity_names.index("KingTower")
        grid[kings] = 0
        grid = grid[:, :, ::-1, :].copy()
        grid[kings] = obs["grid"][kings]
        return dict(obs, grid=grid)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.reflection_rng.seed(seed)
        # A private RNG avoids perturbing opponent/deck randomization.
        self.reflected = self.reflection_rng.random() < 0.5
        return self.observation(obs), info

    def step(self, action):
        slot, y, x = map(int, action)
        if self.reflected and slot:
            x = 17 - x
        obs, reward, terminated, truncated, info = self.env.step((slot, y, x))
        return self.observation(obs), reward, terminated, truncated, info


def make_env(rank):
    def factory():
        seed = 10_000 + SEED * 1_000 + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(1)
        return EpisodeReflection(CREnv(opponent_model=CounterMixture()), seed=seed)
    return factory


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing {CHECKPOINT}; run from src/clasher_new.")
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    env = VecMonitor(SubprocVecEnv(
        [make_env(rank) for rank in range(N_ENVS)], start_method="spawn"
    ), filename=str(RUN_DIR / "monitor.csv"))
    try:
        model = PPO.load(CHECKPOINT, env=env, device="auto",
                         tensorboard_log=str(RUN_DIR / "tensorboard"))
        model.set_random_seed(SEED)
        (RUN_DIR / "experiment.json").write_text(json.dumps({
            "checkpoint": str(CHECKPOINT.resolve()),
            "initial_steps": int(model.num_timesteps),
            "additional_steps": ADDITIONAL_STEPS, "seed": SEED, "n_envs": N_ENVS,
            "counter_fraction": COUNTER_FRACTION, "reflection_probability": 0.5,
        }, indent=2) + "\n")
        callback = CheckpointCallback(
            save_freq=max(100_000 // N_ENVS, 1), save_path=str(RUN_DIR), name_prefix="cr",
        )
        try:
            model.learn(total_timesteps=ADDITIONAL_STEPS, callback=callback,
                        reset_num_timesteps=False, tb_log_name="reflection")
        finally:
            model.save(RUN_DIR / "final.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
