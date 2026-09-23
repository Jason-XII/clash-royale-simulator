"""Fine-tune 4M against 80% original opponents and 20% reflected counterpush.

Run from src/clasher_new: python train_counter.py
Settings below define this experiment; existing output directories are refused.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from environment import CREnv, entity_names
from strategies import DiverseOpponent, make_opponent_pool


CHECKPOINT = Path("cr_decision_dir/cr_4000000_steps.zip")
RUN_DIR = Path("cr_counter")
COUNTER_FRACTION = 0.20
ADDITIONAL_STEPS = 2_000_000
SEED = 0
N_ENVS = 16


class CounterMixture:
    """Choose an opponent once per episode, not once per action."""

    def __init__(self):
        self.pool = make_opponent_pool()
        self.counter = DiverseOpponent("counterpush")

    def reset(self):
        self.reflected = random.random() < COUNTER_FRACTION
        self.opponent = self.counter if self.reflected else random.choice(self.pool)
        if callable(getattr(self.opponent, "reset", None)):
            self.opponent.reset()

    def __call__(self, observation):
        if not self.reflected:
            return self.opponent(observation)
        grid = observation["grid"].copy()
        # King towers sit at physical x=9.0, so their discretized columns
        # must stay fixed rather than move from 9 to 8 (or vice versa).
        kings = grid[..., 0] == entity_names.index("KingTower")
        grid[kings] = 0
        grid = grid[:, :, ::-1, :].copy()
        grid[kings] = observation["grid"][kings]
        slot, y, x = self.opponent(dict(observation, grid=grid))
        return (slot, y, 17 - x) if slot else (0, 0, 0)


def make_env(rank):
    def factory():
        random.seed(10_000 + SEED * 1_000 + rank)
        np.random.seed(10_000 + SEED * 1_000 + rank)
        torch.set_num_threads(1)
        return CREnv(opponent_model=CounterMixture())
    return factory


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing {CHECKPOINT}; run from src/clasher_new.")
    if not 0 <= COUNTER_FRACTION <= 1:
        raise ValueError("COUNTER_FRACTION must be between 0 and 1.")
    # Never silently resume a different experiment or overwrite its files.
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    env = VecMonitor(SubprocVecEnv(
        [make_env(rank) for rank in range(N_ENVS)], start_method="spawn"
    ), filename=str(RUN_DIR / "monitor.csv"))
    try:
        # Restore policy, value network, optimizer, and saved PPO settings.
        model = PPO.load(CHECKPOINT, env=env, device="auto",
                         tensorboard_log=str(RUN_DIR / "tensorboard"))
        model.set_random_seed(SEED)
        initial_steps = int(model.num_timesteps)
        (RUN_DIR / "experiment.json").write_text(json.dumps({
            "checkpoint": str(CHECKPOINT.resolve()), "initial_steps": initial_steps,
            "additional_steps": ADDITIONAL_STEPS, "counter_fraction": COUNTER_FRACTION,
            "seed": SEED, "n_envs": N_ENVS,
        }, indent=2) + "\n")
        callback = CheckpointCallback(
            save_freq=max(100_000 // N_ENVS, 1),
            save_path=str(RUN_DIR), name_prefix="cr",
        )
        try:
            # False retains the 4M counter; total_timesteps is ADDITIONAL work.
            # PPO may exceed the requested budget to finish its last rollout.
            model.learn(total_timesteps=ADDITIONAL_STEPS, callback=callback,
                        reset_num_timesteps=False, tb_log_name="counter")
        finally:
            model.save(RUN_DIR / "final.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
