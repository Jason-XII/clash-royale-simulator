"""Long-horizon PPO initialized from the large imitation policy."""

from collections import OrderedDict
from pathlib import Path
import random
import re

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from environment import CREnv
from strategies import make_opponent_pool


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "cr_spatial_long_horizon_dir"
START = ROOT / "cr_script_imitation_spatial_coord_200k.zip"


class MixedOpponent:
    """50% scripts, 30% imitation policy, 20% latest policy snapshot."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.scripts = make_opponent_pool()
        self.history = [START]
        self.models = OrderedDict()
        self.opponent = self.rng.choice(self.scripts)

    def _load(self, path):
        key = str(path)
        if key not in self.models:
            self.models[key] = PPO.load(key, device="cpu")
            if len(self.models) > 2:
                self.models.popitem(last=False)
        self.models.move_to_end(key)
        return self.models[key]

    def reset(self):
        draw = self.rng.random()
        if draw < 0.5:
            self.opponent = self.rng.choice(self.scripts)
        else:
            step = lambda path: int(re.search(r"cr_(\d+)_steps", path.name).group(1))
            recent = sorted(OUTPUT.glob("cr_*_steps.zip"), key=step)
            choices = recent[-1:] if draw >= 0.8 and recent else self.history
            self.opponent = self._load(self.rng.choice(choices))
        reset = getattr(self.opponent, "reset", None)
        if callable(reset):
            reset()

    def __call__(self, observation):
        if isinstance(self.opponent, PPO):
            return self.opponent.predict(observation, deterministic=False)[0]
        return self.opponent(observation)


def make_env(rank):
    def factory():
        random.seed(10_000 + rank)
        np.random.seed(10_000 + rank)
        torch.set_num_threads(1)
        return CREnv(opponent_model=MixedOpponent(10_000 + rank))
    return factory


if __name__ == "__main__":
    n_envs = 16
    OUTPUT.mkdir(exist_ok=True)
    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method="spawn"))
    model = PPO.load(
        START, env=env, device="auto",
        n_steps=8192 // n_envs, batch_size=256, learning_rate=1e-4,
        n_epochs=4, gamma=0.997, gae_lambda=0.98,
        target_kl=0.03, ent_coef=0.01,
        tensorboard_log=str(OUTPUT),
    )
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(), lr=1e-4, **model.policy.optimizer_kwargs
    )
    callback = CheckpointCallback(
        100_000 // n_envs, str(OUTPUT), name_prefix="cr"
    )
    try:
        model.learn(15_000_000, callback=callback)
    finally:
        model.save(ROOT / "cr_spatial_long_horizon")
        env.close()
