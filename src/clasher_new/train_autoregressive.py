from environment import CREnv, random_strategy

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch
import torch.nn as nn

from train import CRFeatureExtractor

import random
import numpy as np


def make_env(rank):
    def factory():
        random.seed(10_000 + rank)
        np.random.seed(10_000 + rank)
        torch.set_num_threads(1)
        return CREnv(opponent_model=random_strategy)
    return factory

n_envs = 8
env = SubprocVecEnv([make_env(rank) for rank in range(n_envs)], start_method="spawn")
env = VecMonitor(env)
n_steps = 2048 // n_envs

model = PPO(
   "MultiInputPolicy",
   env,
   n_steps=n_steps,
   batch_size=256,
   policy_kwargs={"features_extractor_class": CRFeatureExtractor},
   device="cuda",
   verbose=1,
)
