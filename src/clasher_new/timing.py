from environment import CREnv, random_strategy, entity_names
import time

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch.nn.functional as F
import torch

from train import CRFeatureExtractor

env = CREnv(opponent_model=lambda obs: random_strategy(obs))
model = PPO('MultiInputPolicy', env, policy_kwargs={"features_extractor_class": CRFeatureExtractor},
            verbose=1, tensorboard_log="./cr_logs")
model.save('cr_checkpoint')

# Test env stepping alone (no model)
obs, _ = env.reset()
t0 = time.time()
for _ in range(100):
    env.step(env.action_space.sample())
print(f"Env only: {time.time()-t0:.2f}s")

# Test model inference alone
t0 = time.time()
obs, _ = env.reset()
for _ in range(100):
    model.predict(obs)
print(f"Model only: {time.time()-t0:.2f}s")