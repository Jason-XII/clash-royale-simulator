import sys
import numpy as np

from environment import CREnv, random_strategy
from stable_baselines3 import PPO

opponent_model = PPO.load('cr_checkpoint.zip')

env = CREnv(opponent_model=lambda obs: opponent_model.predict(obs)[0], visualize=True, speed=3)
model = PPO.load("cr_logs/cr_647734_steps", env=env)
opponent_model.set_env(env)

for i in range(1):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, termination, truncation, info = env.step(action)
        done = termination or truncation
        total_reward += reward

    print(total_reward)
