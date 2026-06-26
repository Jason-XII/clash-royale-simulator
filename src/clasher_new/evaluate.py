import sys
import numpy as np

from environment import CREnv, random_strategy
from stable_baselines3 import PPO


env = CREnv(opponent_model=lambda obs: random_strategy(obs), visualize=False, speed=1)
model = PPO.load("cr_logs/cr_1660000_steps", env=env)

wins = 0
for i in range(100):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, termination, truncation, info = env.step(action)
        # print(reward)
        done = termination or truncation
        total_reward += reward
    wins += env.battle.winner == 0
    print(total_reward, env.battle.winner == 0)
print('Won', wins, 'out of 100 games.')
