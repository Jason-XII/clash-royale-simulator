from environment import CREnv, random_strategy
from stable_baselines3 import PPO
import time

env = CREnv(opponent_model=lambda obs: random_strategy(obs))

t0 = time.time()
model = PPO('MultiInputPolicy', env)
model.learn(1000)
t1 = time.time()
print('Played approximately 3 games in: ', t1-t0, 's.')
