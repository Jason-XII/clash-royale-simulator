from environment import CREnv, random_strategy
from stable_baselines3 import PPO

env = CREnv(opponent_model=lambda obs: random_strategy(obs), visualize=True, speed=1)
model = PPO.load("cr_checkpoint", env=env)

obs, _ = env.reset()

done = False
total_reward = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, termination, truncation, info = env.step(action)
    done = termination or truncation
    total_reward += reward

print(total_reward)
