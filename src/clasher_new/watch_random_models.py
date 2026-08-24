from environment import CREnv, random_strategy
from stable_baselines3 import PPO

model = PPO.load("cr_logs/cr_5655600_steps.zip")

env = CREnv(opponent_model=lambda observation: model.predict(observation)[0])

for i in range(50):
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print(f"Winner: player {env.battle.winner}")