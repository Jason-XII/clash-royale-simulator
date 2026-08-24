from environment import CREnv, random_strategy
from stable_baselines3 import PPO
import random

steps = ('5655600', '200000', '600000', '1000000', '1501472', '1803232', '4114416')
elo = [1500, 1364.2788279391582, 1483.4884190224063, 1500, 1500, 1516.5115809775937, 1635.7211720608418]
models = [PPO.load(f"cr_logs/cr_{each}_steps.zip") for each in steps]

def expected(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def update(r_a, r_b, score_a, k=32):
    e_a = expected(r_a, r_b)
    return r_a + k * (score_a - e_a), r_b + k * ((1 - score_a) - (1 - e_a))

games_count = 3

for j in range(200):
    index0, index1 = random.sample(list(range(7)), 2)
    print(index0, index1)
    model1 = models[index0]
    model2 = models[index1]
    env = CREnv(opponent_model=lambda observation: model2.predict(observation)[0])

    for i in range(games_count):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model1.predict(obs)
            obs, reward, termination, truncation, info = env.step(action)
            done = termination or truncation
        updated = update(elo[index0], elo[index1], 1-env.battle.winner)
        elo[index0] = updated[0]
        elo[index1] = updated[1]
    print(elo)
