from environment import CREnv, random_strategy
from stable_baselines3 import PPO
from random import shuffle

model = PPO.load("cr_logs/cr_5000000_steps.zip")

env = CREnv(opponent_model=random_strategy)

for i in range(50):
    obs, _ = env.reset()
    done = False
    step = 0
    while not done:
        step += 1
        hand = env.battle.players[0].cycle[:4]
        new_hand = hand.copy()
        shuffle(new_hand)
        print('hand:', env.battle.players[0].cycle[:4], 'new_hand: ', new_hand)
        observation_prev = env.observe(0)
        env.battle.players[0].cycle[:4] = new_hand
        observation_new = env.observe(0)
        action_prev, _ = model.predict(observation_prev)
        action_new, _ = model.predict(observation_new)
        print(action_prev, action_new)
        env.battle.players[0].cycle[:4] = hand
        obs, reward, terminated, truncated, info = env.step(action_prev)
        done = terminated or truncated

    print(f"Winner: player {env.battle.winner}")