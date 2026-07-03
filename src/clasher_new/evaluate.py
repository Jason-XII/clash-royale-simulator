import sys
import numpy as np

import battle, player
from new_visualization import Visualizer


from environment import CREnv, random_strategy, player_0_deck, shuffle, Position
from stable_baselines3 import PPO

class SequentialEvalEnv(CREnv):
    def __init__(self, start_deck, events, visualize=False, speed=1.0):
        super().__init__(visualize=visualize, speed=speed)
        self.deck = start_deck
        self.events = events

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        shuffle(player_0_deck)
        self.battle = battle.BattleState(player.PlayerState(0, player_0_deck[:], 5.0),
                                         player.PlayerState(1, self.deck[:], 5.0))
        if self.visualize:
            self.visualizer = Visualizer(self.battle)

        # Now return initial observation
        return self.observe(0), {}

    def opponent_action(self):
        for event in self.events:
            card, x, y, t = event
            # print(self.battle.time, t)
            if abs(self.battle.time - t) < 0.1:
                self.battle.deploy_card(1, card, Position((17 - x) + 0.5, (31 - y) + 0.5))
                # print('Deploying at ', (17 - x) + 0.5, )



env = SequentialEvalEnv(start_deck=['Knight', 'MiniPekka', 'Arrows', 'Giant', 'Musketeer', 'Fireball', 'Minions', 'Archer'],
                        events=[('Giant', 3.5, 12.5, 12.0),
                                ('MiniPekka', 3.5, 11.5, 12.5)],
                        visualize=True, speed=2)
model = PPO.load("cr_logs/cr_5261472_steps.zip", env=env)

wins = 0
for i in range(1):
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
