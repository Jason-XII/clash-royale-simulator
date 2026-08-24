import battle, player
from new_visualization import Visualizer

from environment import CREnv, random_strategy, player_0_deck, shuffle, Position
from stable_baselines3 import PPO

from tqdm import tqdm

import torch

class SequentialEvalEnv(CREnv):
    def __init__(self, start_deck, events, visualize=False, speed=1.0):
        super().__init__(visualize=visualize, speed=speed)
        self.deck = start_deck
        self.events = events

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        shuffle(player_0_deck)
        self.battle = battle.BattleState(player.PlayerState(0, player_0_deck[:], 9.0),
                                         player.PlayerState(1, self.deck[:], 9.0))
        if self.visualize:
            self.visualizer = Visualizer(self.battle)

        # Now return initial observation
        return self.observe(0), {}

    def opponent_action(self):
        for event in self.events:
            card, x, y, t = event
            if abs(self.battle.time - t) < 0.1:
                self.battle.deploy_card(1, card, Position(18-(x+0.5), 32-(y+0.5)))



# env = SequentialEvalEnv(start_deck=['Knight', 'MiniPekka', 'Arrows', 'Giant', 'Musketeer', 'Fireball', 'Minions', 'Archer'],
#                         events=[('Giant', 3, 13, 0.5),
#                                 ('MiniPekka', 3, 12, 0.5)],
#                         visualize=True, speed=1)

steps = ('checkpoint',)
for step in steps:
    model = PPO.load(f"cr_logs/cr_{step}.zip")
    env = CREnv(opponent_model=random_strategy)
    print('Evaluating model at', step, 'steps:')
    reward_total = 0
    games_won = 0
    for i in tqdm(range(200)):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs,deterministic=True)
            obs, reward, termination, truncation, info = env.step(action)
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                value = model.policy.predict_values(obs_tensor)
            done = termination or truncation
            reward_total += reward
        games_won += (1-env.battle.winner)
    print("Win rate:", games_won/200, end=' ')
    print("Mean reward:", reward_total/200)
