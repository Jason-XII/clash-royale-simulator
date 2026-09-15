import random

import numpy as np

import battle
import player
from environment import CREnv, Position, player_0_deck, random_strategy, shuffle
from new_visualization import Visualizer
from strategies import defensive_strategy, bridge_pressure_strategy, split_lane_strategy, counterpush_strategy

from stable_baselines3 import PPO

from tqdm import tqdm


class SequentialEvalEnv(CREnv):
    """Replay a fixed sequence of opponent deployments for model inspection."""
    def __init__(self, start_deck, events, visualize=False, speed=1.0):
        super().__init__(visualize=visualize, speed=speed)
        self.deck = start_deck
        self.events = events

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        shuffle(player_0_deck)
        self.battle = battle.BattleState(
            player.PlayerState(0, player_0_deck[:], 9.0),
            player.PlayerState(1, self.deck[:], 9.0),
        )
        if self.visualize:
            self.visualizer = Visualizer(self.battle)
        return self.observe(0), {}

    def opponent_action(self):
        for event in self.events:
            card, x, y, event_time = event
            if abs(self.battle.time - event_time) < 0.1:
                self.battle.deploy_card(1, card, Position(18 - (x + 0.5), 32 - (y + 0.5)))


def evaluate_strategy(strategy, games=100, seed=0):
    """Return wins and game lengths for a strategy playing as player 1."""
    random.seed(seed)
    np.random.seed(seed)
    env = CREnv(opponent_model=strategy, visualize=False)
    wins = 0
    lengths = []
    try:
        for _ in range(games):
            observation, _ = env.reset()
            done = False
            while not done:
                action = random_strategy(observation)
                observation, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            wins += env.battle.winner == 1
            lengths.append(env.battle.time)
    finally:
        env.close()
    return wins, lengths

def evaluate_model(model, strategy_pool, games=10, seed=0, visualize=False):
    random.seed(seed)
    np.random.seed(seed)
    for strategy in strategy_pool:
        env = CREnv(opponent_model=strategy, visualize=visualize)
        wins = 0
        lengths = []
        try:
            for _ in tqdm(range(games)):
                observation, _ = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(observation)
                    observation, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                wins += env.battle.winner == 0
                lengths.append(env.battle.time)
        finally:
            env.close()
        print('against', strategy.__name__, 'wins:', wins)

strategy_pool = [random_strategy, defensive_strategy, bridge_pressure_strategy, split_lane_strategy, counterpush_strategy]
models = ['1000000', '2000000', '3000000', '4000000', '5000000', '6005312']
models_advanced = ['7005312', '7865312']
models_final = ['10004192', '13004192', '15004192', '17004192', '19224192']
for name in models_final:
    print('Evaluating steps', name)
    model = PPO.load(f'cr_moe_dir/cr_{name}_steps.zip')
    evaluate_model(model, strategy_pool, 20, visualize=False)