import random

import numpy as np

import battle
import player
from environment import CREnv, Position, player_0_deck, random_strategy, shuffle
from new_visualization import Visualizer
from strategies import STRATEGIES


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


names = list(STRATEGIES)
games = 10
for name in names:
    wins, lengths = evaluate_strategy(STRATEGIES[name], games, 67)
    print(
        f"{name:12} {wins:3d}-{10 - wins:<3d} "
        f"win_rate={wins / games:6.1%} mean_game={np.mean(lengths):6.1f}s"
    )
