"""Compare spatial control and legal-mask checkpoints on the same games.

From src/clasher_new:
python evaluate_legality.py cr_spatial_legal_seed0/cr_7000000_steps.zip
python evaluate_legality.py cr_spatial_control_seed0/cr_7000000_steps.zip
"""
import argparse
import random
from collections import Counter

import numpy as np
import torch
from stable_baselines3 import PPO

from environment import CREnv, random_strategy, player_0_deck, player_1_deck
from strategies import make_opponent_pool
from train_reflection import EpisodeReflection
from train_counter import CounterMixture
from masked_spatial import LegalPlacement, MaskedSpatialPolicy


class ReflectedCounter(CounterMixture):
    __name__ = "reflected_counterpush"

    def reset(self):
        self.opponent = self.counter
        self.reflected = True
        self.counter.reset()


class FixedView(EpisodeReflection):
    def __init__(self, env, view):
        super().__init__(env)
        self.view = view

    def reset(self, *, seed=None, options=None):
        if self.view == "training":
            return super().reset(seed=seed, options=options)
        obs, info = self.env.reset(seed=seed, options=options)
        self.reflected = self.view == "reflected"
        return self.observation(obs), info


def evaluate(path, games, seed, view):
    torch.set_num_threads(1)
    model = PPO.load(path, device="cpu")
    masked = isinstance(model.policy, MaskedSpatialPolicy)
    deck0, deck1 = player_0_deck[:], player_1_deck[:]
    opponents = [random_strategy, *make_opponent_pool(), ReflectedCounter()]
    overall = Counter()
    for opponent_index, opponent in enumerate(opponents):
        env = FixedView(CREnv(opponent_model=opponent), view)
        if masked:
            env = LegalPlacement(env)
        scores = Counter()
        rewards = []
        try:
            for episode in range(games):
                episode_seed = seed + 100 * opponent_index + episode
                player_0_deck[:] = deck0
                player_1_deck[:] = deck1
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                observation, _ = env.reset(seed=episode_seed)
                done, reward = False, 0.0
                while not done:
                    action, _ = model.predict(observation, deterministic=False)
                    slot = int(action[0])
                    before = tuple(env.unwrapped.battle.players[0].cycle)
                    scores["decisions"] += 1
                    if slot:
                        scores["attempts"] += 1
                    else:
                        scores["waits"] += 1
                    observation, r, terminated, truncated, _ = env.step(action)
                    reward += float(r)
                    done = terminated or truncated
                    if slot and tuple(env.unwrapped.battle.players[0].cycle) != before:
                        scores["accepted"] += 1
                scores["wins"] += env.unwrapped.battle.winner == 0
                rewards.append(reward)
        finally:
            env.close()
        overall.update(scores)
        print(f"{opponent.__name__:26} wins {scores['wins']:2}/{games:<2} "
              f"reward {np.mean(rewards):6.2f} "
              f"WAIT {scores['waits']/scores['decisions']:5.1%} "
              f"valid {scores['accepted']/max(scores['attempts'],1):5.1%}", flush=True)
    print(f"TOTAL {overall['wins']}/{games*len(opponents)} "
          f"WAIT {overall['waits']/overall['decisions']:.1%} "
          f"valid {overall['accepted']/max(overall['attempts'],1):.1%} "
          f"view {view} mask {masked}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=72000)
    parser.add_argument("--view", choices=("native", "reflected", "training"),
                        default="native")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.games, args.seed, args.view)
