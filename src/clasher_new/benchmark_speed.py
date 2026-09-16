"""Measure single-process simulator throughput with random policies."""

import random
import time

import numpy as np

from environment import CREnv, random_strategy


GAMES = 10
SEED = 0


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    env = CREnv(opponent_model=random_strategy, visualize=False)
    simulated_seconds = 0.0
    decisions = 0
    start = time.perf_counter()

    try:
        for _ in range(GAMES):
            observation, _ = env.reset()
            done = False
            while not done:
                action = random_strategy(observation)
                observation, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                decisions += 1
            simulated_seconds += env.battle.time
    finally:
        env.close()

    wall_seconds = time.perf_counter() - start
    print(f"Games: {GAMES}")
    print(f"Wall time: {wall_seconds:.2f} seconds")
    print(f"Average wall time per game: {wall_seconds / GAMES:.3f} seconds")
    print(f"Decisions per second: {decisions / wall_seconds:.1f}")
    print(f"Simulation speed: {simulated_seconds / wall_seconds:.1f}x real time")


if __name__ == "__main__":
    main()
