import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import battle
import player
from environment import CREnv, Position, player_0_deck, random_strategy, shuffle
from strategies import defensive_strategy, bridge_pressure_strategy, split_lane_strategy, counterpush_strategy

from stable_baselines3 import PPO

from tqdm import tqdm

# Importing this class allows SB3 to restore autoregressive checkpoints.
from train_autoregressive import AutoregressivePolicy


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
            from new_visualization import Visualizer
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


def evaluate_checkpoint(checkpoint, strategy_pool, games=20, seed=0):
    """Evaluate one checkpoint and report behavior as well as win rate."""
    model = PPO.load(checkpoint, device="auto")
    results = {}
    total_decisions = 0
    total_noops = 0
    card_attempts = defaultdict(int)
    card_successes = defaultdict(int)
    placement_counts = defaultdict(lambda: np.zeros((32, 18), dtype=np.int64))

    for strategy in strategy_pool:
        # Reset RNGs for every opponent and checkpoint to make comparisons fair.
        random.seed(seed)
        np.random.seed(seed)
        env = CREnv(opponent_model=strategy, visualize=False)
        wins = 0
        rewards = []
        lengths = []
        decisions = 0
        noops = 0
        attempts = 0
        successes = 0

        try:
            for _ in tqdm(range(games), desc=strategy.__name__, leave=False):
                observation, _ = env.reset()
                done = False
                episode_reward = 0.0

                while not done:
                    action, _ = model.predict(observation, deterministic=True)
                    slot, y, x = (int(value) for value in np.asarray(action).reshape(-1))
                    decisions += 1

                    card_name = None
                    cycle_before = tuple(env.battle.players[0].cycle)
                    if slot == 0:
                        noops += 1
                    elif 1 <= slot <= 4:
                        card_name = cycle_before[slot - 1]
                        attempts += 1
                        card_attempts[card_name] += 1

                    observation, reward, terminated, truncated, _ = env.step((slot, y, x))
                    episode_reward += float(reward)
                    done = terminated or truncated

                    if card_name is not None:
                        cycle_after = tuple(env.battle.players[0].cycle)
                        if cycle_after != cycle_before:
                            successes += 1
                            card_successes[card_name] += 1
                            placement_counts[card_name][y, x] += 1

                wins += env.battle.winner == 0
                rewards.append(episode_reward)
                lengths.append(env.battle.time)
        finally:
            env.close()

        total_decisions += decisions
        total_noops += noops
        results[strategy.__name__] = {
            "wins": wins,
            "games": games,
            "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),
            "noop_rate": noops / decisions,
            "deployment_success_rate": successes / attempts if attempts else 0.0,
        }
        print(
            f"{strategy.__name__:26} "
            f"wins {wins:3}/{games:<3}  "
            f"reward {np.mean(rewards):7.2f}  "
            f"length {np.mean(lengths):6.1f}s  "
            f"noop {noops / decisions:6.1%}  "
            f"valid {successes / attempts if attempts else 0.0:6.1%}"
        )

    print(f"Overall no-op rate: {total_noops / total_decisions:.1%}")
    print("Per-card behavior (successful placement coordinates):")
    for card_name in sorted(card_attempts):
        attempts = card_attempts[card_name]
        successes = card_successes[card_name]
        grid = placement_counts[card_name]
        occupied = np.argwhere(grid > 0)
        if successes:
            mean_y, mean_x = np.average(occupied, axis=0, weights=grid[grid > 0])
            top_indices = np.argsort(grid.ravel())[-3:][::-1]
            top_cells = [
                (int(index // 18), int(index % 18), int(grid.ravel()[index]))
                for index in top_indices
                if grid.ravel()[index] > 0
            ]
        else:
            mean_y, mean_x, top_cells = float("nan"), float("nan"), []
        print(
            f"  {card_name:12} attempts {attempts:5}  "
            f"valid {successes / attempts:6.1%}  "
            f"mean(y,x)=({mean_y:4.1f},{mean_x:4.1f})  top={top_cells}"
        )

    return results


def evaluate_checkpoints(checkpoints, strategy_pool, games=20, seed=0):
    """Evaluate several checkpoints under identical deterministic conditions."""
    all_results = {}
    for checkpoint in checkpoints:
        checkpoint = Path(checkpoint)
        if not checkpoint.exists() and not checkpoint.with_suffix(".zip").exists():
            print(f"Skipping missing checkpoint: {checkpoint}")
            continue
        print(f"\n{'=' * 80}\nCheckpoint: {checkpoint}\n{'=' * 80}")
        all_results[str(checkpoint)] = evaluate_checkpoint(
            checkpoint, strategy_pool, games=games, seed=seed
        )
    return all_results


strategy_pool = [
    random_strategy,
    defensive_strategy,
    bridge_pressure_strategy,
    split_lane_strategy,
    counterpush_strategy,
]


if __name__ == "__main__":
    # Change these paths to the checkpoints that you want to compare.
    checkpoints = [
        "cr_autoregressive_moe_dir/cr_5000000_steps.zip",
        "cr_autoregressive_moe_dir/cr_10000000_steps.zip",
        "cr_autoregressive_moe_dir/cr_15000000_steps.zip",
    ]
    evaluate_checkpoints(checkpoints, strategy_pool, games=20, seed=0)
