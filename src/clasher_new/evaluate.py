"""Evaluate current checkpoints against scripts, with placement and entropy reports."""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import tqdm


def reward_options(model):
    """Old metadata-less checkpoints used the legacy reward; never silently relabel it."""
    config = getattr(model, "run_metadata", {}).get("config", {})
    return dict(reward_mode=config.get("reward_mode", "legacy"), gamma=model.gamma,
                shaping_scale=config.get("shaping_scale", 1.0))


def evaluate_checkpoint(checkpoint, strategy_pool, games=20, seed=0, deterministic=True, visualize=False):
    """Evaluate one checkpoint and report behavior as well as win rate."""
    from environment import CREnv

    model = PPO.load(checkpoint, device="auto")
    options = reward_options(model)
    print(f"Reward settings: {options}; compare win rates, not rewards across objectives.")
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
        env = CREnv(opponent_model=strategy, visualize=visualize, **options)
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
                    action, _ = model.predict(observation, deterministic=deterministic)
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
            "reward_settings": options,
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


@torch.no_grad()
def compare_policy_entropy(checkpoints, strategy_pool, games=3, seed=0, deterministic=True, visualize=False):
    """Compare masked-policy uncertainty on one shared set of observations."""
    from environment import CREnv, entity_names

    if not checkpoints or not strategy_pool or games < 1:
        raise ValueError("provide checkpoints, opponents, and a positive game count")
    models = [PPO.load(checkpoint, device="auto") for checkpoint in checkpoints]
    stats = [dict(states=0, forced=0, actionable=0, action_entropy=0.0,
                  joint_entropy=0.0, card_entropy=0.0, actionable_noop=0.0,
                  placement_entropy=0.0, noop_probability=0.0,
                  cards=np.zeros(len(entity_names))) for _ in models]

    def record(model, observation, stat):
        obs, _ = model.policy.obs_to_tensor(observation)
        diagnostics = model.policy.exploration_statistics(obs)
        probabilities = diagnostics["slot_probabilities"]
        hand = diagnostics["hand"]

        stat["states"] += 1
        stat["noop_probability"] += probabilities[0, 0].item()
        if not diagnostics["actionable"].item():
            stat["forced"] += 1
            return

        stat["actionable"] += 1
        stat["action_entropy"] += diagnostics["slot_entropy"].item()
        stat["joint_entropy"] += diagnostics["joint_entropy"].item()
        stat["actionable_noop"] += probabilities[0, 0].item()
        stat["card_entropy"] += diagnostics["card_entropy_given_play"].item()
        stat["placement_entropy"] += diagnostics["placement_entropy_given_play"].item()
        for slot in range(4):
            stat["cards"][hand[0, slot].item()] += probabilities[0, slot + 1].item()

    collector = models[0]
    for strategy in strategy_pool:
        random.seed(seed)
        np.random.seed(seed)
        env = CREnv(opponent_model=strategy, visualize=visualize, **reward_options(collector))
        try:
            for _ in tqdm(range(games), desc=strategy.__name__, leave=False):
                observation, _ = env.reset()
                done = False
                while not done:
                    for model, stat in zip(models, stats):
                        record(model, observation, stat)
                    action, _ = collector.predict(observation, deterministic=deterministic)
                    observation, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
        finally:
            env.close()

    for checkpoint, stat in zip(checkpoints, stats):
        actionable = stat["actionable"]
        print(f"\n{checkpoint}")
        print(f"  forced no-op states: {stat['forced'] / stat['states']:.1%}")
        print(f"  mean no-op probability: {stat['noop_probability'] / stat['states']:.3f}")
        if actionable == 0:
            print("  no playable-card states; conditional metrics unavailable")
            continue
        print(f"  actionable no-op probability: {stat['actionable_noop'] / actionable:.3f}")
        print(f"  actionable slot entropy: {stat['action_entropy'] / actionable:.3f}")
        print(f"  actionable joint entropy: {stat['joint_entropy'] / actionable:.3f}")
        print(f"  card entropy conditional on playing: {stat['card_entropy'] / actionable:.3f}")
        print(f"  conditional placement entropy: {stat['placement_entropy'] / actionable:.3f}")
        print("  actionable per-card probabilities:")
        for card_id in np.argsort(stat["cards"])[::-1]:
            if stat["cards"][card_id] > 0:
                print(f"    {entity_names[card_id]:12} {stat['cards'][card_id] / actionable:.3%}")
    return stats

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--games", type=int, default=20, help="games per opponent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suite", choices=("all", "core", "randomized"), default="all")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking their modes")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--entropy", action="store_true", help="compare uncertainty on shared states")
    args = parser.parse_args(argv)
    if args.games < 1:
        parser.error("--games must be positive")
    checkpoints = [path.expanduser().resolve() for path in args.checkpoints]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint}")
    previous_cwd = Path.cwd()
    try:
        os.chdir(Path(__file__).resolve().parent)
        from strategies import STRATEGIES, make_diverse_opponents, make_opponent_pool
        opponents = (list(STRATEGIES.values()) if args.suite == "core" else
                     make_diverse_opponents() if args.suite == "randomized" else make_opponent_pool())
        options = dict(games=args.games, seed=args.seed,
                       deterministic=not args.stochastic, visualize=args.visualize)
        if args.entropy:
            compare_policy_entropy(checkpoints, opponents, **options)
        else:
            for checkpoint in checkpoints:
                print(f"Checkpoint: {checkpoint}")
                evaluate_checkpoint(checkpoint, opponents, **options)
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
