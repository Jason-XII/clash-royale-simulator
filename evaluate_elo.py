#!/usr/bin/env python3
"""Evaluate agents in a round-robin tournament and compute Elo ratings.

Usage:
    python evaluate_elo.py --games 20
    python evaluate_elo.py --games 50 --verbose
"""

import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from stable_baselines3 import PPO
from clash_env import ClashRoyaleEnv, _silence, Position, resolve_card_name, SPELL_REGISTRY
from train_selfplay import SelfPlayEnv, DECKS, STATE_FILE, save_state, load_state


def play_game(model_a, deck_a, model_b, deck_b, loader=None):
    """Play one game: model_a (player 0) vs model_b (player 1).
    Returns: 0 if model_a wins, 1 if model_b wins, -1 if draw.
    """
    env = SelfPlayEnv(
        agent_deck=deck_a,
        opponent_pool=[("__dummy__", deck_b)],  # won't be used
        random_deck_pool=[deck_b],
        random_ratio=1.0,
    )
    # Override: we'll manually control both sides
    obs, _ = env.reset()
    # Swap in the correct opponent deck
    env.opponent_deck = deck_b
    env._opp_model = model_b

    # Precompute opponent features
    from clash_env import _card_feature, CARD_FEAT_DIM
    defs = env._loader.load_card_definitions()
    env._opp_feats = {}
    for name in deck_b:
        resolved = resolve_card_name(name, defs)
        stats = env._loader.get_card(resolved)
        env._opp_feats[name] = _card_feature(stats)
    env._opp_is_spell = {}
    for name in deck_b:
        resolved = resolve_card_name(name, defs)
        env._opp_is_spell[name] = resolved in SPELL_REGISTRY
    env.opponent_fn = env._model_opponent

    done = False
    while not done:
        action, _ = model_a.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))

    winner = info.get("winner", -1)
    env.close()
    return winner


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_a, elo_b, result, k=32):
    """Update Elo ratings. result: 1.0 = A wins, 0.0 = A loses, 0.5 = draw."""
    ea = expected_score(elo_a, elo_b)
    eb = expected_score(elo_b, elo_a)
    new_a = elo_a + k * (result - ea)
    new_b = elo_b + k * ((1.0 - result) - eb)
    return new_a, new_b


def run_tournament(games_per_matchup=20, verbose=False):
    state = load_state()
    deck_names = list(DECKS.keys())

    # Load latest models
    models = {}
    for name in deck_names:
        path = state["latest_models"].get(name)
        if not path or not os.path.exists(path + ".zip"):
            # Try warmup
            path = os.path.join("models", "warmup", name, f"ppo_{name}_final")
        if os.path.exists(path + ".zip"):
            models[name] = PPO.load(path, device="cpu")
            print(f"Loaded {name}: {path}")
        else:
            print(f"WARNING: No model found for {name}, skipping")

    available = [n for n in deck_names if n in models]
    if len(available) < 2:
        print("Need at least 2 models to run tournament")
        return

    # Elo ratings (start from state or 1200)
    elo = {n: state["elo"].get(n, 1200) for n in available}

    # Win matrix
    wins = {a: {b: 0 for b in available} for a in available}
    draws = {a: {b: 0 for b in available} for a in available}
    total = {a: {b: 0 for b in available} for a in available}

    # Round robin
    matchups = [(a, b) for i, a in enumerate(available) for b in available[i+1:]]
    n_total = len(matchups) * games_per_matchup
    print(f"\nTournament: {len(matchups)} matchups × {games_per_matchup} games = {n_total} games\n")

    game_count = 0
    for name_a, name_b in matchups:
        for g in range(games_per_matchup):
            # Alternate sides for fairness
            if g % 2 == 0:
                winner = play_game(models[name_a], DECKS[name_a],
                                   models[name_b], DECKS[name_b])
                if winner == 0:
                    result = 1.0
                    wins[name_a][name_b] += 1
                elif winner == 1:
                    result = 0.0
                    wins[name_b][name_a] += 1
                else:
                    result = 0.5
                    draws[name_a][name_b] += 1
                    draws[name_b][name_a] += 1
            else:
                winner = play_game(models[name_b], DECKS[name_b],
                                   models[name_a], DECKS[name_a])
                if winner == 0:
                    result = 0.0
                    wins[name_b][name_a] += 1
                elif winner == 1:
                    result = 1.0
                    wins[name_a][name_b] += 1
                else:
                    result = 0.5
                    draws[name_a][name_b] += 1
                    draws[name_b][name_a] += 1

            elo[name_a], elo[name_b] = update_elo(elo[name_a], elo[name_b], result)
            total[name_a][name_b] += 1
            total[name_b][name_a] += 1
            game_count += 1

            if verbose:
                w_label = {0: name_a if g%2==0 else name_b,
                           1: name_b if g%2==0 else name_a}.get(winner, "Draw")
                print(f"  [{game_count}/{n_total}] {name_a} vs {name_b}: {w_label}")

    # Print results
    print(f"\n{'='*60}")
    print("  ELO RATINGS")
    print(f"{'='*60}")
    for name in sorted(available, key=lambda n: elo[n], reverse=True):
        print(f"  {name:20s}  Elo: {elo[name]:.0f}")

    print(f"\n{'='*60}")
    print("  WIN RATE MATRIX (row vs column)")
    print(f"{'='*60}")
    header = f"{'':20s}" + "".join(f"{n:>15s}" for n in available)
    print(header)
    for a in available:
        row = f"{a:20s}"
        for b in available:
            if a == b:
                row += f"{'---':>15s}"
            elif total[a][b] > 0:
                wr = wins[a][b] / total[a][b] * 100
                row += f"{wr:>14.0f}%"
            else:
                row += f"{'N/A':>15s}"
        print(row)

    # Save Elo back to state
    state["elo"] = {n: round(elo[n], 1) for n in available}
    save_state(state)
    print(f"\nElo ratings saved to {STATE_FILE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20,
                        help="Games per matchup")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_tournament(games_per_matchup=args.games, verbose=args.verbose)
