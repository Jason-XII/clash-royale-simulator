"""Track strength across checkpoints: a fixed script panel plus a round-robin Elo.

Example:
    python strength_ladder.py cr_delayed/*.zip --workers 23

Every checkpoint plays the same held-out script panel (same seeds, both sides)
and every other checkpoint. One Bradley-Terry fit over all games gives an Elo
rating per player; error bars come from resampling games. Results are cached
by policy weights in `--cache`, so adding a checkpoint only plays its new games.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from multiprocessing import Pool
from pathlib import Path
import zipfile

import numpy as np

from league_evaluation import evaluate_schedule, make_schedule

PANEL_SEED, PAIR_SEED = 20261003, 20261004
CHUNK = 10   # matches per worker task


def weights_id(path):
    """Same weights, same id, whatever the file name or zip timestamps."""
    return hashlib.sha256(zipfile.ZipFile(path).read("policy.pth")).hexdigest()[:16]


def steps(path):
    return json.loads(zipfile.ZipFile(path).read("data")).get("num_timesteps")


def play(task):
    key, checkpoint, chunk = task
    report = evaluate_schedule(checkpoint, chunk)
    return key, report["total"]["wins"], report["total"]["games"]


def bradley_terry(results, players, iterations=2000):
    """Elo ratings from {(a, b): (wins of a, games)}; mean rating 0.

    Each pair gets half a win each way so 100% results stay finite.
    """
    index = {p: i for i, p in enumerate(players)}
    wins = np.full((len(players),) * 2, 0.0)
    for (a, b), (w, n) in results.items():
        i, j = index[a], index[b]
        wins[i, j] += w + 0.5
        wins[j, i] += n - w + 0.5
    games = wins + wins.T
    strength = np.ones(len(players))
    for _ in range(iterations):   # Hunter's MM algorithm
        denominator = (games / (strength[:, None] + strength[None, :])).sum(1)
        strength = wins.sum(1) / denominator
        strength /= np.exp(np.log(strength).mean())
    return 400 * np.log10(strength)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--panel-games", type=int, default=300, help="per checkpoint, even (default 300)")
    parser.add_argument("--pair-games", type=int, default=40, help="per checkpoint pair, even (default 40)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=Path("ladder_results.json"))
    args = parser.parse_args()

    checkpoints = {}
    for path in args.checkpoints:   # duplicates play once, under the first name
        checkpoints.setdefault(weights_id(path), path.resolve())
    names = {wid: path.name for wid, path in checkpoints.items()}
    for path in args.checkpoints:
        wid = weights_id(path)
        if names[wid] != path.name:
            print(f"{path.name} has the same weights as {names[wid]}; rating them once as {names[wid]}")
    cache = json.loads(args.cache.read_text()) if args.cache.is_file() else {}

    tasks = []
    panel = make_schedule(games=args.panel_games, seed=PANEL_SEED)
    for wid, path in checkpoints.items():
        for style in sorted({m["opponent"]["style"] for m in panel}):
            matches = [m for m in panel if m["opponent"]["style"] == style]
            key = f"{wid}|script:{style}|{len(matches)}|{PANEL_SEED}"
            if key not in cache:
                tasks += [(key, str(path), matches[i:i + CHUNK]) for i in range(0, len(matches), CHUNK)]
    for a, b in itertools.combinations(sorted(checkpoints), 2):
        key = f"{a}|{b}|{args.pair_games}|{PAIR_SEED}"
        if key not in cache:
            schedule = make_schedule(games=args.pair_games, seed=PAIR_SEED, target=str(checkpoints[b]))
            tasks += [(key, str(checkpoints[a]), schedule[i:i + CHUNK]) for i in range(0, len(schedule), CHUNK)]

    if tasks:
        print(f"playing {sum(len(t[2]) for t in tasks)} games on {args.workers} workers", flush=True)
        partial = {}
        with Pool(args.workers) as pool:
            for done, (key, wins, games) in enumerate(pool.imap_unordered(play, tasks), 1):
                w, n = partial.get(key, (0, 0))
                partial[key] = (w + wins, n + games)
                if done % 20 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} tasks", flush=True)
        cache.update({key: list(value) for key, value in partial.items()})
        args.cache.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")

    # Collect this run's results; scripts are players in the rating too.
    results, panel_rows = {}, {wid: {} for wid in checkpoints}
    for key, (w, n) in cache.items():
        a, b, *_ = key.split("|")
        if a not in checkpoints or not (b in checkpoints or b.startswith("script:")):
            continue
        results[(a, b)] = (w, n)
        if b.startswith("script:"):
            panel_rows[a][b[7:]] = (w, n)
    players = sorted(checkpoints, key=lambda wid: (steps(checkpoints[wid]) or 0, names[wid]))
    players += sorted({b for _, b in results if b.startswith("script:")})
    elo = bradley_terry(results, players)
    rng = np.random.default_rng(0)
    boot = np.array([bradley_terry({k: (rng.binomial(n, w / n), n) for k, (w, n) in results.items()},
                                   players, iterations=300) for _ in range(200)])
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)

    styles = sorted({s for row in panel_rows.values() for s in row})
    print(f"\n{'checkpoint':>12} {'steps':>9} {'Elo':>6} {'95% CI':>13}  {'panel':>5}  " + "  ".join(f"{s[:14]:>14}" for s in styles))
    for i, player in enumerate(players):
        if player in checkpoints:
            row = panel_rows[player]
            w, n = sum(v[0] for v in row.values()), sum(v[1] for v in row.values())
            print(f"{names[player]:>12} {steps(checkpoints[player]) / 1e6:>8.2f}M {elo[i]:>6.0f} "
                  f"[{low[i]:>5.0f},{high[i]:>5.0f}]  {w / n:>5.0%}  "
                  + "  ".join(f"{row[s][0] / row[s][1]:>14.0%}" for s in styles))
        else:
            print(f"{player[7:][:12]:>12} {'script':>9} {elo[i]:>6.0f} [{low[i]:>5.0f},{high[i]:>5.0f}]")
    print("\nhead to head (row's win rate vs column):")
    ckpts = [p for p in players if p in checkpoints]
    print(" " * 8 + "".join(f"{names[c][:6]:>7}" for c in ckpts))
    for a in ckpts:
        cells = []
        for b in ckpts:
            if (a, b) in results:
                w, n = results[(a, b)]; cells.append(f"{w / n:>7.0%}")
            elif (b, a) in results:
                w, n = results[(b, a)]; cells.append(f"{1 - w / n:>7.0%}")
            else:
                cells.append(f"{'-':>7}")
        print(f"{names[a][:6]:>8}" + "".join(cells))
    print("\nElo gap -> expected win rate: 50 = 57%, 100 = 64%, 200 = 76%.")


if __name__ == "__main__":
    main()
