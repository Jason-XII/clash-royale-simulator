"""Continuous self-improvement loop for the masked spatial agent.

Example:
    python train_league.py --run-dir runs/basic --cycles 20 \
        --steps-per-cycle 1000000 --exploit-steps 250000

Each cycle trains against frozen opponents, evaluates the candidate, promotes it
only when it passes the gate, and optionally adds a useful exploiter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_core import OpponentFactory, TrainConfig, evaluate, train_cycle


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--steps-per-cycle", type=int, default=1_000_000)
    parser.add_argument("--exploit-steps", type=int, default=250_000)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--script-fraction", type=float, default=0.60)
    parser.add_argument("--promotion-margin", type=float, default=0.02)
    parser.add_argument("--exploiter-threshold", type=float, default=0.55)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--max-exploiters", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main(args=None):
    args = args or parse_args()
    if args.cycles < 1 or args.steps_per_cycle < 1 or args.games < 1:
        raise ValueError("cycles, steps-per-cycle, and games must be positive")
    if args.exploit_steps < 0:
        raise ValueError("exploit-steps cannot be negative")
    if not 0 <= args.script_fraction <= 1:
        raise ValueError("script-fraction must be between 0 and 1")
    if args.max_history < 1 or args.max_exploiters < 0:
        raise ValueError("population limits are invalid")

    root = args.run_dir
    state_path = root / "league.json"
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"cannot resume without {state_path}")
        state = json.loads(state_path.read_text())
        current = state.get("current")
        history = list(state.get("history", []))
        exploiters = list(state.get("exploiters", []))
        start_cycle = int(state.get("next_cycle", 0))
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"run directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=False)
        current = str(args.initial_checkpoint.resolve()) if args.initial_checkpoint else None
        if current and not Path(current).is_file():
            raise FileNotFoundError(current)
        history = [current] if current else []
        exploiters = []
        start_cycle = 0
        state = {}

    config = TrainConfig(
        workers=args.workers,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        device=args.device,
    )
    config.validate()
    state.update({
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "current": current,
        "history": history,
        "exploiters": exploiters,
        "next_cycle": start_cycle,
    })
    save(state_path, state)

    for cycle in range(start_cycle, start_cycle + args.cycles):
        cycle_dir = root / f"cycle_{cycle:04d}"
        factory = OpponentFactory(
            history=tuple(history),
            exploiters=tuple(exploiters),
            script_fraction=args.script_fraction,
        )
        candidate = train_cycle(
            output=cycle_dir / "candidate",
            steps=args.steps_per_cycle,
            seed=args.seed + cycle,
            opponent_factory=factory,
            config=config,
            initial=current,
        )
        candidate_report = evaluate(candidate, games=args.games, seed=args.seed)
        save(cycle_dir / "candidate_evaluation.json", candidate_report)
        candidate_rate = candidate_report["total"]["win_rate"]
        candidate_valid = candidate_report["total"]["deployment_success_rate"] >= 0.99
        accepted = (
            candidate_valid and (
                current is None
                or candidate_rate >= state.get("current_win_rate", 0.0) - args.promotion_margin
            )
        )
        if accepted:
            current = str(candidate.resolve())
            history = (history + [current])[-args.max_history:]
            state["current_win_rate"] = candidate_rate
        state.update({"current": current, "history": history, "exploiters": exploiters})
        save(cycle_dir / "promotion.json", {
            "accepted": accepted,
            "candidate": str(candidate.resolve()),
            "candidate_win_rate": candidate_rate,
            "candidate_valid_deployment_rate": candidate_report["total"]["deployment_success_rate"],
            "current": current,
            "history_size": len(history),
        })

        if accepted and args.exploit_steps:
            exploiter = train_cycle(
                output=cycle_dir / "exploiter",
                steps=args.exploit_steps,
                seed=args.seed + 100_000 + cycle,
                opponent_factory=OpponentFactory(forced=current),
                config=config,
                initial=current,
            )
            exploit_report = evaluate(
                exploiter,
                games=args.games,
                seed=args.seed + 1,
                opponents=[OpponentFactory(forced=current)()],
            )
            exploit_rate = exploit_report["total"]["win_rate"]
            useful = exploit_rate >= args.exploiter_threshold
            save(cycle_dir / "exploiter_evaluation.json", exploit_report)
            save(cycle_dir / "exploiter_promotion.json", {
                "accepted": useful,
                "exploiter": str(exploiter.resolve()),
                "target": current,
                "win_rate": exploit_rate,
            })
            if useful:
                exploiters = (exploiters + [str(exploiter.resolve())])[-args.max_exploiters:]
                state["exploiters"] = exploiters

        state.update({"current": current, "history": history,
                      "exploiters": exploiters, "next_cycle": cycle + 1})
        save(state_path, state)
        print(
            f"cycle={cycle} candidate={candidate_rate:.1%} "
            f"accepted={accepted} current={current} "
            f"history={len(history)} exploiters={len(exploiters)}"
        )

    return state


if __name__ == "__main__":
    main()
