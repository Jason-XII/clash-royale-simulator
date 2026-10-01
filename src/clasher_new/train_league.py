"""Continuous self-improvement loop for the masked spatial agent.

Example:
    python train_league.py --run-dir runs/basic --cycles 20 \
        --steps-per-cycle 1000000 --exploit-steps 250000

Each cycle trains against frozen opponents, evaluates the candidate, promotes it
only when it passes the gate, and optionally adds a useful exploiter.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import time_ns

from train_core import OpponentFactory, TrainConfig, train_cycle
from league_evaluation import (
    evaluate_schedule, exploiter_decision, make_schedule, promotion_decision,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--initial-checkpoint", type=Path,
                      help="Start a new run from policy weights, with a fresh optimizer")
    mode.add_argument("--resume", action="store_true",
                      help="Continue the run using its saved settings")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--steps-per-cycle", type=int, default=1_000_000)
    parser.add_argument("--exploit-steps", type=int, default=250_000)
    parser.add_argument("--games", type=int, default=50,
                        help="Total games per policy, split equally between sides (default: 50)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--script-fraction", type=float, default=0.60)
    parser.add_argument("--promotion-margin", type=float, default=0.02,
                        help="Minimum positive paired win-rate gain for promotion")
    parser.add_argument("--rollback-margin", type=float, default=0.10)
    parser.add_argument("--significance", type=float, default=0.05)
    parser.add_argument("--exploiter-threshold", type=float, default=0.55)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--max-exploiters", type=int, default=4)
    parser.add_argument("--device", default="auto")
    preliminary, _ = parser.parse_known_args()
    if preliminary.resume:
        state_path = preliminary.run_dir / "league.json"
        if not state_path.is_file():
            parser.error(f"cannot resume without {state_path}")
        saved = json.loads(state_path.read_text()).get("settings", {})
        defaults = {key: value for key, value in saved.items() if key != "training"}
        defaults.update({key: value for key, value in saved.get("training", {}).items()
                         if key in ("workers", "rollout_steps", "batch_size", "device")})
        parser.set_defaults(**defaults)
    return parser.parse_args()


def main(args=None):
    args = args or parse_args()
    root = args.run_dir.resolve()
    state_path = root / "league.json"
    state = json.loads(state_path.read_text()) if args.resume else {}
    if args.resume and state.get("version") != 3:
        raise ValueError("This league predates bank actions; start a new run directory")
    if args.cycles < 1 or args.steps_per_cycle < 1 or args.games < 1:
        raise ValueError("cycles, steps-per-cycle, and games must be positive")
    if args.exploit_steps < 0:
        raise ValueError("exploit-steps cannot be negative")
    if not 0 <= args.script_fraction <= 1:
        raise ValueError("script-fraction must be between 0 and 1")
    if args.max_history < 1 or args.max_exploiters < 0:
        raise ValueError("population limits are invalid")
    if args.games < 6 or args.games % 2:
        raise ValueError("games must be even and at least 6")
    if not 0 < args.promotion_margin <= 1 or not 0 < args.rollback_margin <= 1:
        raise ValueError("promotion and rollback margins must be in (0, 1]")
    if not 0 < args.significance < .5 or not .5 <= args.exploiter_threshold <= 1:
        raise ValueError("invalid statistical gate settings")

    config = TrainConfig(**state["settings"]["training"]) if args.resume else TrainConfig()
    config.workers, config.rollout_steps = args.workers, args.rollout_steps
    config.batch_size, config.device = args.batch_size, args.device
    config.validate()
    settings = {name: getattr(args, name) for name in (
        "seed", "games", "script_fraction", "promotion_margin", "rollback_margin",
        "significance", "exploiter_threshold", "max_history", "max_exploiters",
        "steps_per_cycle", "exploit_steps",
    )}
    settings["training"] = asdict(config)

    if args.resume:
        if state.get("settings") != settings:
            raise ValueError("Resume settings differ from league.json; omit overrides to use saved settings")
        current = state.get("current")
        learner = state.get("learner")
        history = list(state.get("history", []))
        exploiters = list(state.get("exploiters", []))
        references = list(state.get("references", []))
        start_cycle = int(state.get("next_cycle", 0))
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"run directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        current = str(args.initial_checkpoint.resolve()) if args.initial_checkpoint else None
        if current and not Path(current).is_file():
            raise FileNotFoundError(current)
        history = [current] if current else []
        references = [current] if current else []
        learner = current
        exploiters = []
        start_cycle = 0
        state = {"initial_checkpoint": current}

    state.update({
        "version": 3, "settings": settings,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "current": current,
        "learner": learner,
        "history": history,
        "references": references,
        "exploiters": exploiters,
        "next_cycle": start_cycle,
    })
    save(state_path, state)

    for cycle in range(start_cycle, start_cycle + args.cycles):
        cycle_dir = root / f"cycle_{cycle:04d}"
        if cycle_dir.exists():
            if not args.resume:
                raise FileExistsError(f"cycle directory already exists: {cycle_dir}")
            archived = root / "interrupted" / f"{cycle_dir.name}_{time_ns()}"
            archived.parent.mkdir(parents=True, exist_ok=True)
            cycle_dir.rename(archived)
            print(f"Preserved incomplete cycle in {archived}; restarting from {learner}", flush=True)
        schedule = make_schedule(
            games=args.games, seed=args.seed + cycle * 1_000_003,
            history=history, exploiters=exploiters, references=references,
        )
        save(cycle_dir / "evaluation_schedule.json", schedule)
        previous_champion = current
        champion_report = None
        if current:
            print(f"cycle={cycle} evaluating champion: {args.games} games", flush=True)
            champion_report = evaluate_schedule(current, schedule)
            save(cycle_dir / "champion_evaluation.json", champion_report)
            if cycle == 0:
                # This is also the initializer baseline; no extra matches.
                save(root / "initial_evaluation.json", champion_report)
        factory = OpponentFactory(
            history=tuple(dict.fromkeys(references + history)),
            exploiters=tuple(exploiters),
            script_fraction=args.script_fraction,
        )
        candidate = train_cycle(
            output=cycle_dir / "candidate",
            steps=args.steps_per_cycle,
            seed=args.seed + cycle,
            opponent_factory=factory,
            config=config,
            initial=learner,
            initialize=bool(learner and learner == state.get("initial_checkpoint")),
        )
        print(f"cycle={cycle} evaluating candidate: {args.games} games", flush=True)
        candidate_report = evaluate_schedule(candidate, schedule)
        save(cycle_dir / "candidate_evaluation.json", candidate_report)
        candidate_rate = candidate_report["total"]["win_rate"]
        candidate_valid = candidate_report["total"]["deployment_success_rate"] >= 0.99
        if champion_report is None:
            decision = {
                "status": "baseline" if candidate_valid else "invalid_baseline",
                "accepted": False,
                "reason": "first_checkpoint_establishes_baseline" if candidate_valid
                          else "invalid_deployments_or_no_deployments",
            }
        else:
            decision = promotion_decision(
                candidate_report, champion_report, margin=args.promotion_margin,
                rollback_margin=args.rollback_margin, alpha=args.significance,
            )
        installed = decision["status"] in ("baseline", "promoted")
        if installed:
            current = str(candidate.resolve())
            history = (history + [current])[-args.max_history:]
            if len(references) < 2:
                references = list(dict.fromkeys(references + [current]))
        learner = current if decision["status"] == "rollback" else str(candidate.resolve())
        save(cycle_dir / "promotion.json", {
            **decision,
            "installed_as_champion": installed,
            "candidate": str(candidate.resolve()),
            "candidate_win_rate": candidate_rate,
            "candidate_valid_deployment_rate": candidate_report["total"]["deployment_success_rate"],
            "current": current,
            "previous_champion": previous_champion,
            "learner": learner,
            "history_size": len(history),
        })

        if installed and args.exploit_steps and args.max_exploiters:
            exploiter = train_cycle(
                output=cycle_dir / "exploiter",
                steps=args.exploit_steps,
                seed=args.seed + 100_000 + cycle,
                opponent_factory=OpponentFactory(forced=current),
                config=config,
                initial=current,
            )
            exploit_schedule = make_schedule(
                games=args.games, seed=args.seed + 500_000_003 + cycle * 1_000_003,
                target=current,
            )
            save(cycle_dir / "exploiter_schedule.json", exploit_schedule)
            print(f"cycle={cycle} evaluating exploiter: {args.games} additional games", flush=True)
            exploit_report = evaluate_schedule(exploiter, exploit_schedule)
            exploit_decision = exploiter_decision(
                exploit_report, threshold=args.exploiter_threshold, alpha=args.significance,
            )
            save(cycle_dir / "exploiter_evaluation.json", exploit_report)
            save(cycle_dir / "exploiter_promotion.json", {
                **exploit_decision,
                "exploiter": str(exploiter.resolve()),
                "target": current,
            })
            if exploit_decision["accepted"]:
                exploiters = (exploiters + [str(exploiter.resolve())])[-args.max_exploiters:]

        state.update({"current": current, "learner": learner,
                      "references": references, "history": history,
                      "exploiters": exploiters, "next_cycle": cycle + 1})
        save(state_path, state)
        print(
            f"cycle={cycle} candidate={candidate_rate:.1%} "
            f"status={decision['status']} current={current} learner={learner} "
            f"history={len(history)} exploiters={len(exploiters)}"
        )

    return state


if __name__ == "__main__":
    main()
