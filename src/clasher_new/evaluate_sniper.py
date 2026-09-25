"""Side-balanced head-to-head evaluation for a trained sniper checkpoint."""
import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET = HERE / "cr_spatial_scratch/selfplay/cr_11100000_steps.zip"
EVALUATOR = HERE / "cr_spatial_scratch/evaluation_20260925/evaluate.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=120000)
    parser.add_argument("--output", type=Path, default=Path("sniper_evaluation.json"))
    args = parser.parse_args()
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number")
    for path in (args.checkpoint, args.target, EVALUATOR):
        if not path.is_file():
            parser.error(f"Missing file: {path}")

    spec = importlib.util.spec_from_file_location("sniper_match_eval", EVALUATOR)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    evaluator.PATHS.update(sniper=args.checkpoint, target=args.target)
    rows = [
        evaluator.play("sniper", "target", args.seed + index // 2, index % 2)
        for index in range(args.games)
    ]
    summary = evaluator.summarize(rows)
    result = {
        "sniper": str(args.checkpoint.resolve()),
        "target": str(args.target.resolve()),
        "promotion_threshold": 0.60,
        "promoted": summary["wins"] / args.games >= 0.60,
        "summary": summary,
        "games": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"sniper wins {summary['wins']}/{args.games}; "
        f"by side {summary['by_side']}; promoted {result['promoted']}"
    )


if __name__ == "__main__":
    main()
