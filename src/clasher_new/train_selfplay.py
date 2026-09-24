"""Continue masked 7M against 25% frozen past policies and 75% existing opponents.

Run from src/clasher_new: python train_selfplay.py [--seed 0] [--steps 2000000]
"""
import argparse
import random
from pathlib import Path

from stable_baselines3 import PPO

from train_counter import CounterMixture
from train_legality import main as train_masked


HERE = Path(__file__).resolve().parent
START = HERE / "cr_spatial_scratch/cr_7000000_steps.zip"
HISTORY = tuple(HERE / f"cr_spatial_scratch/cr_{steps}_steps.zip"
                for steps in (5_300_000, 5_800_000, 6_500_000))
HISTORY_FRACTION = 0.25


class SelfPlayMixture(CounterMixture):
    """Pick one frozen policy or one existing opponent per episode."""

    def __init__(self):
        super().__init__()
        self.models = {}
        self.using_history = False

    def reset(self):
        self.using_history = random.random() < HISTORY_FRACTION
        if not self.using_history:
            return super().reset()
        path = random.choice(HISTORY)
        if path not in self.models:
            self.models[path] = PPO.load(path, device="cpu")
        self.opponent = self.models[path]
        self.reflected = False

    def __call__(self, observation):
        if self.using_history:
            action, _ = self.opponent.predict(observation, deterministic=False)
            return tuple(map(int, action))
        return super().__call__(observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    missing = [str(path) for path in (START, *HISTORY) if not path.is_file()]
    if missing:
        parser.error("Missing checkpoints: " + ", ".join(missing))
    train_masked(
        seed=args.seed,
        run_dir=args.run_dir or f"cr_spatial_selfplay_seed{args.seed}",
        checkpoint=START,
        additional_steps=args.steps,
        opponent_factory=SelfPlayMixture,
        arm_name="selfplay",
        experiment_info={
            "opponent_history_fraction": HISTORY_FRACTION,
            "opponent_history": [str(path) for path in HISTORY],
            "other_opponent": "CounterMixture",
        },
    )


if __name__ == "__main__":
    main()
