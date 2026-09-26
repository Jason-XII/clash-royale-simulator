"""Continue 11.1M for 5M steps against 20% sniper and 80% existing self-play.

Run from src/clasher_new: python train_adaptation.py
"""
import random
from pathlib import Path

from train_legality import main as train_masked
from train_selfplay import HISTORY, HISTORY_FRACTION, SelfPlayMixture
from train_sniper import TARGET, FrozenTarget


HERE = Path(__file__).resolve().parent
SNIPER = HERE / "cr_sniper/cr_14100000_steps.zip"
RUN_DIR = HERE / "cr_adaptation"
ADDITIONAL_STEPS = 5_000_000
SNIPER_FRACTION = 0.20
SEED = 0


class AdaptationMixture:
    """Choose once per episode; retain the existing mixture's proportions."""

    def __init__(self):
        self.sniper = FrozenTarget(SNIPER)
        self.existing = SelfPlayMixture()
        self.opponent = None

    def bind_env(self, env):
        self.sniper.bind_env(env)

    def reset(self):
        self.opponent = self.sniper if random.random() < SNIPER_FRACTION else self.existing
        self.opponent.reset()

    def __call__(self, observation):
        return self.opponent(observation)


def main():
    missing = [str(path) for path in (TARGET, SNIPER, *HISTORY) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))
    train_masked(
        seed=SEED,
        run_dir=RUN_DIR,
        checkpoint=TARGET,
        additional_steps=ADDITIONAL_STEPS,
        opponent_factory=AdaptationMixture,
        arm_name="adaptation",
        experiment_info={
            "frozen_sniper": str(SNIPER),
            "sniper_fraction": SNIPER_FRACTION,
            "existing_mixture_fraction": 1 - SNIPER_FRACTION,
            "opponent_history_fraction": (1 - SNIPER_FRACTION) * HISTORY_FRACTION,
            "opponent_history": [str(path) for path in HISTORY],
            "other_opponent": "CounterMixture",
        },
    )


if __name__ == "__main__":
    main()
