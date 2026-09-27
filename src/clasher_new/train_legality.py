"""Compatibility wrapper for the former one-off legality continuation.

Use train_league.py for new runs. This file remains only for old benchmark
scripts that import ``make_env``.
"""
import argparse
from pathlib import Path

from masked_spatial import LegalPlacement
from environment import CREnv
from train_core import (
    EpisodeReflection,
    OpponentFactory,
    TrainConfig,
    train_cycle,
)

CHECKPOINT = Path("cr_spatial_scratch/cr_6500000_steps.zip")


def make_env(rank, seed, control=False, opponent_factory=None):
    if opponent_factory is None:
        opponent_factory = OpponentFactory()

    def factory():
        env = CREnv(opponent_model=opponent_factory())
        if callable(getattr(env.opponent, "bind_env", None)):
            env.opponent.bind_env(env)
        env = EpisodeReflection(env, seed=10_000 + seed * 1_000 + rank)
        return env if control else LegalPlacement(env)

    return factory


def main(control=False, seed=0, run_dir=None, checkpoint=CHECKPOINT,
         additional_steps=5_000_000, opponent_factory=None,
         experiment_info=None, arm_name=None):
    config = TrainConfig()
    factory = opponent_factory or OpponentFactory()
    output = run_dir or f"cr_spatial_{arm_name or 'legal'}"
    train_cycle(
        output=output,
        steps=additional_steps,
        seed=seed,
        opponent_factory=factory if hasattr(factory, "__call__") else OpponentFactory(),
        config=config,
        initial=checkpoint,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(seed=args.seed)
