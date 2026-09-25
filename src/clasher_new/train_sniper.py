"""Fine-tune an adversary against the frozen masked 11.1M policy.

Run from src/clasher_new:
python train_sniper.py --steps 2000000 --run-dir cr_spatial_sniper_seed0
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from card_utils import Card
from core import Position
from train_legality import main as train_masked


HERE = Path(__file__).resolve().parent
TARGET = HERE / "cr_spatial_scratch/selfplay/cr_11100000_steps.zip"


class FrozenTarget:
    """Sample the frozen target with exact player-1 legality."""

    def __init__(self):
        self.model = PPO.load(TARGET, device="cpu")
        self.env = None

    def bind_env(self, env):
        self.env = env

    def reset(self):
        pass

    def __call__(self, observation):
        battle = self.env.battle
        player = battle.players[1]
        mask = np.zeros((4, 32, 18), dtype=np.int8)
        playable = [player.can_play_card(card) for card in player.cycle[:4]]
        if any(playable):
            tiles = np.array([
                [battle.can_place_troop(1, Position(17.5 - x, 31.5 - y))
                 for x in range(18)] for y in range(32)
            ], dtype=np.int8)
            for slot, card in enumerate(player.cycle[:4]):
                if playable[slot]:
                    mask[slot] = 1 if Card(card).type == "spell" else tiles
        action, _ = self.model.predict(
            dict(observation, legal_mask=mask), deterministic=False
        )
        return tuple(map(int, action))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not TARGET.is_file():
        parser.error(f"Missing target checkpoint: {TARGET}")
    train_masked(
        seed=args.seed,
        run_dir=args.run_dir or f"cr_spatial_sniper_seed{args.seed}",
        checkpoint=TARGET,
        additional_steps=args.steps,
        opponent_factory=FrozenTarget,
        arm_name="sniper",
        experiment_info={
            "role": "adversary",
            "frozen_target": str(TARGET),
            "promotion_rule": "at least 60% wins in 200 side-balanced games",
        },
    )


if __name__ == "__main__":
    main()
