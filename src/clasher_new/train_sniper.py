"""Compatibility entrypoint for the old frozen-target benchmark."""
from pathlib import Path

from train_core import FrozenPolicy

TARGET = Path("cr_spatial_scratch/selfplay/cr_11100000_steps.zip")


class FrozenTarget(FrozenPolicy):
    def __init__(self, checkpoint=None):
        super().__init__(checkpoint or TARGET)


__all__ = ["TARGET", "FrozenTarget"]
