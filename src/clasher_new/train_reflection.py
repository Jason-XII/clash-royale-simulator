"""Compatibility exports for old evaluation scripts.

New training runs use train_core.py and train_league.py directly.
"""
from train_core import CounterMixture, EpisodeReflection

__all__ = ["CounterMixture", "EpisodeReflection"]
