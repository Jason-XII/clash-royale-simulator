"""Compatibility exports for old evaluation scripts.

New training runs use train_core.py and train_league.py directly.
"""
from train_core import COUNTER_FRACTION, CounterMixture

__all__ = ["COUNTER_FRACTION", "CounterMixture"]
