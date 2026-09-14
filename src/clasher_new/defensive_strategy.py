"""A small rule-based opponent for :mod:`environment`.

The strategy uses the compact observation already produced by ``CREnv``.  It
does not inspect ``BattleState`` directly, so it can also be used as a policy
baseline for a learned agent.
"""

from __future__ import annotations

import numpy as np


ENTITY_NAMES = (
    "None", "Knight", "MiniPekka", "Arrows", "Minions", "Archer",
    "Musketeer", "Fireball", "Giant", "King_PrincessTowers", "KingTower",
    "ArrowsSpell", "FireballSpell",
)

# The default deck in environment.py.  Keeping the costs here makes this
# policy independent of card_utils' data-file working-directory assumptions.
ELIXIR_COST = {
    "Knight": 3, "MiniPekka": 4, "Arrows": 3, "Minions": 3,
    "Archer": 3, "Musketeer": 4, "Fireball": 4, "Giant": 5,
}


def _action(hand, name, y, x):
    """Return an action for the first playable copy of ``name``."""
    try:
        slot = list(hand[:4]).index(ENTITY_NAMES.index(name)) + 1
    except (ValueError, IndexError):
        return None
    return slot, int(np.clip(y, 0, 31)), int(np.clip(x, 0, 17))


def defensive_strategy(observation):
    """Defend the near side, then make a measured Giant counter-push.

    ``CREnv.observe`` presents the acting player's arena in a common frame:
    own towers are near ``y=0`` and the opponent approaches from ``y=31``.
    The returned coordinates therefore work for either player through the
    environment's existing coordinate transform.
    """
    hand = np.asarray(observation["hand"])
    elixir = float(np.asarray(observation["elixir"])[0])
    grid = np.asarray(observation["grid"])

    # Row fields are [entity_id, type, owner, elixir, speed, air, ground,
    # air_target, hp_log, hp_fraction, hit_speed, range, sight, damage,
    # projectile_damage].
    rows = grid.reshape(-1, grid.shape[-1])
    entities = rows[rows[:, 0] > 0]
    enemies = entities[(entities[:, 2] == 1) & (entities[:, 1] == 1)]
    own = entities[(entities[:, 2] == 0) & (entities[:, 1] == 1)]

    # Remove the next card from consideration: only cycle[:4] is playable.
    available = {
        ENTITY_NAMES[int(index)]
        for index in hand[:4]
        if 0 <= int(index) < len(ENTITY_NAMES)
    }

    def affordable(name):
        return name in available and elixir >= ELIXIR_COST[name]

    # Spells are the highest-value response to a compact group of attackers.
    if len(enemies) >= 2:
        # Grid coordinates are encoded by array position, not in the feature
        # vector, so recover them from non-empty cells for spell placement.
        cells = np.argwhere((grid[:, :, 0] > 0) & (grid[:, :, 2] == 1) &
                            (grid[:, :, 1] == 1))
        if len(cells):
            y, x = np.mean(cells, axis=0)
            if affordable("Fireball"):
                return _action(hand, "Fireball", y, x)
            if affordable("Arrows"):
                return _action(hand, "Arrows", y, x)

    # A threat below the river is already attacking our side.  Deploy a
    # suitable defender two tiles in front of the nearest tower.
    intruder_cells = np.argwhere((grid[:, :, 0] > 0) & (grid[:, :, 2] == 1) &
                                 (grid[:, :, 1] == 1) &
                                 (np.indices(grid.shape[:2])[0] < 16))
    if len(intruder_cells):
        y, x = np.mean(intruder_cells, axis=0)
        threat_is_air = bool(np.any(grid[intruder_cells[:, 0], intruder_cells[:, 1], 5] > 0))
        priority = (
            ("Musketeer", "Archer", "Minions", "MiniPekka", "Knight")
            if threat_is_air else
            ("MiniPekka", "Knight", "Musketeer", "Archer", "Minions")
        )
        for name in priority:
            if affordable(name):
                return _action(hand, name, max(8, y - 2), x)

    # Support a surviving front-line unit instead of stacking another tank.
    if len(own) and affordable("Musketeer"):
        cells = np.argwhere((grid[:, :, 0] > 0) & (grid[:, :, 2] == 0) &
                            (grid[:, :, 1] == 1))
        if len(cells):
            y, x = np.mean(cells, axis=0)
            return _action(hand, "Musketeer", max(5, y - 2), x)

    # Otherwise save elixir for a simple, repeatable counter-push.
    if elixir >= ELIXIR_COST["Giant"] and affordable("Giant"):
        return _action(hand, "Giant", 10, 9)

    return 0, 0, 0


__all__ = ["defensive_strategy"]
