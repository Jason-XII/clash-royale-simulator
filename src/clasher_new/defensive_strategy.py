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


def defensive_strategy(observation, *, defend_y=16, min_defend_y=8):
    """Defend the near side, then build a patient Giant counter-push.

    ``CREnv.observe`` presents the acting player's arena in a common frame:
    own towers are near ``y=0`` and the opponent approaches from ``y=31``.
    The returned coordinates therefore work for either player through the
    environment's existing coordinate transform.

    ``defend_y`` sets the engagement line; ``min_defend_y`` allows deeper
    placements when this policy is reused by the tower-defense opponent.
    """
    hand = np.asarray(observation["hand"])
    elixir = float(np.asarray(observation["elixir"])[0])
    grid = np.asarray(observation["grid"])
    if grid.ndim == 4:
        grid = grid[-1]

    cells = []
    for y, x in np.argwhere(grid[:, :, 0] > 0):
        row = grid[y, x]
        cells.append({
            "id": int(row[0]),
            "type": int(row[1]),
            "owner": int(row[2]),
            "air": bool(row[5]),
            "hp": float(row[9]),
            "y": int(y),
            "x": int(x),
        })

    enemy_troops = [
        entity for entity in cells
        if entity["owner"] == 1 and entity["type"] == 1
    ]
    own_troops = [
        entity for entity in cells
        if entity["owner"] == 0 and entity["type"] == 1
    ]
    available = {
        ENTITY_NAMES[int(index)]
        for index in hand[:4]
        if 0 <= int(index) < len(ENTITY_NAMES)
        and ENTITY_NAMES[int(index)] in ELIXIR_COST
        and elixir >= ELIXIR_COST[ENTITY_NAMES[int(index)]]
    }

    def first_action(names, y, x):
        for name in names:
            if name in available:
                return _action(hand, name, y, x)
        return None

    # Spend a spell only when at least three enemies form one local cluster.
    if len(enemy_troops) >= 3:
        best_group = []
        for center in enemy_troops:
            group = [
                entity for entity in enemy_troops
                if (entity["x"] - center["x"]) ** 2
                + (entity["y"] - center["y"]) ** 2 <= 2.5 ** 2
            ]
            if len(group) > len(best_group):
                best_group = group
        if len(best_group) >= 3:
            target_y = round(sum(entity["y"] for entity in best_group) / len(best_group))
            target_x = round(sum(entity["x"] for entity in best_group) / len(best_group))
            action = first_action(("Fireball", "Arrows"), target_y, target_x)
            if action is not None:
                return action

    # Answer the deepest intruder with a suitable counter. Ranged defenders
    # stay behind the threat, while melee defenders meet it directly.
    intruders = [entity for entity in enemy_troops if entity["y"] <= defend_y]
    if intruders:
        threat = min(intruders, key=lambda entity: entity["y"])
        priority = (
            ("Musketeer", "Archer", "Minions")
            if threat["air"] else
            ("MiniPekka", "Knight", "Musketeer", "Archer", "Minions")
        )
        for name in priority:
            if name not in available:
                continue
            target_y = threat["y"] - 3 if name in ("Musketeer", "Archer") else threat["y"]
            return _action(hand, name, int(np.clip(target_y, min_defend_y, 14)), threat["x"])
        return 0, 0, 0

    # Add ranged support behind an existing Giant instead of supporting every
    # surviving troop and accidentally turning defense into bridge spam.
    giants = [
        entity for entity in own_troops
        if entity["id"] == ENTITY_NAMES.index("Giant")
    ]
    if giants:
        giant = max(giants, key=lambda entity: entity["y"])
        action = first_action(
            ("Musketeer", "Archer", "Minions"),
            int(np.clip(giant["y"] - 3, 5, 14)),
            giant["x"],
        )
        if action is not None:
            return action

    # Bank elixir, then start a push in the weaker enemy princess-tower lane.
    if elixir < 8:
        return 0, 0, 0
    enemy_towers = [
        entity for entity in cells
        if entity["owner"] == 1
        and entity["id"] == ENTITY_NAMES.index("King_PrincessTowers")
        and entity["hp"] > 0
    ]
    target_x = min(enemy_towers, key=lambda entity: (entity["hp"], entity["x"]))["x"] \
        if enemy_towers else 3
    action = first_action(
        ("Giant", "Musketeer", "Archer", "Minions", "Knight", "MiniPekka"),
        8,
        target_x,
    )
    if action is not None:
        return action

    return 0, 0, 0


__all__ = ["defensive_strategy"]
