"""Rule-based opponents with different play styles for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from defensive_strategy import ELIXIR_COST, ENTITY_NAMES, defensive_strategy


@dataclass
class BattleView:
    grid: np.ndarray
    hand: np.ndarray
    elixir: float
    own: list
    enemies: list
    own_troops: list
    enemy_troops: list
    lane_x: int

    @classmethod
    def from_observation(cls, observation):
        grid = np.asarray(observation["grid"])
        hand = np.asarray(observation["hand"])
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

        own = [entity for entity in cells if entity["owner"] == 0]
        enemies = [entity for entity in cells if entity["owner"] == 1]
        own_troops = [entity for entity in own if entity["type"] == 1]
        enemy_troops = [entity for entity in enemies if entity["type"] == 1]
        enemy_towers = [
            entity for entity in enemies
            if entity["id"] == ENTITY_NAMES.index("King_PrincessTowers") and entity["hp"] > 0
        ]
        target = min(enemy_towers, key=lambda entity: (entity["hp"], entity["x"]), default=None)
        lane_x = 3 if target is None or target["x"] < 9 else 14
        return cls(
            grid=grid,
            hand=hand,
            elixir=float(np.asarray(observation["elixir"])[0]),
            own=own,
            enemies=enemies,
            own_troops=own_troops,
            enemy_troops=enemy_troops,
            lane_x=lane_x,
        )

    def action(self, name, y, x=None):
        if self.elixir < ELIXIR_COST[name]:
            return None
        try:
            slot = list(self.hand[:4]).index(ENTITY_NAMES.index(name)) + 1
        except (ValueError, IndexError):
            return None
        return slot, int(np.clip(y, 0, 31)), int(np.clip(self.lane_x if x is None else x, 0, 17))

    def first_action(self, names, y, x=None):
        for name in names:
            action = self.action(name, y, x)
            if action is not None:
                return action
        return None

    def densest_enemy_group(self, radius=3.0):
        if not self.enemy_troops:
            return None, 0
        best = None
        best_group = []
        for center in self.enemy_troops:
            group = [
                entity for entity in self.enemy_troops
                if (entity["x"] - center["x"]) ** 2 + (entity["y"] - center["y"]) ** 2 <= radius ** 2
            ]
            if len(group) > len(best_group):
                best = center
                best_group = group
        x = round(sum(entity["x"] for entity in best_group) / len(best_group))
        y = round(sum(entity["y"] for entity in best_group) / len(best_group))
        return (y, x), len(best_group)


def _spell_group(view, minimum_size=2):
    target, size = view.densest_enemy_group()
    if target is None or size < minimum_size:
        return None
    y, x = target
    return view.first_action(("Fireball", "Arrows"), y, x)


def _defend(view, trigger_y, ground_priority):
    threats = [entity for entity in view.enemy_troops if entity["y"] <= trigger_y]
    if not threats:
        return None
    threat = min(threats, key=lambda entity: (entity["y"], entity["hp"]))
    if threat["air"]:
        priority = ("Musketeer", "Minions", "Archer")
    else:
        priority = ground_priority
    # Meet attackers above the towers while avoiding their occupied tiles.
    return view.first_action(priority, max(9, threat["y"] - 2), threat["x"])


def bridge_pressure_strategy(observation):
    """Build one tank-and-support push in the weaker enemy lane."""
    view = BattleView.from_observation(observation)

    action = _spell_group(view, minimum_size=2)
    if action is not None:
        return action
    action = _defend(view, 12, ("MiniPekka", "Knight", "Musketeer", "Archer", "Minions"))
    if action is not None:
        return action

    giants = [entity for entity in view.own_troops if entity["id"] == ENTITY_NAMES.index("Giant")]
    if giants:
        giant = max(giants, key=lambda entity: entity["y"])
        action = view.first_action(
            ("Musketeer", "Minions", "Archer", "MiniPekka", "Knight"),
            max(8, giant["y"] - 3), giant["x"],
        )
        if action is not None:
            return action

    if view.elixir >= 8:
        action = view.action("Giant", 13)
        if action is not None:
            return action
    action = view.first_action(("MiniPekka", "Knight", "Minions", "Musketeer", "Archer"), 14)
    return action or (0, 0, 0)


def split_lane_strategy(observation):
    """Apply cheap pressure opposite the lane containing the most friendly troops."""
    view = BattleView.from_observation(observation)

    action = _defend(view, 10, ("Knight", "MiniPekka", "Minions", "Archer", "Musketeer"))
    if action is not None:
        return action
    action = _spell_group(view, minimum_size=3)
    if action is not None:
        return action

    left = sum(entity["x"] < 9 for entity in view.own_troops)
    right = len(view.own_troops) - left
    pressure_x = 14 if left > right else 3
    action = view.first_action(("MiniPekka", "Knight", "Minions", "Archer", "Musketeer"), 14, pressure_x)
    if action is not None:
        return action
    if view.elixir >= 9:
        action = view.action("Giant", 12, pressure_x)
        if action is not None:
            return action
    return (0, 0, 0)


def counterpush_strategy(observation):
    """Bank elixir, defend deeply, then place a Giant in front of survivors."""
    view = BattleView.from_observation(observation)

    action = _spell_group(view, minimum_size=2)
    if action is not None:
        return action
    action = _defend(view, 15, ("MiniPekka", "Knight", "Musketeer", "Minions", "Archer"))
    if action is not None:
        return action

    survivors = [entity for entity in view.own_troops if entity["y"] >= 9]
    if survivors and view.elixir >= 7:
        front = max(survivors, key=lambda entity: entity["y"])
        action = view.action("Giant", min(14, front["y"] + 2), front["x"])
        if action is not None:
            return action
    if view.elixir >= 9.5:
        action = view.action("Giant", 8)
        if action is not None:
            return action
        return view.first_action(("Knight", "MiniPekka", "Musketeer", "Minions", "Archer"), 10) or (0, 0, 0)
    return (0, 0, 0)


STRATEGIES = {
    "defensive": defensive_strategy,
    "bridge": bridge_pressure_strategy,
    "split": split_lane_strategy,
    "counterpush": counterpush_strategy,
}


__all__ = [
    "bridge_pressure_strategy",
    "split_lane_strategy",
    "counterpush_strategy",
    "STRATEGIES",
]
