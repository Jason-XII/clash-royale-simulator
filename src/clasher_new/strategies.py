"""Rule-based opponents with different play styles for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import random

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
        if grid.ndim == 4:
            grid = grid[-1]
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
                "cost": float(row[3]),
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


def _spell_group(view, minimum_size=2, minimum_value=6):
    target, size = view.densest_enemy_group()
    if target is None or size < minimum_size:
        return None
    y, x = target
    group_value = sum(
        entity["cost"] for entity in view.enemy_troops
        if (entity["x"] - x) ** 2 + (entity["y"] - y) ** 2 <= 3.0 ** 2
    )
    if group_value < minimum_value:
        return None
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
    for name in priority:
        y = threat["y"] - 3 if name in ("Musketeer", "Archer") else threat["y"]
        action = view.action(name, int(np.clip(y, 8, 14)), threat["x"])
        if action is not None:
            return action
    # Do not start an attack while an unanswered threat is on our side.
    return (0, 0, 0)


def bridge_pressure_strategy(observation):
    """Build a protected Giant push instead of feeding troops at the bridge."""
    view = BattleView.from_observation(observation)

    action = _spell_group(view, minimum_size=2, minimum_value=6)
    if action is not None:
        return action
    action = _defend(view, 16, ("MiniPekka", "Knight", "Musketeer", "Archer", "Minions"))
    if action is not None:
        return action

    giants = [entity for entity in view.own_troops if entity["id"] == ENTITY_NAMES.index("Giant")]
    if giants:
        giant = max(giants, key=lambda entity: entity["y"])
        action = view.first_action(
            ("Musketeer", "Archer", "Minions", "MiniPekka", "Knight"),
            int(np.clip(giant["y"] - 3, 5, 14)), giant["x"],
        )
        if action is not None:
            return action

    if view.elixir >= 8:
        action = view.action("Giant", 8)
        if action is not None:
            return action
        return view.first_action(
            ("Musketeer", "Archer", "Minions", "Knight", "MiniPekka"), 8
        ) or (0, 0, 0)
    return (0, 0, 0)


def split_lane_strategy(observation):
    """Defend first, then force the quieter lane when enough elixir is banked."""
    view = BattleView.from_observation(observation)

    action = _spell_group(view, minimum_size=2, minimum_value=7)
    if action is not None:
        return action
    action = _defend(view, 16, ("MiniPekka", "Knight", "Musketeer", "Minions", "Archer"))
    if action is not None:
        return action

    if view.elixir >= 8:
        enemy_left = sum(entity["x"] < 9 for entity in view.enemy_troops)
        enemy_right = len(view.enemy_troops) - enemy_left
        pressure_x = 14 if enemy_left > enemy_right else 3
        action = view.first_action(
            ("MiniPekka", "Knight", "Minions", "Musketeer", "Archer"),
            13,
            pressure_x,
        )
        if action is not None:
            return action
    return (0, 0, 0)


def counterpush_strategy(observation):
    """Bank elixir, defend deeply, then place a Giant in front of survivors."""
    view = BattleView.from_observation(observation)

    action = _spell_group(view, minimum_size=2, minimum_value=6)
    if action is not None:
        return action
    action = _defend(view, 17, ("MiniPekka", "Knight", "Musketeer", "Minions", "Archer"))
    if action is not None:
        return action

    survivors = [entity for entity in view.own_troops if entity["y"] >= 8]
    if survivors and view.elixir >= 8:
        front = max(survivors, key=lambda entity: entity["y"])
        action = view.action("Giant", min(14, front["y"] + 2), front["x"])
        if action is not None:
            return action
    if view.elixir >= 8:
        action = view.action("Giant", 8)
        if action is not None:
            return action
        return view.first_action(
            ("Musketeer", "Archer", "Minions", "Knight", "MiniPekka"), 8
        ) or (0, 0, 0)
    return (0, 0, 0)


def punish_strategy(observation):
    """Bait predictable bridge pressure into tower-supported defense.

    Wait until attackers reach y=10, deploy melee directly onto them and
    ranged units behind them, then bank elixir for a supported Giant push.
    Uses only the observation and the standard eight-card deck.
    """
    return defensive_strategy(observation, defend_y=10, min_defend_y=2)


class DiverseOpponent:
    def __init__(self, style):
        self.style = style
        self.__name__ = 'randomized_' + style
        self.parameters = None

    def reset(self):
        # Separate RNG: action frequency does not change later deck shuffles.
        rng = random.Random(random.getrandbits(64))
        self.parameters = dict(
            depth=rng.choice((8, 9, 10, 11, 12)),
            gap=rng.choice((2, 3, 4)),
            offset=rng.choice((-1, 0, 1)),
            reserve=rng.choice((7, 8, 9)),
            push_y=rng.choice((6, 8, 10)),
            spell_count=rng.choice((2, 3)),
            counter_reserve=rng.choice((5, 6, 7)),
            lane=rng.choice((3, 14)),
        )
        if self.style != 'opposite_lane':
            # Broad placement jitter weakened tower defense in benchmarks.
            # Preserve its geometry; vary engagement depth and push timing.
            self.parameters.update(gap=3, offset=0, reserve=8, push_y=8, spell_count=3)
            self.parameters['depth'] = min(self.parameters['depth'], 10)

    def __call__(self, observation):
        if self.parameters is None:
            self.reset()
        p = self.parameters
        view = BattleView.from_observation(observation)
        names = ENTITY_NAMES

        # Match the established defender's spell rule, with varied selectivity.
        target, count = view.densest_enemy_group(radius=2.5)
        if target is not None and count >= p['spell_count']:
            priority = ('Arrows', 'Fireball') if self.style == 'spell_control' else ('Fireball', 'Arrows')
            action = view.first_action(priority, *target)
            if action is not None:
                return action

        threats = [e for e in view.enemy_troops if e['y'] <= p['depth']]
        if threats:
            threat = min(threats, key=lambda e: e['y'])
            priority = (('Musketeer', 'Archer', 'Minions') if threat['air'] else
                        ('MiniPekka', 'Knight', 'Musketeer', 'Archer', 'Minions'))
            for name in priority:
                y = threat['y'] - p['gap'] if name in ('Musketeer', 'Archer') else threat['y']
                action = view.action(name, np.clip(y, 2, 14),
                                     np.clip(threat['x'] + p['offset'], 1, 16))
                if action is not None:
                    return action
            return (0, 0, 0)

        giants = [e for e in view.own_troops if e['id'] == names.index('Giant')]
        if giants:
            giant = max(giants, key=lambda e: e['y'])
            action = view.first_action(('Musketeer', 'Archer', 'Minions'),
                                       np.clip(giant['y'] - p['gap'], 5, 14), giant['x'])
            if action is not None:
                return action

        if self.style == 'counterpush' and not giants:
            survivors = [e for e in view.own_troops if 8 <= e['y'] <= 17]
            if survivors and view.elixir >= p['counter_reserve']:
                front = max(survivors, key=lambda e: e['y'])
                action = view.action('Giant', min(14, front['y'] + 2), front['x'])
                if action is not None:
                    return action

        if view.elixir < p['reserve']:
            return (0, 0, 0)
        if self.style == 'opposite_lane':
            left = sum(e['cost'] * e['hp'] for e in view.enemy_troops if e['x'] < 9)
            right = sum(e['cost'] * e['hp'] for e in view.enemy_troops if e['x'] >= 9)
            lane = 14 if left > right else 3 if right > left else p['lane']
            return view.first_action(('MiniPekka', 'Knight', 'Minions', 'Musketeer', 'Archer'),
                                     13, lane) or (0, 0, 0)
        return view.first_action(('Giant', 'Musketeer', 'Archer', 'Minions', 'Knight', 'MiniPekka'),
                                 p['push_y'], view.lane_x) or (0, 0, 0)


def make_diverse_opponents():
    """New instances per environment or evaluation suite; style lasts one game."""
    return [DiverseOpponent(style) for style in
            ('deep_defense', 'counterpush', 'opposite_lane', 'spell_control')]


def make_opponent_pool():
    """Independent stateful opponents for each environment."""
    return list(STRATEGIES.values()) + make_diverse_opponents()


STRATEGIES = {
    "defensive": defensive_strategy,
    "bridge": bridge_pressure_strategy,
    "split": split_lane_strategy,
    "counterpush": counterpush_strategy,
    "punish": punish_strategy,
}


__all__ = [
    "bridge_pressure_strategy",
    "split_lane_strategy",
    "counterpush_strategy",
    "punish_strategy",
    "DiverseOpponent",
    "make_diverse_opponents",
    "make_opponent_pool",
    "STRATEGIES",
]
