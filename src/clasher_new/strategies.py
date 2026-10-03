"""Rule-based opponents with different play styles for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from defensive_strategy import ELIXIR_COST, ENTITY_NAMES, defensive_strategy, spell_action


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

    action = spell_action(observation)
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

    action = spell_action(observation, threshold=1.25)   # more selective than the others
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

    action = spell_action(observation)
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


MELEE = ("MiniPekka", "Knight")
TANKS = ("Giant", "Knight", "MiniPekka")


def _value(troops):
    """Elixir on the board; units of one card share a cost (two Archers = 3)."""
    return sum(ELIXIR_COST.get(ENTITY_NAMES[name], 0) for name in {t["id"] for t in troops})


def human_style_strategy(observation, low=4, push=9, counter=8, shift=0, depth=0, bridge_y=14,
                         stack=1.0):
    """The user's live style, measured from three recorded wins over the model.

    Bank and play rarely (~20 cards a game, usually from 8-10 elixir). Pull
    attackers to the centre, where both towers shoot, instead of meeting them at
    the tower; jump ranged attackers at the bridge with melee; give each push one
    answer, and none below 4 elixir unless the attacker reaches the towers (the
    user ignored 6 of 8 plays made while they were low); push at the bridge when
    full; no spells.

    Defaults are the measured style. `HumanStyleOpponent` varies them per game:
    elixir thresholds (`low`, `push`, `counter`), centre tiles shifted sideways
    (`shift`) or deeper (`depth`), bridge row, and `stack` > 1 adds a second
    defender until the defence is worth `stack` times the push.
    """
    view = BattleView.from_observation(observation)
    names = ENTITY_NAMES
    crossing = [e for e in view.enemy_troops if e["y"] <= 17]
    if crossing:
        threat = min(crossing, key=lambda e: e["y"])
        left = threat["x"] < 9
        lane = [e for e in crossing if (e["x"] < 9) == left]
        defenders = [e for e in view.own_troops if e["y"] <= 18 and (e["x"] < 11 if left else e["x"] > 6)]
        if _value(defenders) >= stack * _value(lane):
            return (0, 0, 0)                          # one answer per push
        if view.elixir < low and threat["y"] > 8:
            return (0, 0, 0)                          # low: let the tower start on it
        bridge_x, centre_x, corner_x = (4, 8 + shift, 0) if left else (14, 10 - shift, 17)
        name = names[threat["id"]]
        if threat["air"]:
            options = (("Archer", 8 - depth, centre_x), ("Musketeer", 6, corner_x), ("Minions", 7 - depth, centre_x))
        elif name in TANKS:
            options = (("MiniPekka", 9 - depth, centre_x), ("Knight", 10 - depth, centre_x),
                       ("Archer", 8 - depth, centre_x), ("Musketeer", 6, corner_x))
        elif threat["y"] >= 13:                       # ranged attacker at the bridge: jump it
            options = (("MiniPekka", bridge_y, bridge_x), ("Knight", bridge_y, bridge_x),
                       ("Minions", bridge_y, bridge_x), ("Archer", 8 - depth, centre_x))
        else:
            options = (("Knight", 10 - depth, centre_x), ("MiniPekka", 9 - depth, centre_x),
                       ("Minions", 7 - depth, centre_x), ("Archer", 8 - depth, centre_x))
        for card, y, x in options:
            action = view.action(card, y, x)
            if action is not None:
                return action
        return (0, 0, 0)

    enemy_towers = {e["x"] < 9 for e in view.enemies if e["id"] == names.index("King_PrincessTowers")}
    survivors = [e for e in view.own_troops if 14 <= e["y"] <= 22]
    if survivors and view.elixir >= counter:          # back up a counterpush at the bridge
        x = 4 if np.mean([e["x"] for e in survivors]) < 9 else 14
        action = view.first_action(("MiniPekka", "Minions", "Knight"), bridge_y, x)
        if action is not None:
            return action
    if view.elixir < push:
        return (0, 0, 0)
    for left in (True, False):                        # a princess tower is down: pocket
        if left not in enemy_towers:
            action = view.first_action(MELEE, 20, 8 if left else 10)
            if action is not None:
                return action
    action = view.first_action(("MiniPekka", "Minions"), bridge_y, view.lane_x)
    if action is not None:
        return action
    action = (view.action("Knight", 1, 9) or view.action("Archer", 1, 9)
              or view.action("Musketeer", 2, 0 if view.lane_x < 9 else 17))
    if action is not None:
        return action                                 # slow build from behind the king
    if view.elixir >= 10:
        return view.action("Giant", bridge_y, view.lane_x) or (0, 0, 0)
    return (0, 0, 0)


class HumanStyleOpponent:
    """`human_style_strategy` with its thresholds and tiles redrawn every game, so
    the learner has to beat the style rather than one fixed set of numbers."""

    __name__ = "randomized_human"

    def __init__(self):
        self.parameters = None

    def reset(self):
        rng = random.Random(random.getrandbits(64))   # like DiverseOpponent: seeded by the episode
        self.parameters = dict(low=rng.choice((3, 4, 5)), push=rng.choice((8, 9, 10)),
                               counter=rng.choice((7, 8, 9)), shift=rng.choice((-1, 0, 1)),
                               depth=rng.choice((0, 1)), bridge_y=rng.choice((13, 14)),
                               stack=rng.choice((1.0, 1.0, 1.5)))

    def __call__(self, observation):
        if self.parameters is None:
            self.reset()
        return human_style_strategy(observation, **self.parameters)


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

        # Shared spell rule; spell_count 3 styles are more selective.
        priority = ('Arrows', 'Fireball') if self.style == 'spell_control' else ('Fireball', 'Arrows')
        action = spell_action(observation, priority, threshold=1.0 if p['spell_count'] == 2 else 1.25)
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


STRATEGIES = {
    "defensive": defensive_strategy,
    "bridge": bridge_pressure_strategy,
    "split": split_lane_strategy,
    "counterpush": counterpush_strategy,
    "punish": punish_strategy,
    "human": human_style_strategy,
}
