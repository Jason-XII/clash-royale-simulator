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
                         stack=1.0, giant=False, ignore_ranged=False, switch=7):
    """The user's live style, measured from five recorded wins over the model.

    Answer each push as it crosses the river (median 0.5 s after), usually with
    one card for an even trade: melee in the attacker's path about 2.5 tiles past
    the river (Knight 3 tiles toward the centre), Minions onto melee attackers,
    Archer in the centre, Musketeer in the corner by the tower. The defenders
    survive and walk on as a counterpush (72% of pushes), backed up at the bridge
    when elixir allows. Below `low` elixir, let the tower start on an attacker.
    Bank otherwise; push the bridge when full; pocket after a tower falls; no
    spells. `giant` plays the user's Giant pushes: from behind the king when
    full, supported at the bridge as it crosses. `ignore_ranged` leaves a lone
    Archer or Musketeer to the tower, as the user did with a quarter of them.
    `switch`: once the learner has this much elixir of troops in one lane and
    none in the other, push the empty lane at the bridge (how the user took a
    tower from cycle6: the learner kept stacking a defended lane and ran dry).

    `HumanStyleOpponent` varies the thresholds, tiles (`shift` sideways toward
    the centre, `depth` deeper), bridge row, `stack` > 1 (a second defender until
    the defence is worth `stack` times the push), and the two style flags.
    """
    view = BattleView.from_observation(observation)
    names = ENTITY_NAMES
    stacked_left = [e for e in view.enemy_troops if e["x"] < 9]
    stacked_right = [e for e in view.enemy_troops if e["x"] >= 9]
    empty = None                                      # the lane to switch to, if any
    if _value(stacked_left) >= switch and not stacked_right:
        empty = 14
    elif _value(stacked_right) >= switch and not stacked_left:
        empty = 4
    crossing = [e for e in view.enemy_troops if e["y"] <= 17]
    if crossing:
        threat = min(crossing, key=lambda e: e["y"])
        left = threat["x"] < 9
        lane = [e for e in crossing if (e["x"] < 9) == left]
        defenders = [e for e in view.own_troops if e["y"] <= 18 and (e["x"] < 11 if left else e["x"] > 6)]
        if _value(defenders) >= stack * _value(lane):
            if empty is not None:                     # defence holds: punish the other lane
                return view.first_action(("MiniPekka", "Knight", "Minions"), bridge_y, empty) or (0, 0, 0)
            return (0, 0, 0)                          # one answer per push
        if view.elixir < low and threat["y"] > 8:
            return (0, 0, 0)                          # low: let the tower start on it
        name = names[threat["id"]]
        if ignore_ranged and len(lane) <= 2 and name in ("Archer", "Musketeer") and threat["y"] > 9:
            return (0, 0, 0)
        tx, ty = threat["x"], threat["y"]
        inward = 1 if left else -1
        path_y = int(np.clip(ty - 2.5 - depth, 9, 14))
        path = ("MiniPekka", path_y, int(np.clip(tx + inward * (1 + shift), 1, 16)))
        knight = ("Knight", int(np.clip(path_y, 9, 13)), int(np.clip(tx + inward * (3 + shift), 1, 16)))
        minions = ("Minions", int(np.clip(ty - 5 - depth, 7, 14)), int(np.clip(tx, 1, 16)))
        archer = ("Archer", 8 - depth, (8 if left else 10) + inward * shift)
        musketeer = ("Musketeer", 6, 0 if left else 17)
        if threat["air"]:
            options = (musketeer, archer, minions)
        elif name in ("Archer", "Musketeer"):         # jump ranged attackers in their path
            options = (path, knight, minions, archer)
        elif name == "MiniPekka":
            options = (knight, path, minions, archer)
        elif name == "Knight":
            options = (knight, archer, minions, path, musketeer)
        else:                                         # Giant and anything else heavy
            options = (path, minions, knight, archer, musketeer)
        for card, y, x in options:
            action = view.action(card, y, x)
            if action is not None:
                return action
        return (0, 0, 0)

    if empty is not None:                             # they stacked one lane: hit the other
        action = view.first_action(("MiniPekka", "Knight", "Minions"), bridge_y, empty)
        if action is not None:
            return action
    enemy_towers = {e["x"] < 9 for e in view.enemies if e["id"] == names.index("King_PrincessTowers")}
    giants = [e for e in view.own_troops if e["id"] == names.index("Giant") and 10 <= e["y"] <= 20]
    if giants and view.elixir >= 4:                   # support the Giant as it crosses
        x = 4 if giants[0]["x"] < 9 else 14
        action = view.first_action(("MiniPekka", "Minions", "Knight", "Archer"), bridge_y - 1, x)
        if action is not None:
            return action
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
            cards = ("Giant",) + MELEE if giant else MELEE
            action = view.first_action(cards, 20, 8 if left else 10)
            if action is not None:
                return action
    if giant:
        action = view.action("Giant", 1, 9)           # slow Giant push from behind the king
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
    """`human_style_strategy` with its thresholds, tiles and style flags redrawn
    every game, so the learner has to beat the style rather than one set of numbers."""

    __name__ = "randomized_human"

    def __init__(self):
        self.parameters = None

    def reset(self):
        rng = random.Random(random.getrandbits(64))   # like DiverseOpponent: seeded by the episode
        self.parameters = dict(low=rng.choice((3, 4, 5)), push=rng.choice((8, 9, 10)),
                               counter=rng.choice((7, 8, 9)), shift=rng.choice((-1, 0, 1)),
                               depth=rng.choice((0, 1)), bridge_y=rng.choice((13, 14)),
                               stack=rng.choice((1.0, 1.0, 1.5)), giant=rng.random() < 0.5,
                               ignore_ranged=rng.random() < 0.3, switch=rng.choice((6, 7, 8, 99)))

    def __call__(self, observation):
        if self.parameters is None:
            self.reset()
        return human_style_strategy(observation, **self.parameters)


ANTI_AIR = ("Archer", "Musketeer", "Minions", "Arrows", "Fireball")


class CardCountingOpponent:
    """The human style plus card counting: punish the learner from surplus.

    Reads the learner's hand and elixir from the simulator, as a human counting
    cards would. With no attacker on its own side:
    - backs up its own troops that have crossed (survivors of a won defence)
      when the learner is weak, as the user does;
    - starts a fresh push in the lane away from the learner's troops only when
      the learner cannot afford any card within ~2 s;
    and only ever attacks if it keeps `reserve` elixir to answer the next push
    (`type_reserve` when punishing a card-type mismatch rather than low elixir).
    The attacker is what the learner's coming cards answer worst: Minions when
    none can hit air, melee when only Archer or Musketeer can answer.
    `dead` lists cards the learner never plays (true of every checkpoint so
    far); a counting human would discount them. Call `bind_env` before use.
    """

    __name__ = "card_counting"

    def __init__(self, dead=("Giant", "Fireball", "Arrows"), reserve=3, type_reserve=5):
        self.dead = set(dead)
        self.reserve, self.type_reserve = reserve, type_reserve
        self.style = HumanStyleOpponent()
        self.env = None

    def bind_env(self, env):
        self.env = env

    def reset(self):
        self.style.reset()

    def _attack(self, view, x):
        learner = self.env.battle.players[0]
        soon = learner.elixir + 2 / (1.4 if self.env.battle.time >= 120 else 2.8)
        coming = [c for c in learner.cycle[:4] if c not in self.dead and ELIXIR_COST[c] <= soon]
        if not coming:
            cards, reserve = ("MiniPekka", "Minions", "Knight"), self.reserve
        elif not any(c in ANTI_AIR for c in coming):
            cards, reserve = ("Minions",), self.type_reserve
        elif set(coming) <= {"Archer", "Musketeer"}:
            cards, reserve = ("MiniPekka", "Knight"), self.type_reserve
        else:
            return None
        for card in cards:
            if view.elixir - ELIXIR_COST[card] >= reserve:
                action = view.action(card, self.style.parameters["bridge_y"], x)
                if action is not None:
                    return action
        return None

    def __call__(self, observation):
        if self.style.parameters is None:
            self.style.reset()
        view = BattleView.from_observation(observation)
        if self.env is not None and not any(e["y"] <= 17 for e in view.enemy_troops):
            survivors = [e for e in view.own_troops if 14 <= e["y"] <= 22]
            if survivors:                             # back up a won defence
                action = self._attack(view, 4 if np.mean([e["x"] for e in survivors]) < 9 else 14)
            else:                                     # fresh push only on low elixir
                learner = self.env.battle.players[0]
                soon = learner.elixir + 2 / (1.4 if self.env.battle.time >= 120 else 2.8)
                broke = not [c for c in learner.cycle[:4] if c not in self.dead and ELIXIR_COST[c] <= soon]
                theirs = [e["x"] for e in view.enemy_troops]
                x = (14 if np.mean(theirs) < 9 else 4) if theirs else view.lane_x
                action = self._attack(view, x) if broke else None
            if action is not None:
                return action
        return self.style(observation)


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
