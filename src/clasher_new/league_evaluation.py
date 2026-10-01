"""Fixed-budget, reproducible league comparisons, balanced across player sides."""
from collections import Counter
from contextlib import contextmanager
from math import comb
from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3 import PPO

from environment import CREnv, CardSaving, entity_names
from strategies import DiverseOpponent
from train_core import ReflectedOpponent


@contextmanager
def preserve_random_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def checkpoint_spec(path):
    return {"kind": "checkpoint", "path": str(Path(path).resolve())}


def make_schedule(games=50, seed=0, history=(), exploiters=(), references=(), target=None):
    """Return exactly `games` independent starts, half on each player side.

    Each side has the same opponent allocation but independent random seeds.
    Candidate and champion reuse the complete schedule, including sampling seeds.
    Available group weights are scripts/history/exploiters = 2/2/1; absent
    groups redistribute their budget proportionally. No extra diagnostic games.
    """
    if games < 6 or games % 2:
        raise ValueError("games must be even and at least 6")
    if target:
        pools = {"target": [checkpoint_spec(target)]}
        weights = {"target": 1}
    else:
        pools = {"scripts": [
            {"kind": "script", "style": style}
            for style in ("opposite_lane", "spell_control", "counterpush_mirrored")
        ]}
        paths = list(dict.fromkeys(str(Path(p).resolve()) for p in (*references, *history)))
        if paths:
            pools["history"] = [checkpoint_spec(p) for p in paths]
        if exploiters:
            pools["exploiters"] = [checkpoint_spec(p) for p in dict.fromkeys(exploiters)]
        weights = {name: {"scripts": 2, "history": 2, "exploiters": 1}[name]
                   for name in pools}
    pairs = games // 2
    quotas = {name: pairs * weight / sum(weights.values()) for name, weight in weights.items()}
    counts = {name: int(quota) for name, quota in quotas.items()}
    order = sorted(pools, key=lambda name: quotas[name] - counts[name], reverse=True)
    for name in order[:pairs - sum(counts.values())]:
        counts[name] += 1
    rng = random.Random(seed)
    schedule = []
    for group, pool in pools.items():
        offset = seed % len(pool)
        for index in range(counts[group]):
            opponent = pool[(offset + index) % len(pool)]
            for side in (0, 1):
                schedule.append({
                    "match_id": len(schedule), "group": group, "opponent": opponent,
                    "side": side, "seed": rng.randrange(2**32),
                    "policy_seed": rng.randrange(2**32),
                    "opponent_seed": rng.randrange(2**32),
                })
    return schedule


class EvaluationActor:
    """An actor with its own sampling stream, independent of the other player."""

    def __init__(self, env, side, seed, model=None, script=None):
        self.env, self.side, self.seed = env, side, seed
        self.model, self.script = model, script
        self.reset()

    def reset(self):
        self.torch_state = torch.Generator(device="cpu").manual_seed(self.seed).get_state()
        self.python_state = random.Random(self.seed).getstate()
        self.numpy_state = np.random.RandomState(self.seed).get_state()
        self.decisions = self.deployments = self.valid = 0
        self.saving = CardSaving()
        self.saves = 0
        self.available, self.selected = Counter(), Counter()
        self.last_action = (0, 0, 0)
        if self.script is not None:
            with self.random_stream():
                self.script.reset()

    @contextmanager
    def random_stream(self):
        with preserve_random_state():
            torch.set_rng_state(self.torch_state)
            random.setstate(self.python_state)
            np.random.set_state(self.numpy_state)
            try:
                yield
            finally:
                self.torch_state = torch.get_rng_state()
                self.python_state = random.getstate()
                self.numpy_state = np.random.get_state()

    def __call__(self, observation):
        mask = self.env.legal_mask(self.side)
        self.last_action = (0, 0, 0)
        if self.saving.waiting(self.env.battle.players[self.side]) or not mask.any():
            return self.last_action
        self.decisions += 1
        for slot in range(4):
            if mask[slot].any():
                self.available[entity_names[int(observation["hand"][slot])]] += 1
        with self.random_stream():
            if self.model is not None:
                action, _ = self.model.predict(dict(observation, legal_mask=mask), deterministic=False)
            else:
                action = self.script(observation)
        self.last_action = tuple(map(int, action))
        if self.model is not None and getattr(self.model.policy, "allow_saving", False):
            self.last_action = self.saving.resolve(
                self.env.battle.players[self.side], self.last_action)
            self.saves += int(self.saving.card is not None)
        slot, _, _ = self.last_action
        if slot:
            self.deployments += 1
            self.selected[entity_names[int(observation["hand"][slot - 1])]] += 1
        return self.last_action


def _summary(rows):
    games = len(rows)
    wins = sum(row["win"] for row in rows)
    deployments = sum(row["deployments"] for row in rows)
    valid = sum(row["valid_deployments"] for row in rows)
    result = {"games": games, "wins": wins, "losses": games - wins,
              "win_rate": wins / games if games else 0.0,
              "deployments": deployments, "valid_deployments": valid,
              "deployment_success_rate": valid / deployments if deployments else 0.0,
              "decisions": sum(row["decisions"] for row in rows),
              "saving_actions": sum(row.get("saving_actions", 0) for row in rows)}
    for key in ("cards_available", "cards_selected"):
        counts = Counter()
        for row in rows:
            counts.update(row[key])
        result[key] = dict(counts)
    return result


def evaluate_schedule(checkpoint, schedule):
    """Evaluate exactly the supplied starts, without supplementary games."""
    if not schedule:
        raise ValueError("evaluation schedule is empty")
    with preserve_random_state():
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return _evaluate_schedule(checkpoint, schedule)
        finally:
            torch.set_num_threads(previous_threads)


def _evaluate_schedule(checkpoint, schedule):
    path = str(Path(checkpoint).resolve())
    paths = {path} | {match["opponent"]["path"] for match in schedule
                      if match["opponent"]["kind"] == "checkpoint"}
    models = {name: PPO.load(name, device="cpu") for name in sorted(paths)}
    rows = []
    for match in schedule:
        env = CREnv()
        spec, side = match["opponent"], match["side"]
        candidate = EvaluationActor(env, side, match["policy_seed"], model=models[path])
        if spec["kind"] == "checkpoint":
            opponent = EvaluationActor(env, 1 - side, match["opponent_seed"], model=models[spec["path"]])
        else:
            script = (ReflectedOpponent(DiverseOpponent("counterpush"))
                      if spec["style"] == "counterpush_mirrored"
                      else DiverseOpponent(spec["style"]))
            opponent = EvaluationActor(env, 1 - side, match["opponent_seed"], script=script)
        actors = [candidate, opponent] if side == 0 else [opponent, candidate]
        env.opponent = actors[1]
        try:
            observation, _ = env.reset(seed=match["seed"])
            starting_decks = [list(p.cycle) for p in env.battle.players]
            while True:
                before = tuple(env.battle.players[side].cycle)
                action = actors[0](observation)
                # Both sides receive one action opportunity per half-second.
                # Actors enforce saving commitments and skip forced waits.
                observation, _, terminated, truncated, _ = env._step_once(action)
                if candidate.last_action[0]:
                    candidate.valid += before != tuple(env.battle.players[side].cycle)
                if terminated or truncated:
                    break
            if env.battle.winner not in (0, 1):
                raise RuntimeError(f"invalid battle winner: {env.battle.winner!r}")
            rows.append({**match, "starting_decks": starting_decks,
                         "win": int(env.battle.winner == side),
                         "decisions": candidate.decisions,
                         "deployments": candidate.deployments,
                         "valid_deployments": candidate.valid,
                         "saving_actions": candidate.saves,
                         "cards_available": dict(candidate.available),
                         "cards_selected": dict(candidate.selected)})
        finally:
            env.close()
    return {"checkpoint": path, "deterministic": False, "matches": rows,
            "total": _summary(rows),
            "groups": {group: _summary([r for r in rows if r["group"] == group])
                       for group in sorted({r["group"] for r in rows})},
            "sides": {str(side): _summary([r for r in rows if r["side"] == side])
                      for side in (0, 1)}}


def binomial_tail(wins, games):
    """One-sided exact fair-coin tail, including the observed result."""
    return sum(comb(games, k) for k in range(wins, games + 1)) / 2**games


def paired_evidence(candidate, champion):
    """Exact McNemar test on independent matched game starts (not side pairs)."""
    if len(candidate) != len(champion) or not candidate:
        raise ValueError("candidate and champion must have the same nonempty schedule")
    keys = ("match_id", "group", "opponent", "side", "seed", "policy_seed", "opponent_seed")
    for left, right in zip(candidate, champion):
        if any(left[key] != right[key] for key in keys):
            raise ValueError("candidate and champion schedules differ")
    gains = sum(a["win"] > b["win"] for a, b in zip(candidate, champion))
    losses = sum(a["win"] < b["win"] for a, b in zip(candidate, champion))
    return {"games": len(candidate), "gains": gains, "losses": losses,
            "difference": (gains - losses) / len(candidate),
            "improvement_p": binomial_tail(gains, gains + losses),
            "regression_p": binomial_tail(losses, gains + losses)}


def promotion_decision(candidate, champion, margin=.02, rollback_margin=.10, alpha=.05):
    evidence = paired_evidence(candidate["matches"], champion["matches"])
    groups = {}
    names = sorted({row["group"] for row in candidate["matches"]})
    for name in names:
        groups[name] = paired_evidence(
            [r for r in candidate["matches"] if r["group"] == name],
            [r for r in champion["matches"] if r["group"] == name],
        )
    # Bonferroni correction covers the overall and all group regression tests.
    regression_alpha = alpha / (len(groups) + 1)
    regressions = [name for name, result in {"overall": evidence, **groups}.items()
                   if result["difference"] <= -rollback_margin
                   and result["regression_p"] <= regression_alpha]
    valid = candidate["total"]["deployment_success_rate"] >= .99
    if not valid:
        status, reason = "rollback", "invalid_deployments_or_no_deployments"
    elif regressions:
        status, reason = "rollback", "clear_regression"
    elif evidence["difference"] >= margin and evidence["improvement_p"] <= alpha:
        status, reason = "promoted", "paired_improvement"
    else:
        status, reason = "inconclusive", "insufficient_evidence_of_improvement"
    return {"status": status, "reason": reason, "accepted": status == "promoted",
            "overall": evidence, "groups": groups, "regressed_groups": regressions,
            "minimum_gain": margin, "rollback_margin": rollback_margin,
            "alpha": alpha, "regression_alpha": regression_alpha,
            "test": "one_sided_exact_mcnemar"}


def exploiter_decision(report, threshold=.55, alpha=.05):
    total = report["total"]
    p = binomial_tail(total["wins"], total["games"])
    accepted = (total["deployment_success_rate"] >= .99
                and total["win_rate"] >= threshold and p <= alpha)
    return {"accepted": accepted, "win_rate": total["win_rate"],
            "advantage_p": p, "alpha": alpha, "threshold": threshold,
            "games": total["games"], "test": "one_sided_exact_binomial"}
