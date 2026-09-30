"""Small, explicit training core for the masked spatial agent.

This file owns the mechanics shared by every run:
- seeded spawned environments;
- learner-side reflection and exact legality masks;
- frozen scripted/checkpoint opponents;
- one corrected parallel PPO training cycle; and
- compact candidate evaluation.

The outer improvement loop lives in train_league.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from environment import CREnv, CardSaving, entity_names
from masked_spatial import LegalPlacement, MaskedSpatialPolicy
from parallel_rollout import ParallelPPO, ParallelVecEnv, VariableDiscountBuffer
from strategies import (
    DiverseOpponent,
    STRATEGIES,
    counterpush_strategy,
)
from card_utils import Card
from core import Position


DEFAULT_GAMMA = 0.997
DEFAULT_GAE_LAMBDA = 0.995
COUNTER_FRACTION = 0.20
# Bump this when the return or entropy objective changes, even if gamma does not.
TRAINING_SEMANTICS = "elapsed_half_seconds_balanced_entropy_potential_v2"
PPO_SETTINGS = {
    "workers": "n_envs", "rollout_steps": "n_steps", "batch_size": "batch_size",
    "epochs": "n_epochs", "learning_rate": "learning_rate", "target_kl": "target_kl",
    "entropy": "ent_coef", "gamma": "gamma", "gae_lambda": "gae_lambda",
}


def training_semantics(config):
    return TRAINING_SEMANTICS + ("_card_saving_v1" if config.allow_saving else "")


class EpisodeReflection(gym.Wrapper):
    """Reflect only the learner's view and actions for a whole episode."""

    def __init__(self, env, seed=0):
        super().__init__(env)
        self.rng = random.Random(seed)
        self.reflected = False

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.rng.seed(seed)
        self.reflected = self.rng.random() < 0.5
        return self.observation(obs), info

    def observation(self, obs):
        if not self.reflected:
            return obs
        grid = obs["grid"].copy()
        king_id = entity_names.index("KingTower")
        kings = grid[..., 0] == king_id
        grid[kings] = 0
        grid = grid[:, :, ::-1, :].copy()
        grid[kings] = obs["grid"][kings]
        return dict(obs, grid=grid)

    def step(self, action):
        slot, y, x = map(int, action)
        if self.reflected and slot:
            x = 17 - x
        obs, reward, terminated, truncated, info = self.env.step((slot, y, x))
        return self.observation(obs), reward, terminated, truncated, info


def _opponent_mask(env):
    """Build the exact player-1 mask expected by a frozen learner."""
    battle = env.battle
    player = battle.players[1]
    mask = np.zeros((4, 32, 18), dtype=np.int8)
    playable = [player.can_play_card(card) for card in player.cycle[:4]]
    if not any(playable):
        return mask
    tiles = np.array([
        [battle.can_place_troop(1, Position(17.5 - x, 31.5 - y))
         for x in range(18)] for y in range(32)
    ], dtype=np.int8)
    for slot, card in enumerate(player.cycle[:4]):
        if playable[slot]:
            mask[slot] = 1 if Card(card).type == "spell" else tiles
    return mask


class FrozenPolicy:
    """Use one checkpoint as a player-1 opponent."""

    def __init__(self, checkpoint, deterministic=False):
        self.path = str(Path(checkpoint))
        self.model = PPO.load(self.path, device="cpu")
        self.deterministic = deterministic
        self.env = None
        self.saving = CardSaving()
        self.__name__ = "frozen_" + Path(checkpoint).stem

    def bind_env(self, env):
        self.env = env

    def reset(self):
        self.saving = CardSaving()

    def __call__(self, observation):
        if self.env is None:
            raise RuntimeError("FrozenPolicy must be bound to its battle environment")
        player = self.env.battle.players[1]
        mask = _opponent_mask(self.env)
        allow_saving = getattr(self.model.policy, "allow_saving", False)
        if allow_saving and (self.saving.waiting(player) or not mask.any()):
            return (0, 0, 0)
        action, _ = self.model.predict(
            dict(observation, legal_mask=mask),
            deterministic=self.deterministic,
        )
        action = tuple(map(int, np.asarray(action).reshape(-1)[:3]))
        if allow_saving:
            action = self.saving.resolve(player, action)
        return action


class ReflectedOpponent:
    """Mirror one opponent's observation and deployment across the center line."""

    def __init__(self, inner):
        self.inner = inner
        self.__name__ = getattr(inner, "__name__", "opponent") + "_mirrored"

    def reset(self):
        if callable(getattr(self.inner, "reset", None)):
            self.inner.reset()

    def __call__(self, observation):
        grid = observation["grid"].copy()
        king_id = entity_names.index("KingTower")
        kings = grid[..., 0] == king_id
        grid[kings] = 0
        grid = grid[:, :, ::-1, :].copy()
        grid[kings] = observation["grid"][kings]
        slot, y, x = self.inner(dict(observation, grid=grid))
        return (int(slot), int(y), 17 - int(x)) if slot else (0, 0, 0)


@dataclass
class OpponentFactory:
    """Pick one frozen opponent per episode in every worker process."""

    history: tuple[str, ...] = ()
    exploiters: tuple[str, ...] = ()
    script_fraction: float = 0.60
    counter_fraction: float = 0.20
    forced: str | None = None

    def __call__(self):
        return LeagueOpponent(
            history=self.history,
            exploiters=self.exploiters,
            script_fraction=self.script_fraction,
            counter_fraction=self.counter_fraction,
            forced=self.forced,
        )


class LeagueOpponent:
    """Scripts plus frozen policies, with selection fixed for each episode."""

    def __init__(self, history=(), exploiters=(), script_fraction=0.60,
                 counter_fraction=0.20, forced=None):
        self.history = tuple(map(str, history))
        self.exploiters = tuple(map(str, exploiters))
        self.script_fraction = float(script_fraction)
        self.counter_fraction = float(counter_fraction)
        self.forced = str(forced) if forced else None
        self.scripts = training_opponents()
        self.cache = {}
        self.opponent = None
        self.env = None

    def bind_env(self, env):
        self.env = env
        if hasattr(self.opponent, "bind_env"):
            self.opponent.bind_env(env)

    def _policy(self, path):
        if path not in self.cache:
            self.cache[path] = FrozenPolicy(path)
        return self.cache[path]

    def reset(self):
        if self.forced:
            self.opponent = self._policy(self.forced)
        else:
            value = random.random()
            if self.history and value >= self.script_fraction:
                paths = self.exploiters if self.exploiters and random.random() < 0.25 else self.history
                self.opponent = self._policy(random.choice(paths))
            elif random.random() < self.counter_fraction:
                self.opponent = ReflectedOpponent(DiverseOpponent("counterpush"))
            else:
                self.opponent = random.choice(self.scripts)
        if callable(getattr(self.opponent, "reset", None)):
            self.opponent.reset()
        if hasattr(self.opponent, "bind_env"):
            self.opponent.bind_env(self.env)

    def __call__(self, observation):
        if self.opponent is None:
            self.reset()
        return self.opponent(observation)


def make_env(rank, seed, opponent_factory, reflect=True, discount_gamma=DEFAULT_GAMMA,
             allow_saving=False):
    """Return a picklable worker factory for ParallelVecEnv."""
    def factory():
        env_seed = 10_000 + seed * 1_000 + rank
        random.seed(env_seed)
        np.random.seed(env_seed)
        torch.set_num_threads(1)
        torch.manual_seed(env_seed)
        opponent = opponent_factory()
        env = CREnv(opponent_model=opponent, discount_gamma=discount_gamma,
                    allow_saving=allow_saving)
        if callable(getattr(opponent, "bind_env", None)):
            opponent.bind_env(env)
        if reflect:
            env = EpisodeReflection(env, seed=env_seed)
        return LegalPlacement(env)
    return factory


class CounterMixture(LeagueOpponent):
    """Compatibility name for the old scripted/counter opponent mixture."""

    def __init__(self):
        super().__init__(script_fraction=1.0, counter_fraction=0.20)
        self.counter = DiverseOpponent("counterpush")

    def reset(self):
        self.reflected = random.random() < self.counter_fraction
        self.opponent = self.counter if self.reflected else random.choice(self.scripts)
        if callable(getattr(self.opponent, "reset", None)):
            self.opponent.reset()
        if hasattr(self.opponent, "bind_env"):
            self.opponent.bind_env(self.env)

    def __call__(self, observation):
        if not getattr(self, "reflected", False):
            return super().__call__(observation)
        grid = observation["grid"].copy()
        king_id = entity_names.index("KingTower")
        kings = grid[..., 0] == king_id
        grid[kings] = 0
        grid = grid[:, :, ::-1, :].copy()
        grid[kings] = observation["grid"][kings]
        slot, y, x = self.opponent(dict(observation, grid=grid))
        return (slot, y, 17 - x) if slot else (0, 0, 0)


@dataclass
class TrainConfig:
    workers: int = 16
    rollout_steps: int = 512
    batch_size: int = 256
    epochs: int = 4
    learning_rate: float = 1e-4
    target_kl: float = 0.03
    entropy: float = 0.005
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    device: str = "auto"
    allow_saving: bool = False

    def validate(self):
        if self.workers < 1 or self.rollout_steps < 1 or self.batch_size < 1:
            raise ValueError("workers, rollout_steps, and batch_size must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in (0, 1]")


def effective_settings(model):
    return {key: getattr(model, attribute) for key, attribute in PPO_SETTINGS.items()}


def check_continuation(model, config):
    differences = [f"{key}: checkpoint={value!r}, requested={getattr(config, key)!r}"
                   for key, value in effective_settings(model).items()
                   if value != getattr(config, key)]
    if getattr(model, "training_semantics", None) != training_semantics(config):
        differences.append("checkpoint has an older or unknown training objective")
    if getattr(model.policy, "allow_saving", False) != config.allow_saving:
        differences.append("checkpoint saving-action setting differs")
    if not isinstance(model.policy, MaskedSpatialPolicy):
        differences.append("checkpoint does not use MaskedSpatialPolicy")
    if differences:
        raise ValueError("Cannot continue this checkpoint: " + "; ".join(differences)
                         + ". Use --initial-checkpoint in a new run to transfer its weights.")


def train_cycle(output, steps, seed, opponent_factory, config=None, initial=None,
                initialize=False):
    """Continue optimizer state, or explicitly initialize a fresh PPO from weights."""
    config = config or TrainConfig()
    config.validate()
    if initialize and not initial:
        raise ValueError("initialize requires an initial checkpoint")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    source = None
    if initial:
        # Validate continuation before starting processes or creating run output.
        source = ParallelPPO.load(initial, device=config.device)
        if not initialize:
            check_continuation(source, config)
    source_steps = int(source.num_timesteps) if source else None
    reset_value_head = bool(initialize and (
        source.gamma != config.gamma
        or getattr(source, "training_semantics", None) != training_semantics(config)
    ))
    output.mkdir(parents=True, exist_ok=False)
    env = ParallelVecEnv(
        [make_env(rank, seed, opponent_factory, discount_gamma=config.gamma,
                  allow_saving=config.allow_saving)
         for rank in range(config.workers)],
        monitor=output / "monitor.csv",
    )
    try:
        if source is not None and not initialize:
            model = source
            model.set_env(env)
            model.tensorboard_log = str(output / "tensorboard")
            model.set_random_seed(seed)
            reset_num_timesteps = False
        else:
            model = ParallelPPO(
                MaskedSpatialPolicy,
                env,
                n_steps=config.rollout_steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                n_epochs=config.epochs,
                target_kl=config.target_kl,
                ent_coef=config.entropy,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                rollout_buffer_class=VariableDiscountBuffer,
                device=config.device,
                seed=seed,
                policy_kwargs={**(source.policy_kwargs if source is not None else {}),
                               "allow_saving": config.allow_saving},
                verbose=1,
                tensorboard_log=str(output / "tensorboard"),
            )
            if source is not None:
                weights = source.policy.state_dict().copy()
                if reset_value_head:
                    for key, value in model.policy.state_dict().items():
                        if key.startswith("value_net."):
                            weights[key] = value
                model.policy.load_state_dict(weights, strict=True)
            reset_num_timesteps = True
        model.training_semantics = training_semantics(config)
        check_continuation(model, config)
        buffer = model.rollout_buffer
        if (not isinstance(buffer, VariableDiscountBuffer)
                or (buffer.buffer_size, buffer.n_envs, buffer.gamma, buffer.gae_lambda)
                != (model.n_steps, model.n_envs, model.gamma, model.gae_lambda)):
            raise ValueError("PPO rollout buffer does not match the configured training settings")
        metadata = {
            "initial": str(Path(initial).resolve()) if initial else None,
            "mode": "initialize" if initialize else "continue" if initial else "fresh",
            "source_steps": source_steps,
            "reset_value_head": reset_value_head,
            "training_semantics": model.training_semantics,
            "allow_saving": config.allow_saving,
            "initial_steps": int(model.num_timesteps),
            "requested_steps": int(steps),
            "seed": seed,
            **effective_settings(model),
            "discount_unit_seconds": 0.5,
            "elapsed_time_discounting": True,
            "elapsed_time_gae_lambda": True,
            "policy": type(model.policy).__name__,
            "collector": "parallel_rollout",
        }
        (output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            model.learn(
                total_timesteps=steps,
                reset_num_timesteps=reset_num_timesteps,
                tb_log_name="train",
            )
        finally:
            model.save(output / "final.zip")
    finally:
        env.close()
    return output / "final.zip"


def training_opponents():
    """The scripts sampled during learner training."""
    return [
        STRATEGIES[name] for name in
        ("defensive", "bridge", "split", "counterpush", "punish")
    ] + [DiverseOpponent("deep_defense"), DiverseOpponent("counterpush")]


def evaluation_opponents():
    """Fresh opponents reserved for promotion decisions."""
    return [
        DiverseOpponent("opposite_lane"),
        DiverseOpponent("spell_control"),
        ReflectedOpponent(DiverseOpponent("counterpush")),
    ]


def evaluate_as_player_one(checkpoint, games=20, seed=0, deterministic=False):
    """Diagnostic for the same frozen policy acting through the opponent side."""
    opponent = FrozenPolicy(checkpoint, deterministic=deterministic)
    env = CREnv(opponent_model=opponent)
    opponent.bind_env(env)
    wins = 0
    try:
        for game in range(games):
            obs, _ = env.reset(seed=seed + game)
            done = False
            while not done:
                action = counterpush_strategy(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            winner = env.battle.winner
            if winner not in (0, 1):
                raise RuntimeError(f"game ended without a valid winner: {winner!r}")
            wins += int(winner == 1)
    finally:
        env.close()
    return {"wins": wins, "games": games, "win_rate": wins / games if games else 0.0}


def evaluate(checkpoint, games=20, seed=0, opponents=None, deterministic=False):
    """Evaluate a checkpoint and return machine-readable behavioral metrics."""
    model = PPO.load(checkpoint, device="cpu")
    opponents = list(opponents if opponents is not None else evaluation_opponents())
    results = {}
    total = {"wins": 0, "games": 0, "losses": 0, "decisions": 0, "deployments": 0,
             "valid_deployments": 0, "cards_available": {}, "cards_selected": {}}
    for opponent_index, opponent in enumerate(opponents):
        name = getattr(opponent, "__name__", type(opponent).__name__)
        base = CREnv(opponent_model=opponent,
                     allow_saving=getattr(model.policy, "allow_saving", False))
        if callable(getattr(opponent, "bind_env", None)):
            opponent.bind_env(base)
        env = LegalPlacement(EpisodeReflection(base, seed=seed))
        wins = decisions = deployments = valid = 0
        available = {}
        selected = {}
        try:
            for game in range(games):
                obs, _ = env.reset(seed=seed + game + opponent_index * 10_000)
                done = False
                while not done:
                    hand = np.asarray(obs["hand"])
                    legal = np.asarray(obs["legal_mask"])
                    for slot in range(4):
                        if legal[slot].any():
                            card = entity_names[int(hand[slot])]
                            available[card] = available.get(card, 0) + 1
                    action, _ = model.predict(obs, deterministic=deterministic)
                    slot, y, x = map(int, np.asarray(action).reshape(-1)[:3])
                    decisions += 1
                    saving = bool(slot and base.allow_saving and not legal[slot - 1].any())
                    if slot and not saving:
                        deployments += 1
                        card = entity_names[int(hand[slot - 1])]
                        selected[card] = selected.get(card, 0) + 1
                        before = tuple(env.unwrapped.battle.players[0].cycle)
                        obs, _, terminated, truncated, _ = env.step((slot, y, x))
                        after = tuple(env.unwrapped.battle.players[0].cycle)
                        valid += before != after
                    else:
                        obs, _, terminated, truncated, _ = env.step((slot, y, x))
                    done = terminated or truncated
                winner = env.unwrapped.battle.winner
                if winner not in (0, 1):
                    raise RuntimeError(f"game ended without a valid winner: {winner!r}")
                wins += int(winner == 0)
        finally:
            env.close()
        result = {
            "wins": wins, "games": games,
            "losses": games - wins,
            "win_rate": wins / games if games else 0.0,
            "decisions": decisions,
            "deployments": deployments,
            "valid_deployments": valid,
            "deployment_success_rate": valid / deployments if deployments else 0.0,
            "cards_available": available,
            "cards_selected": selected,
        }
        results[name] = result
        for key in ("wins", "games", "losses", "decisions", "deployments", "valid_deployments"):
            total[key] += result.get(key, 0)
        for key, values in (("cards_available", available), ("cards_selected", selected)):
            for card, count in values.items():
                total[key][card] = total[key].get(card, 0) + count
    total["losses"] = total["games"] - total["wins"]
    total["win_rate"] = total["wins"] / total["games"] if total["games"] else 0.0
    total["deployment_success_rate"] = (
        total["valid_deployments"] / total["deployments"]
        if total["deployments"] else 0.0
    )
    report = {"checkpoint": str(Path(checkpoint).resolve()),
              "deterministic": deterministic, "total": total,
              "opponents": results}
    report["player1_diagnostic"] = evaluate_as_player_one(
        checkpoint, games, seed, deterministic=deterministic
    )
    return report
