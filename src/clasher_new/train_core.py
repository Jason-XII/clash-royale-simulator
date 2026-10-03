"""Training core for the masked spatial agent.

- `LearnerView`: what the learner sees (random left-right reflection + legal mask).
- Opponents: scripts, frozen checkpoints, and the per-episode league mixture.
- `train_cycle`: one parallel PPO run, fresh, continued, or initialized from weights.

The outer improvement loop lives in train_league.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO

from environment import Bank, BANK_TARGETS, CREnv, LIVE_PLAY_DELAY, entity_names, enemy_troops_in_half
from masked_spatial import MaskedSpatialPolicy
from parallel_rollout import ParallelPPO, ParallelVecEnv, VariableDiscountBuffer
from strategies import DiverseOpponent, HumanStyleOpponent, STRATEGIES


DEFAULT_GAMMA = 0.997
DEFAULT_GAE_LAMBDA = 0.995
# Bump this when the return or entropy objective changes, even if gamma does not.
# v3: both players decide from the same snapshot (the opponent used to see the
# learner's card before choosing). v4: bank-until-elixir actions. v5: cards land
# after a random play delay.
TRAINING_SEMANTICS = "elapsed_half_seconds_balanced_entropy_potential_v5_delay"
PPO_SETTINGS = {  # TrainConfig field -> PPO attribute
    "workers": "n_envs", "rollout_steps": "n_steps", "batch_size": "batch_size",
    "epochs": "n_epochs", "learning_rate": "learning_rate", "target_kl": "target_kl",
    "entropy": "ent_coef", "gamma": "gamma", "gae_lambda": "gae_lambda",
}


def mirror_grid(grid):
    """Flip the board left-right. King towers sit on the center line and stay put."""
    kings = grid[..., 0] == entity_names.index("KingTower")
    mirrored = grid.copy()
    mirrored[kings] = 0
    mirrored = mirrored[:, :, ::-1, :].copy()
    mirrored[kings] = grid[kings]
    return mirrored


class LearnerView(gym.Wrapper):
    """Player 0's view: mirrored for a random half of episodes, plus `legal_mask`."""

    def __init__(self, env, seed=0):
        super().__init__(env)
        self.rng = random.Random(seed)
        self.reflected = False
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "legal_mask": spaces.MultiBinary((4, 32, 18)),
        })

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.rng.seed(seed)
        self.reflected = self.rng.random() < 0.5
        return self.observation(obs), info

    def step(self, action):
        slot, y, x = map(int, action)
        if self.reflected and slot:
            x = 17 - x
        obs, reward, terminated, truncated, info = self.env.step((slot, y, x))
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        mask = self.env.unwrapped.legal_mask(0)
        if self.reflected:
            obs = dict(obs, grid=mirror_grid(obs["grid"]))
            mask = mask[:, :, ::-1].copy()
        return dict(obs, legal_mask=mask)


# --- opponents (player 1) ----------------------------------------------------

class FrozenPolicy:
    """A saved checkpoint playing as player 1. Call `bind_env` before use."""

    def __init__(self, checkpoint, deterministic=False):
        self.path = str(Path(checkpoint))
        self.model = PPO.load(self.path, device="cpu")
        self.deterministic = deterministic
        self.env = None
        self.bank = Bank()
        self.__name__ = "frozen_" + Path(checkpoint).stem

    def bind_env(self, env):
        self.env = env

    def reset(self):
        self.bank = Bank()

    def __call__(self, observation):
        if self.env is None:
            raise RuntimeError("FrozenPolicy must be bound to its battle environment")
        mask = self.env.legal_mask(1)
        intruders = lambda: enemy_troops_in_half(self.env.battle, 1)
        # Like the learner: no decision while banking or while no card is playable.
        if self.bank.active(self.env.battle.players[1].elixir, intruders) or not mask.any():
            return (0, 0, 0)
        action, _ = self.model.predict(dict(observation, legal_mask=mask),
                                       deterministic=self.deterministic)
        action = tuple(map(int, np.asarray(action).reshape(-1)[:3]))
        return self.bank.resolve(action, intruders)


class ReflectedOpponent:
    """Run an opponent on the mirrored board and mirror its deployments back."""

    def __init__(self, inner):
        self.inner = inner
        self.__name__ = getattr(inner, "__name__", "opponent") + "_mirrored"

    def reset(self):
        if callable(getattr(self.inner, "reset", None)):
            self.inner.reset()

    def __call__(self, observation):
        slot, y, x = self.inner(dict(observation, grid=mirror_grid(observation["grid"])))
        return (int(slot), int(y), 17 - int(x)) if slot else (0, 0, 0)


def training_opponents():
    """The scripts sampled during learner training."""
    return [STRATEGIES[name] for name in ("defensive", "bridge", "split", "counterpush", "punish")] \
        + [DiverseOpponent("deep_defense"), DiverseOpponent("counterpush"), HumanStyleOpponent()]


class LeagueOpponent:
    """Each episode, pick a script or a frozen checkpoint (history/exploiter).

    With history: `script_fraction` of episodes use scripts, the rest checkpoints
    (exploiters 25% of those). Of script episodes, `counter_fraction` use the
    mirrored counterpush script. `forced` always uses that checkpoint.

    Within scripts and within checkpoints, opponents are picked by difficulty
    (PFSP): weight (1 - win rate)^2, from this worker's recent games against each.
    """

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
        self.key = None
        self.win_rate = {}
        self.env = None

    def bind_env(self, env):
        self.env = env
        if hasattr(self.opponent, "bind_env"):
            self.opponent.bind_env(env)

    def _policy(self, path):
        if path not in self.cache:
            self.cache[path] = FrozenPolicy(path)
        return self.cache[path]

    def _record_result(self):
        """Fold the finished game into the learner's win rate against its opponent."""
        battle = getattr(self.env, "battle", None)
        if self.key is None or battle is None or not battle.game_over:
            return
        result = 1.0 if battle.winner == 0 else 0.5 if battle.winner is None else 0.0
        rate = self.win_rate.get(self.key, 0.5)
        self.win_rate[self.key] = rate + 0.1 * (result - rate)  # ponytail: EMA over ~10 games

    def _pick(self, keys):
        # ponytail: unseen opponents start at 50%; the floor keeps beaten ones in rotation.
        weights = [(1 - self.win_rate.get(key, 0.5)) ** 2 + 0.05 for key in keys]
        return random.choices(keys, weights)[0]

    def reset(self):
        self._record_result()
        # The first draw happens even without history, keeping seeded runs stable.
        draw = None if self.forced else random.random()
        if self.forced:
            self.key = self.forced
            self.opponent = self._policy(self.forced)
        elif self.history and draw >= self.script_fraction:
            paths = self.exploiters if self.exploiters and random.random() < 0.25 else self.history
            self.key = self._pick(paths)
            self.opponent = self._policy(self.key)
        elif random.random() < self.counter_fraction:
            self.key = "counterpush_mirrored"
            self.opponent = ReflectedOpponent(DiverseOpponent("counterpush"))
        else:
            self.key = self._pick(range(len(self.scripts)))
            self.opponent = self.scripts[self.key]
        if callable(getattr(self.opponent, "reset", None)):
            self.opponent.reset()
        if hasattr(self.opponent, "bind_env"):
            self.opponent.bind_env(self.env)

    def __call__(self, observation):
        if self.opponent is None:
            self.reset()
        return self.opponent(observation)


@dataclass
class OpponentFactory:
    """Picklable recipe: builds one `LeagueOpponent` inside each worker process."""

    history: tuple[str, ...] = ()
    exploiters: tuple[str, ...] = ()
    script_fraction: float = 0.60
    counter_fraction: float = 0.20
    forced: str | None = None

    def __call__(self):
        return LeagueOpponent(self.history, self.exploiters, self.script_fraction,
                              self.counter_fraction, self.forced)


def make_env(rank, seed, opponent_factory, discount_gamma=DEFAULT_GAMMA, play_delay=LIVE_PLAY_DELAY):
    """Return a picklable worker factory for ParallelVecEnv."""
    def factory():
        env_seed = 10_000 + seed * 1_000 + rank
        random.seed(env_seed)
        np.random.seed(env_seed)
        torch.set_num_threads(1)
        torch.manual_seed(env_seed)
        opponent = opponent_factory()
        env = CREnv(opponent_model=opponent, discount_gamma=discount_gamma, play_delay=play_delay)
        if callable(getattr(opponent, "bind_env", None)):
            opponent.bind_env(env)
        return LearnerView(env, seed=env_seed)
    return factory


# --- training ----------------------------------------------------------------

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
    play_delay: tuple[float, float] = LIVE_PLAY_DELAY

    def validate(self):
        if self.workers < 1 or self.rollout_steps < 1 or self.batch_size < 1:
            raise ValueError("workers, rollout_steps, and batch_size must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in (0, 1]")
        self.play_delay = list(self.play_delay)  # matches league.json after a JSON round-trip
        if not 0 <= self.play_delay[0] <= self.play_delay[1]:
            raise ValueError("play_delay must be (low, high) with 0 <= low <= high")


def effective_settings(model):
    return {key: getattr(model, attribute) for key, attribute in PPO_SETTINGS.items()}


def check_continuation(model, config):
    differences = [f"{key}: checkpoint={value!r}, requested={getattr(config, key)!r}"
                   for key, value in effective_settings(model).items()
                   if value != getattr(config, key)]
    if getattr(model, "training_semantics", None) != TRAINING_SEMANTICS:
        differences.append("checkpoint has an older or unknown training objective")
    if not isinstance(model.policy, MaskedSpatialPolicy):
        differences.append("checkpoint does not use MaskedSpatialPolicy")
    if differences:
        raise ValueError("Cannot continue this checkpoint: " + "; ".join(differences)
                         + ". Use --initial-checkpoint in a new run to transfer its weights.")


def train_cycle(output, steps, seed, opponent_factory, config=None, initial=None,
                initialize=False):
    """Train for `steps` and save `output/final.zip`.

    Without `initial`: fresh weights. With `initial`: continue its optimizer and
    step count (settings must match), or with `initialize=True` copy its weights
    into a fresh PPO using `config`. The value head is reset when the return
    objective changed.
    """
    config = config or TrainConfig()
    config.validate()
    if initialize and not initial:
        raise ValueError("initialize requires an initial checkpoint")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    source = ParallelPPO.load(initial, device=config.device) if initial else None
    if source is not None and not initialize:
        check_continuation(source, config)  # fail before starting worker processes
    reset_value_head = bool(initialize and (
        source.gamma != config.gamma
        or getattr(source, "training_semantics", None) != TRAINING_SEMANTICS))

    output.mkdir(parents=True, exist_ok=False)
    env = ParallelVecEnv([make_env(rank, seed, opponent_factory, config.gamma, tuple(config.play_delay))
                          for rank in range(config.workers)],
                         monitor=output / "monitor.csv")
    try:
        if source is not None and not initialize:
            model = source
            model.set_env(env)
            model.tensorboard_log = str(output / "tensorboard")
            model.set_random_seed(seed)
        else:
            model = ParallelPPO(
                MaskedSpatialPolicy, env,
                n_steps=config.rollout_steps, batch_size=config.batch_size,
                learning_rate=config.learning_rate, n_epochs=config.epochs,
                target_kl=config.target_kl, ent_coef=config.entropy,
                gamma=config.gamma, gae_lambda=config.gae_lambda,
                rollout_buffer_class=VariableDiscountBuffer, device=config.device, seed=seed,
                policy_kwargs=source.policy_kwargs if source is not None else None,
                verbose=1, tensorboard_log=str(output / "tensorboard"),
            )
            if source is not None:
                weights = source.policy.state_dict()
                if reset_value_head:
                    weights.update({key: value for key, value in model.policy.state_dict().items()
                                    if key.startswith("value_net.")})
                model.policy.load_state_dict(weights, strict=True)
        model.training_semantics = TRAINING_SEMANTICS
        check_continuation(model, config)
        metadata = {
            "initial": str(Path(initial).resolve()) if initial else None,
            "mode": "initialize" if initialize else "continue" if initial else "fresh",
            "source_steps": int(source.num_timesteps) if source else None,
            "reset_value_head": reset_value_head,
            "training_semantics": model.training_semantics,
            "bank_targets": list(BANK_TARGETS),
            "initial_steps": int(model.num_timesteps),
            "requested_steps": int(steps),
            "seed": seed,
            **effective_settings(model),
            "discount_unit_seconds": 0.5,
            "play_delay": list(config.play_delay),
            "policy": type(model.policy).__name__,
        }
        (output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            model.learn(total_timesteps=steps, tb_log_name="train",
                        reset_num_timesteps=source is None or initialize)
        finally:
            model.save(output / "final.zip")
    finally:
        env.close()
    return output / "final.zip"
