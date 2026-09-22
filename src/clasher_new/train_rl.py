"""Configured PPO training and explicit checkpoint continuation.

Run from any directory; relative CLI paths are relative to that directory.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor


ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "train_rl.py", "policy.py", "environment.py",
    "strategies.py", "defensive_strategy.py", "battle.py", "player.py",
    "arena.py", "core.py", "card_utils.py", "card_mechanics.py",
    "pathfinding.py", "pathfinding_heap.py",
)
DATA_FILES = (
    "gamedata.json", "cards_stats_characters.json", "cards_stats_spell.json",
    "cards_stats_building.json", "cards_stats_projectile.json",
)


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    n_envs: int = 16
    n_steps: int = 512
    batch_size: int = 256
    learning_rate: float = 1e-4
    n_epochs: int = 4
    gamma: float = 0.997
    gae_lambda: float = 0.98
    reward_mode: str = "potential"
    shaping_scale: float = 1.0
    ent_coef: float = 0.001
    entropy_samples: int = 256
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    script_fraction: float = 0.6
    opponent_spread: float = 0.25
    scatter_opponent: bool = False
    start_elixir_min: float = 5.0
    start_elixir_max: float = 5.0
    checkpoint_every: int = 100_000

    def validate(self):
        for key in ("n_envs", "n_steps", "batch_size", "n_epochs", "checkpoint_every", "entropy_samples"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        rollout_size = self.n_envs * self.n_steps
        if self.batch_size < 2 or rollout_size < 2 or rollout_size % self.batch_size:
            raise ValueError("rollout size must be divisible by batch_size, both >= 2")
        for key in ("gamma", "gae_lambda", "script_fraction", "opponent_spread"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be between 0 and 1")
        if not 0 <= self.start_elixir_min <= self.start_elixir_max <= 10:
            raise ValueError("start_elixir must satisfy 0 <= min <= max <= 10")
        for key in ("learning_rate", "clip_range", "max_grad_norm", "target_kl"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("ent_coef", "vf_coef", "shaping_scale"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not 0 <= self.seed < 2**32 - 10_000 - self.n_envs:
            raise ValueError("seed must fit the NumPy seed range, including worker offsets")
        if self.reward_mode not in ("legacy", "potential"):
            raise ValueError("reward_mode must be legacy or potential")


def file_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES + DATA_FILES}


def git_revision():
    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--short")}


def describe_space(space):
    if hasattr(space, "spaces"):
        return {key: describe_space(value) for key, value in space.spaces.items()}
    result = {"type": type(space).__name__, "shape": list(space.shape),
              "dtype": str(space.dtype)}
    if hasattr(space, "nvec"):
        result["nvec"] = space.nvec.tolist()
    elif hasattr(space, "n"):
        result["n"] = int(space.n)
    return result


def make_manifest(config, env, initialized_from=None):
    return {
        "schema_version": 1,
        "config": asdict(config),
        "architecture": {
            "policy": "policy.ClashPolicy",
            "extractor": "policy.SpatialEncoder",
            "features_dim": 256,
        },
        "observation_space": describe_space(env.observation_space),
        "action_space": describe_space(env.action_space),
        "entropy": {"objective": "joint_action_entropy", "units": "nats", "wait_actions": 1},
        "reward": {"mode": config.reward_mode, "outcome": {"win": 10, "loss": -10},
                   "potential": "0.5 * (own_hp_fraction - enemy_hp_fraction + crown_balance / 3)",
                   "shaping": "scale * (gamma * Phi(next) - Phi(current)); terminal Phi = 0",
                   "legacy": "5 * crown_delta + 0.001 * enemy_damage - 0.0012 * own_damage",
                   "decision_seconds": 0.5},
        "opponents": {"type": "HistoricalOpponent", "pool": "make_opponent_pool",
                      "directory": "checkpoints", "sampling": "uniform_history",
                      "script_fraction": config.script_fraction,
                      "spread": config.opponent_spread,
                      "scatter_opponent": config.scatter_opponent},
        "start_elixir": {"min": config.start_elixir_min, "max": config.start_elixir_max,
                         "sampling": "uniform_per_player_per_episode"},
        "files": file_hashes(),
        "git": git_revision(),
        "packages": {name: version(name) for name in
                     ("numpy", "torch", "gymnasium", "stable-baselines3")},
        "python": sys.version,
        "initialized_from": initialized_from,
    }


def make_env(rank, config, checkpoint_dir):
    def factory():
        from environment import CREnv
        from strategies import HistoricalOpponent

        worker_seed = config.seed + 10_000 + rank
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        torch.set_num_threads(1)
        return CREnv(opponent_model=HistoricalOpponent(
            worker_seed, checkpoint_dir, script_fraction=config.script_fraction,
            opponent_spread=config.opponent_spread,
            scatter_opponent=config.scatter_opponent,
        ), reward_mode=config.reward_mode, gamma=config.gamma, shaping_scale=config.shaping_scale,
           start_elixir=(config.start_elixir_min, config.start_elixir_max))
    return factory


def save_checkpoint(model, path):
    """Publish only complete zip files to self-play workers."""
    temporary = path.with_name(f".pending_{path.name}")
    model.save(temporary)
    temporary.replace(path)


class RunCheckpoints(BaseCallback):
    def __init__(self, directory, every):
        super().__init__()
        self.directory = directory
        self.every = every

    def _on_training_start(self):
        self.next_save = (self.num_timesteps // self.every + 1) * self.every

    def _on_step(self):
        if self.num_timesteps >= self.next_save:
            save_checkpoint(self.model, self.directory / f"cr_{self.num_timesteps}_steps.zip")
            self.next_save += self.every
        return True


class ExplorationMetrics(BaseCallback):
    """Record rollout rewards and bounded-sample policy diagnostics."""

    def __init__(self, samples=256):
        super().__init__()
        self.samples = samples

    def _on_rollout_start(self):
        self.reward_steps = self.shaping_abs = self.completed_games = self.wins = 0

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            self.reward_steps += 1
            self.shaping_abs += abs(info["reward_shaping"])
            if done:
                self.completed_games += 1
                self.wins += info["reward_outcome"] > 0
        return True

    def _on_rollout_end(self):
        from environment import entity_names

        self.logger.record("reward/mean_abs_shaping", self.shaping_abs / self.reward_steps)
        self.logger.record("reward/completed_games", self.completed_games)
        if self.completed_games:
            self.logger.record("reward/training_win_rate", self.wins / self.completed_games)
        buffer = self.model.rollout_buffer
        count = buffer.buffer_size * buffer.n_envs
        # Evenly spaced indices avoid changing the training RNG streams.
        indices = np.linspace(0, count - 1, min(count, self.samples), dtype=int)
        parts = []
        policy = self.model.policy
        previous_mode = policy.training
        policy.set_training_mode(False)
        try:
            for start in range(0, len(indices), 64):
                batch = indices[start:start + 64]
                times, envs = np.divmod(batch, buffer.n_envs)
                # Use buffer tensors directly, as PPO does. Discrete fields in
                # the buffer have an extra trailing dimension (e.g. phase).
                tensors = {key: torch.as_tensor(values[times, envs], device=policy.device)
                           for key, values in buffer.observations.items()}
                parts.append({key: value.cpu().numpy() for key, value in
                              policy.exploration_statistics(tensors).items()})
        finally:
            policy.set_training_mode(previous_mode)
        stats = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        actionable = stats["actionable"]
        self.logger.record("exploration/samples", len(indices))
        self.logger.record("exploration/actionable_samples", int(actionable.sum()))
        self.logger.record("exploration/forced_wait_fraction", float((~actionable).mean()))
        self.logger.record("exploration/sampled_wait_rate", float((buffer.actions[..., 0] == 0).mean()))
        for key in ("wait_probability", "slot_entropy", "wait_play_entropy", "joint_entropy"):
            self.logger.record(f"exploration/{key}", float(stats[key].mean()))
        if actionable.any():
            for key in ("wait_probability", "card_entropy_given_play", "placement_entropy_given_play"):
                self.logger.record(f"exploration/actionable_{key}", float(stats[key][actionable].mean()))
        for name in policy.CARD_COSTS:
            matches = stats["hand"] == entity_names.index(name)
            probability = (stats["slot_probabilities"][:, 1:] * matches).sum(axis=1)
            self.logger.record(f"exploration/card_probability/{name}", float(probability.mean()))
            eligible = matches & stats["playable"]
            self.logger.record(f"exploration/available_samples/{name}", int(eligible.sum()))
            if eligible.any():
                self.logger.record(f"exploration/placement_entropy/{name}",
                                   float(stats["placement_entropies"][eligible].mean()))


def build_model(config, env, device, log_dir, init_from=None):
    from policy import SpatialEncoder, ClashPolicy

    options = {key: value for key, value in asdict(config).items()
               if key not in ("n_envs", "script_fraction", "checkpoint_every", "entropy_samples",
                              "reward_mode", "shaping_scale",
                              "opponent_spread", "scatter_opponent", "start_elixir_min", "start_elixir_max")}
    options.update(device=device, tensorboard_log=str(log_dir), verbose=1,
                   policy_kwargs={"features_extractor_class": SpatialEncoder})
    if init_from is None:
        return PPO(ClashPolicy, env, **options)
    model = PPO.load(init_from, env=env, **options)
    if type(model.policy) is not ClashPolicy:
        raise ValueError("initializer must use ClashPolicy")
    # Initialization is a new experiment; resume below preserves Adam state.
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(), lr=config.learning_rate,
        **model.policy.optimizer_kwargs,
    )
    model._n_updates = 0
    return model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="new experiment directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path, help="checkpoint from this trainer")
    mode.add_argument("--init-from", type=Path, help="compatible policy; start a new run and optimizer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--total-timesteps", type=int, default=15_000_000,
                        help="additional steps; rounded up to full PPO rollouts")
    for key, default in asdict(RunConfig()).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(default, bool):
            # type=bool would turn "--flag False" into True; expose --flag/--no-flag.
            parser.add_argument(flag, action=argparse.BooleanOptionalAction,
                                default=None, help=f"new-run default: {default}")
        else:
            parser.add_argument(flag, type=type(default),
                                default=None, help=f"new-run default: {default}")
    args = parser.parse_args(argv)
    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive")
    if args.run_dir is None and args.resume is None:
        parser.error("provide --run-dir for a new experiment, or --resume")
    return args


def run(args):
    overrides = {key: getattr(args, key) for key in asdict(RunConfig())
                 if getattr(args, key) is not None}
    manifest = None
    if args.resume:
        run_dir = args.resume.parent.parent
        if args.run_dir and args.run_dir != run_dir:
            raise ValueError("--run-dir must match the checkpoint's original run")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest["schema_version"] != 1:
            raise ValueError("unsupported run manifest version")
        if manifest["files"] != file_hashes():
            raise ValueError("training code or game data changed; start a new experiment")
        for key, value in overrides.items():
            if value != manifest["config"][key]:
                raise ValueError(f"cannot change {key} on resume; start a new experiment")
        config = RunConfig(**manifest["config"])
    else:
        run_dir = args.run_dir
        config = RunConfig(**overrides)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError("run directory is not empty; use --resume or a new directory")
    config.validate()
    checkpoint_dir = run_dir / "checkpoints"
    factories = [make_env(rank, config, checkpoint_dir) for rank in range(config.n_envs)]
    env = VecMonitor(DummyVecEnv(factories) if config.n_envs == 1 else
                     SubprocVecEnv(factories, start_method="spawn"))
    try:
        if args.resume:
            model = PPO.load(args.resume, env=env, device=args.device,
                             tensorboard_log=str(run_dir / "logs"))
            if getattr(model, "run_metadata", None) != manifest:
                raise ValueError("checkpoint does not match the run manifest")
            newer = [path for path in checkpoint_dir.glob("cr_*_steps.zip")
                     if int(path.stem.split("_")[1]) > model.num_timesteps]
            if newer:
                raise ValueError("newer checkpoints exist; resume the latest checkpoint")
        else:
            model = build_model(config, env, args.device, run_dir / "logs", args.init_from)
            initialized_from = None if args.init_from is None else {
                "path": str(args.init_from),
                "sha256": hashlib.sha256(args.init_from.read_bytes()).hexdigest(),
            }
            manifest = make_manifest(config, env, initialized_from)
            model.run_metadata = manifest
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Run: {run_dir}\nConfiguration: {json.dumps(asdict(config), sort_keys=True)}")
        try:
            model.learn(
                total_timesteps=args.total_timesteps,
                reset_num_timesteps=args.resume is None,
                callback=[RunCheckpoints(checkpoint_dir, config.checkpoint_every),
                          ExplorationMetrics(config.entropy_samples)],
                tb_log_name="ppo",
            )
        finally:
            save_checkpoint(model, checkpoint_dir / f"cr_{model.num_timesteps}_steps.zip")
    finally:
        env.close()


def main(argv=None):
    args = parse_args(argv)
    for key in ("run_dir", "resume", "init_from"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    for key in ("resume", "init_from"):
        value = getattr(args, key)
        if value is not None and not value.is_file():
            raise ValueError(f"checkpoint does not exist: {value}")
    previous_cwd = Path.cwd()
    try:
        # Simulator data files currently resolve relative to the working directory.
        os.chdir(ROOT)
        run(args)
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
