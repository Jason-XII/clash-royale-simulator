"""PPO fine-tuning with varied opponents and light placement rehearsal."""

from collections import OrderedDict
from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from environment import CREnv
from strategies import make_opponent_pool


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "cr_selfplay_large_dir"


class MixedOpponent:
    """50% scripts, 30% historical policies, 20% latest policy snapshot."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.scripts = make_opponent_pool()
        self.history = [ROOT / "cr_script_imitation_large.zip"]
        self.history += sorted((ROOT / "cr_imitation_large").glob("*.zip"))
        self.models = OrderedDict()
        self.opponent = self.rng.choice(self.scripts)

    def _load(self, path):
        key = str(path)
        if key not in self.models:
            self.models[key] = PPO.load(key, device="cpu")
            if len(self.models) > 2:
                self.models.popitem(last=False)
        self.models.move_to_end(key)
        return self.models[key]

    def reset(self):
        draw = self.rng.random()
        if draw < 0.5:
            self.opponent = self.rng.choice(self.scripts)
        else:
            recent = sorted(OUTPUT.glob("cr_*_steps.zip"))
            choices = recent[-1:] if draw >= 0.8 and recent else self.history
            self.opponent = self._load(self.rng.choice(choices))
        reset = getattr(self.opponent, "reset", None)
        if callable(reset):
            reset()

    def __call__(self, observation):
        if isinstance(self.opponent, PPO):
            return self.opponent.predict(observation, deterministic=False)[0]
        return self.opponent(observation)


def make_env(rank):
    def factory():
        random.seed(10_000 + rank)
        np.random.seed(10_000 + rank)
        torch.set_num_threads(1)
        return CREnv(opponent_model=MixedOpponent(10_000 + rank))
    return factory


class PlacementRehearsal(BaseCallback):
    """One supervised placement update before each PPO rollout."""

    def __init__(self, weight=0.05, batch_size=128):
        super().__init__()
        shards = sorted((ROOT / "script_imitation_data_50k").glob("*.npz"))
        rng = np.random.default_rng(0)
        rng.shuffle(shards)
        self.shards = shards[20:]  # Keep the original validation split excluded.
        self.weight = weight
        self.batch_size = batch_size

    def _on_rollout_start(self):
        with np.load(random.choice(self.shards)) as data:
            valid = np.flatnonzero(data["action"][:, 0] != 0)
            indices = np.random.choice(valid, self.batch_size, replace=len(valid) < self.batch_size)
            obs = {
                "grid": data["grid"][indices], "hand": data["hand"][indices],
                "elixir": data["elixir"][indices], "phase": data["phase"][indices],
                "time_till_next_phase": data["time"][indices],
            }
            action = torch.as_tensor(
                data["action"][indices], device=self.model.device, dtype=torch.long
            )
        obs, _ = self.model.policy.obs_to_tensor(obs)
        latent, _ = self.model.policy._latents(obs)
        selected = self.model.policy._selected_card_ids(obs["hand"][:, :4].long(), action[:, 0])
        distribution = self.model.policy._placement_distribution(latent, selected)
        loss = -self.weight * distribution.log_prob(action[:, 1] * 18 + action[:, 2]).mean()
        optimizer = self.model.policy.optimizer
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
        optimizer.step()
        self.logger.record("train/placement_rehearsal_loss", loss.item())

    def _on_step(self):
        return True


if __name__ == "__main__":
    n_envs = 16
    OUTPUT.mkdir(exist_ok=True)
    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method="spawn"))
    model = PPO.load(
        ROOT / "cr_script_imitation_large.zip", env=env, device="auto",
        n_steps=8192 // n_envs, batch_size=256, learning_rate=1e-4,
        n_epochs=4, target_kl=0.03, ent_coef=0.01,
        tensorboard_log=str(OUTPUT),
    )
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(), lr=1e-4, **model.policy.optimizer_kwargs
    )
    callbacks = [
        PlacementRehearsal(),
        CheckpointCallback(100_000 // n_envs, str(OUTPUT), name_prefix="cr"),
    ]
    try:
        model.learn(15_000_000, callback=callbacks)
    finally:
        model.save(ROOT / "cr_selfplay_large")
        env.close()
