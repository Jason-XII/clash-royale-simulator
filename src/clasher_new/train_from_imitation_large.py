"""PPO fine-tuning with varied opponents and full-action rehearsal."""

from collections import OrderedDict
from pathlib import Path
import random
import re

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from environment import CREnv
from strategies import make_opponent_pool


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "cr_selfplay_rehearsal_dir"
START = ROOT / "cr_script_imitation_large_200k.zip"


class MixedOpponent:
    """50% scripts, 30% historical policies, 20% latest policy snapshot."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.scripts = make_opponent_pool()
        self.history = [
            ROOT / "cr_script_imitation_large_200k.zip",
            ROOT / "cr_selfplay_large_dir" / "cr_1000000_steps.zip",
        ]
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
            step = lambda path: int(re.search(r"cr_(\d+)_steps", path.name).group(1))
            recent = sorted(OUTPUT.glob("cr_*_steps.zip"), key=step)
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


class ActionRehearsal(BaseCallback):
    """Four full demonstrated-action updates before each PPO rollout."""

    def __init__(self, batches=4, batch_size=128):
        super().__init__()
        shards = sorted((ROOT / "script_imitation_data_200k").rglob("*.npz"))
        rng = np.random.default_rng(0)
        rng.shuffle(shards)
        self.shards = shards[max(1, round(0.1 * len(shards))):]
        self.batches = batches
        self.batch_size = batch_size

    def _on_rollout_start(self):
        losses = []
        for _ in range(self.batches):
            with np.load(random.choice(self.shards)) as data:
                indices = np.random.choice(len(data["action"]), self.batch_size)
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
            card_dist, hand = self.model.policy._card_distribution(latent, obs)
            slot = action[:, 0]
            loss = -card_dist.log_prob(slot).mean()
            play = slot != 0
            if play.any():
                selected = self.model.policy._selected_card_ids(hand, slot)
                placement = self.model.policy._placement_distribution(latent, selected)
                loss -= placement.log_prob(action[:, 1] * 18 + action[:, 2])[play].mean()
            optimizer = self.model.policy.optimizer
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())
        self.logger.record("train/action_rehearsal_loss", np.mean(losses))

    def _on_step(self):
        return True


if __name__ == "__main__":
    n_envs = 16
    OUTPUT.mkdir(exist_ok=True)
    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method="spawn"))
    model = PPO.load(
        START, env=env, device="auto",
        n_steps=8192 // n_envs, batch_size=256, learning_rate=1e-4,
        n_epochs=4, target_kl=0.03, ent_coef=0.01,
        tensorboard_log=str(OUTPUT),
    )
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(), lr=1e-4, **model.policy.optimizer_kwargs
    )
    callbacks = [
        ActionRehearsal(),
        CheckpointCallback(100_000 // n_envs, str(OUTPUT), name_prefix="cr"),
    ]
    try:
        model.learn(15_000_000, callback=callbacks)
    finally:
        model.save(ROOT / "cr_selfplay_large")
        env.close()
