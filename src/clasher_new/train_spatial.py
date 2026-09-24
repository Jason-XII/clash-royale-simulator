"""Fresh spatial-head experiment and unchanged-head control.

From src/clasher_new: python train_spatial.py [--control] [--seed 0]
Both arms use reflection, the counter mixture, and identical PPO settings.
"""
import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import train_reflection
from train import CRFeatureExtractor
from train_autoregressive import ContentMaskedAutoregressivePolicy
from spatial_policy import SpatialPlacementPolicy
from train_counter import COUNTER_FRACTION


TOTAL_STEPS = 10_000_000
N_ENVS = 16
N_STEPS = 512
BATCH_SIZE = 256


def main(control=False, seed=0, run_dir=None):
    arm = "global" if control else "spatial"
    run_dir = Path(run_dir or f"cr_{arm}_scratch")
    run_dir.mkdir(parents=True, exist_ok=False)
    # Pass seed explicitly: spawned subprocesses re-import modules, so changing
    # a module global in the parent would not reliably seed their environments.
    env = VecMonitor(SubprocVecEnv(
        [make_env(rank, seed) for rank in range(N_ENVS)], start_method="spawn"
    ), filename=str(run_dir / "monitor.csv"))
    try:
        policy = ContentMaskedAutoregressivePolicy if control else SpatialPlacementPolicy
        policy_kwargs = {"features_extractor_class": CRFeatureExtractor} if control else {}
        model = PPO(
            policy, env, policy_kwargs=policy_kwargs, n_steps=N_STEPS,
            batch_size=BATCH_SIZE, learning_rate=1e-4, n_epochs=4,
            target_kl=0.03, ent_coef=0.005, gamma=0.99, gae_lambda=0.95,
            device="auto", seed=seed, verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
        )
        (run_dir / "experiment.json").write_text(json.dumps({
            "arm": arm, "initialization": "scratch", "seed": seed,
            "total_steps": TOTAL_STEPS, "n_envs": N_ENVS,
            "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
            "counter_fraction": COUNTER_FRACTION, "reflection_probability": 0.5,
            "parameters": sum(p.numel() for p in model.policy.parameters()),
        }, indent=2) + "\n")
        callback = CheckpointCallback(
            save_freq=max(100_000 // N_ENVS, 1),
            save_path=str(run_dir), name_prefix="cr",
        )
        try:
            model.learn(total_timesteps=TOTAL_STEPS, callback=callback, tb_log_name=arm)
        finally:
            model.save(run_dir / "final.zip")
    finally:
        env.close()


def make_env(rank, seed):
    def factory():
        import random
        import numpy as np
        import torch
        from environment import CREnv
        from train_counter import CounterMixture
        env_seed = 10_000 + seed * 1_000 + rank
        random.seed(env_seed)
        np.random.seed(env_seed)
        torch.set_num_threads(1)
        return train_reflection.EpisodeReflection(
            CREnv(opponent_model=CounterMixture()), seed=env_seed
        )
    return factory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="store_true", help="Use the original placement head")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(control=args.control, seed=args.seed)
