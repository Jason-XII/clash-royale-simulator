"""Fine-tune the behavior-cloned policy with a fresh PPO optimizer."""

from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from environment import CREnv
from strategies import make_opponent_pool
from train import CRFeatureExtractor
from train_autoregressive import ContentMaskedAutoregressivePolicy


IMITATION_CHECKPOINT = Path("cr_script_imitation_50k.zip")
MODEL_NAME = "cr_imitation_ppo"
TOTAL_TIMESTEPS = 15_000_000
N_ENVS = 16


def make_env(rank):
    def factory():
        random.seed(10_000 + rank)
        np.random.seed(10_000 + rank)
        torch.set_num_threads(1)
        return CREnv(opponent_pool=make_opponent_pool())

    return factory


def new_model(env, n_steps):
    if not IMITATION_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing imitation checkpoint: {IMITATION_CHECKPOINT}")

    # Loading the teacher checkpoint is only a convenient way to read its policy
    # parameters. Its supervised-learning optimizer is deliberately not copied.
    imitation = PPO.load(IMITATION_CHECKPOINT, device="cpu")
    model = PPO(
        ContentMaskedAutoregressivePolicy,
        env,
        policy_kwargs={"features_extractor_class": CRFeatureExtractor},
        n_steps=n_steps,
        batch_size=256,
        learning_rate=1e-4,
        n_epochs=4,
        target_kl=0.03,
        ent_coef=0.01,
        device="auto",
        seed=0,
        verbose=1,
        tensorboard_log=f"./{MODEL_NAME}_dir/",
    )
    model.policy.load_state_dict(imitation.policy.state_dict(), strict=True)

    source = imitation.policy.state_dict()
    copied = model.policy.state_dict()
    if not all(torch.equal(source[name].cpu(), copied[name].cpu()) for name in source):
        raise RuntimeError("Imitation policy parameters were not copied exactly.")

    # Save the untouched initialization so it can always be recovered or tested.
    model.save(f"{MODEL_NAME}_initial")
    print(f"Initialized PPO policy from {IMITATION_CHECKPOINT}")
    return model


def main():
    env = SubprocVecEnv(
        [make_env(rank) for rank in range(N_ENVS)], start_method="spawn"
    )
    env = VecMonitor(env)
    n_steps = 8192 // N_ENVS
    checkpoint = Path(f"{MODEL_NAME}.zip")

    if checkpoint.exists():
        print(f"Resuming PPO training from {checkpoint}")
        model = PPO.load(
            checkpoint,
            env=env,
            device="auto",
            learning_rate=1e-4,
            n_epochs=4,
            target_kl=0.03,
            ent_coef=0.01,
            tensorboard_log=f"./{MODEL_NAME}_dir/",
        )
    else:
        model = new_model(env, n_steps)

    callback = CheckpointCallback(
        save_freq=100_000 // N_ENVS,
        save_path=f"./{MODEL_NAME}_dir/",
        name_prefix="cr",
    )
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            reset_num_timesteps=False,
            callback=callback,
        )
    finally:
        model.save(MODEL_NAME)
        env.close()


if __name__ == "__main__":
    main()
