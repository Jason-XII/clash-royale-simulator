"""Continue spatial 6.5M for 5M decisions, with or without exact action masks.

From src/clasher_new: python train_legality.py [--control] [--seed 0]
"""
import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from masked_spatial import LegalPlacement, MaskedSpatialPolicy
from train_spatial import make_env as base_env


CHECKPOINT = Path("cr_spatial_scratch/cr_6500000_steps.zip")
ADDITIONAL_STEPS = 5_000_000
N_ENVS = 16


def make_env(rank, seed, control, opponent_factory=None):
    def factory():
        env = base_env(rank, seed)()
        if opponent_factory is not None:
            env.unwrapped.opponent = opponent_factory()
        return env if control else LegalPlacement(env)
    return factory


def main(control=False, seed=0, run_dir=None, checkpoint=CHECKPOINT,
         additional_steps=ADDITIONAL_STEPS, opponent_factory=None,
         experiment_info=None, arm_name=None):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {checkpoint}; run from src/clasher_new.")
    arm = arm_name or ("control" if control else "legal")
    run_dir = Path(run_dir or f"cr_spatial_{arm}")
    run_dir.mkdir(parents=True, exist_ok=False)
    env = VecMonitor(SubprocVecEnv(
        [make_env(rank, seed, control, opponent_factory) for rank in range(N_ENVS)],
        start_method="spawn"
    ), filename=str(run_dir / "monitor.csv"))
    try:
        changes = None if control else {
            "observation_space": env.observation_space,
            "policy_class": MaskedSpatialPolicy,
        }
        model = PPO.load(checkpoint, env=env, device="auto",
                         custom_objects=changes,
                         tensorboard_log=str(run_dir / "tensorboard"))
        model.set_random_seed(seed)
        experiment = {
            "arm": arm, "checkpoint": str(checkpoint.resolve()),
            "initial_steps": int(model.num_timesteps),
            "additional_steps": additional_steps, "seed": seed, "n_envs": N_ENVS,
            "policy": type(model.policy).__name__,
        }
        if experiment_info:
            experiment.update(experiment_info)
        (run_dir / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
        callback = CheckpointCallback(
            save_freq=max(100_000 // N_ENVS, 1),
            save_path=str(run_dir), name_prefix="cr",
        )
        try:
            model.learn(total_timesteps=additional_steps, callback=callback,
                        reset_num_timesteps=False, tb_log_name=arm)
        finally:
            model.save(run_dir / "final.zip")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(control=args.control, seed=args.seed)
