"""Continue the imitation policy with PPO and a fresh optimizer."""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from train_autoregressive import make_env


if __name__ == "__main__":
    n_envs = 16
    env = SubprocVecEnv(
        [make_env(rank) for rank in range(n_envs)], start_method="spawn"
    )
    env = VecMonitor(env)

    model = PPO.load(
        "cr_script_imitation_50k.zip",
        env=env,
        device="auto",
        n_steps=8192 // n_envs,
        batch_size=256,
        learning_rate=1e-4,
        n_epochs=4,
        target_kl=0.03,
        ent_coef=0.01,
        tensorboard_log="./cr_imitation_ppo_dir/",
    )

    # Discard Adam statistics learned from the supervised objective.
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(), lr=1e-4, **model.policy.optimizer_kwargs
    )
    assert len(model.policy.optimizer.state) == 0

    callback = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path="./cr_imitation_ppo_dir/",
        name_prefix="cr",
    )
    try:
        model.learn(total_timesteps=15_000_000, callback=callback)
    finally:
        model.save("cr_imitation_ppo")
        env.close()
